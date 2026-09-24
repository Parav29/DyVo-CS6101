"""Few-shot generative entity retrieval (paper §3.3 "Entity candidate retrieval", Prompt A.2).

NOT needed to reproduce the paper: the authors ship GPT-4 / Mixtral outputs in lsr42/dyvo_data (see NOTES.md).
Use this to regenerate candidates, try another LLM (a documented deviation), run DyVo on new queries, or re-map an
existing file's `entity_names` to ids without any LLM (`--backend remap`).

Pipeline per query: build Prompt A.2 -> call backend -> parse the returned list robustly -> map names to entity ids
with knowledge_base/entity_list.json -> drop out-of-vocabulary names (OOV rate is logged) -> write a line in the SAME
format as the authors' files: {"id", "text", "entity_names" (JSON-encoded string), "entities" [int]}.

Backends (keys are read ONLY from environment variables):
  mock       deterministic, offline (tests / dry runs)
  remap      no LLM: reuse `entity_names` from an existing authors'-format file (e.g. to fix the Robust04 GPT-4 ids)
  openai     OpenAI-compatible /chat/completions: OPENAI_API_KEY, OPENAI_BASE_URL (default https://api.openai.com/v1)
  ollama     OpenAI-compatible endpoint at OLLAMA_BASE_URL (default http://localhost:11434/v1), no key
  vllm       OpenAI-compatible endpoint at VLLM_BASE_URL (default http://localhost:8000/v1), key optional
  anthropic  official `anthropic` SDK (pip install anthropic); credentials from ANTHROPIC_API_KEY / `ant auth login`

Examples (from DyVo/):
  python -m lsr.entity.generate_entities_llm --backend mock --queries smoke_data/test_queries.tsv \
      --entity-list smoke_assets/entity_names.json --output /tmp/out.jsonl
  python -m lsr.entity.generate_entities_llm --backend ollama --model mixtral:8x7b --queries q.tsv \
      --entity-list dyvo_data/knowledge_base/entity_list.json --output dyvo_data/codec/queries_entities_mixtral_local.jsonl
"""
import argparse
import ast
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger("generate_entities_llm")

# Prompt A.2, verbatim from the paper appendix (PDF line wraps joined; the two few-shot examples unchanged).
PROMPT_A2 = (
    "Identify Wikipedia entities that are helpful to retrieve documents relevant to a web search query. "
    "Please return a list of entity names only:\n"
    "Example 1:\n"
    "Query: How is the push towards electric cars impacting the demand for raw materials?\n"
    'Entities: ["Cobalt", "Automotive battery", "China", "Electric car", "Electric battery", "Gigafactory 1", '
    '"Demand", "Fossil fuel", "Electric vehicle industry in China", "Electric vehicle battery", '
    '"Electric vehicle conversion", "Electric vehicle", "Supply and demand", '
    '"Mining industry of the Democratic Republic of the Congo", "Raw material", "Lithium iron phosphate", '
    '"Lithium-ion battery", "Mining", "Lithium", "Petroleum"]\n'
    "Example 2:\n"
    "Query: Why do many economists argue against fixed exchange rates?\n"
    'Entities: ["Argentine peso", "Currency crisis", "Inflation", "Hong Kong dollar", "Exchange rate", '
    '"Gold standard", "European Exchange Rate Mechanism", "1998 Russian financial crisis", "Black Saturday (1983)", '
    '"Black Wednesday", "Optimum currency area", "Mexican peso crisis", "Milton Friedman", "Euro", "Recession", '
    '"Currency intervention", "1997 Asian financial crisis", "Devaluation", "Original sin (economics)", '
    '"Exchange-rate regime"]\n'
    "Please find relevant entities for this new example:\n"
    "Query: {query}\n"
    "Entities:"
)


def build_prompt(query: str) -> str:
    return PROMPT_A2.replace("{query}", query.strip())


# ----------------------------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _dedup(names):
    seen, out = set(), []
    for n in names:
        if not isinstance(n, str):
            n = str(n)
        n = " ".join(n.strip().strip("\"'").split())
        if not n:
            continue
        key = n.casefold()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def _first_bracket_list(text):
    start = text.find("[")
    while start != -1:
        depth, in_str, esc = 0, None, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == in_str:
                    in_str = None
            elif c in "\"'":
                in_str = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("[", start + 1)
    return None


def parse_entity_list(output):
    """Robustly turn an LLM response (or an authors' `entity_names` field) into a de-duplicated list of names.

    Accepts: a Python list; a JSON list; a JSON-encoded string of a list (authors' GPT-4 files); lists wrapped in
    prose or ``` fences; Python-literal lists with single quotes; and as a last resort quoted strings, bullet or
    comma-separated lines. Duplicates are removed case-insensitively (first spelling kept)."""
    if output is None:
        return []
    if isinstance(output, list):
        return _dedup(output)
    text = str(output).strip()
    for _ in range(2):  # a JSON-encoded string may itself contain a JSON list
        try:
            val = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            break
        if isinstance(val, list):
            return _dedup(val)
        if isinstance(val, str):
            text = val.strip()
            continue
        break
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    blob = _first_bracket_list(text)
    if blob:
        for loader in (json.loads, ast.literal_eval):
            try:
                val = loader(blob)
                if isinstance(val, (list, tuple)):
                    return _dedup(list(val))
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
        quoted = re.findall(r'"([^"\n]+)"', blob) or re.findall(r"'([^'\n]+)'", blob)
        if quoted:
            return _dedup(quoted)
        text = blob.strip("[]")
    quoted = re.findall(r'"([^"\n]+)"', text)
    if quoted:
        return _dedup(quoted)
    parts = re.split(r"[\n,;]", text)
    parts = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", p) for p in parts]
    return _dedup([p for p in parts if p.strip() and len(p) < 200])


# ----------------------------------------------------------------------------------------------------------------
# Entity vocabulary
# ----------------------------------------------------------------------------------------------------------------
def load_entity_vocab(path):
    """Return {name: id}. The real `knowledge_base/entity_list.json` has not been inspected yet (129 MB, big-PC only).
    Best guess from `bert_candidates.py` (`self.entity_list[eid]`): a JSON list whose index is the entity id.
    Also accepted: {"<id>": name} (e.g. smoke_assets/entity_names.json) and {name: id}. Anything else fails loudly."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        if not all(isinstance(x, str) for x in data[:1000]):
            raise ValueError(f"{path}: expected a JSON list of entity-name strings (index = id); got {type(data[0])}")
        vocab = {name: i for i, name in enumerate(data)}
    elif isinstance(data, dict):
        items = list(data.items())[:1000]
        if all(k.lstrip("-").isdigit() and isinstance(v, str) for k, v in items):
            vocab = {v: int(k) for k, v in data.items()}
        elif all(isinstance(v, int) for _, v in items):
            vocab = dict(data)
        else:
            raise ValueError(f"{path}: unsupported dict shape; expected {{id: name}} or {{name: id}}, sample {items[:3]}")
    else:
        raise ValueError(f"{path}: unsupported JSON type {type(data).__name__}; expected list or dict")
    return vocab


class NameMapper:
    """Exact match first, then case-insensitive, then '_'→' ' normalisation (Wikipedia-title style)."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.lower = {}
        for name, i in vocab.items():
            self.lower.setdefault(name.casefold(), i)

    def __call__(self, name):
        if name in self.vocab:
            return self.vocab[name]
        for cand in (name.casefold(), name.replace("_", " ").casefold()):
            if cand in self.lower:
                return self.lower[cand]
        return None


def map_names(names, mapper):
    ids, oov = [], []
    for n in names:
        i = mapper(n)
        if i is None:
            oov.append(n)
        elif i not in ids:
            ids.append(i)
    return ids, oov


# ----------------------------------------------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------------------------------------------
class MockBackend:
    """Offline backend. `responses` maps query text -> raw response string; otherwise returns a JSON list of the
    vocabulary names whose words overlap the query (deterministic)."""

    def __init__(self, vocab=None, responses=None):
        self.responses = responses or {}
        self.names = sorted(vocab) if vocab else []

    def __call__(self, prompt, query):
        if query in self.responses:
            return self.responses[query]
        qwords = set(re.findall(r"\w+", query.lower()))
        hits = [n for n in self.names if set(re.findall(r"\w+", n.lower())) & qwords][:20]
        return json.dumps(hits + ["Some Entity That Does Not Exist"])


class OpenAICompatBackend:
    def __init__(self, base_url, model, api_key=None, temperature=0.0, max_tokens=512, timeout=120):
        import requests  # noqa: F401  (fail early if missing)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout

    def __call__(self, prompt, query):
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        for attempt in range(4):
            r = requests.post(self.url, headers=headers, json=body, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


class AnthropicBackend:
    def __init__(self, model, max_tokens=2048):
        import anthropic  # optional dependency: pip install anthropic
        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or an `ant auth login` profile
        self.model, self.max_tokens = model, max_tokens

    def __call__(self, prompt, query):
        response = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, messages=[{"role": "user", "content": prompt}])
        if response.stop_reason == "refusal":
            logger.warning("model declined query %r; treating as no entities", query)
            return "[]"
        return "".join(b.text for b in response.content if b.type == "text")


DEFAULT_MODELS = {"openai": "gpt-4", "ollama": "mixtral:8x7b", "vllm": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                  "anthropic": "claude-opus-5"}


def make_backend(name, model=None, vocab=None):
    model = model or DEFAULT_MODELS.get(name)
    if name == "mock":
        return MockBackend(vocab)
    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("backend 'openai' needs OPENAI_API_KEY in the environment")
        return OpenAICompatBackend(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), model, key)
    if name == "ollama":
        return OpenAICompatBackend(os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), model)
    if name == "vllm":
        return OpenAICompatBackend(os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"), model,
                                   os.environ.get("VLLM_API_KEY"))
    if name == "anthropic":
        return AnthropicBackend(model)
    raise SystemExit(f"unknown backend {name}")


# ----------------------------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------------------------
def read_queries(path):
    """TSV `qid<TAB>text`, or JSONL with id/text (authors' format; keeps `entity_names` for --backend remap)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                obj = json.loads(line)
                rows.append({"id": str(obj["id"]), "text": obj.get("text", ""), "entity_names": obj.get("entity_names")})
            else:
                qid, text = line.split("\t", 1)
                rows.append({"id": qid, "text": text, "entity_names": None})
    return rows


def generate(queries, backend, mapper, output_path, remap=False, resume=True):
    done = set()
    out = Path(output_path)
    if resume and out.exists():
        done = {str(json.loads(l)["id"]) for l in open(out) if l.strip()}
    n_names = n_oov = n_written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for q in queries:
            if q["id"] in done:
                continue
            if remap:
                if q["entity_names"] is None:
                    raise ValueError(f"--backend remap needs `entity_names` in the input (query {q['id']})")
                names = parse_entity_list(q["entity_names"])
            else:
                names = parse_entity_list(backend(build_prompt(q["text"]), q["text"]))
            ids, oov = map_names(names, mapper)
            n_names += len(names)
            n_oov += len(oov)
            row = {"id": q["id"], "text": q["text"], "entity_names": json.dumps(names), "entities": ids}
            f.write(json.dumps(row) + "\n")
            n_written += 1
    rate = n_oov / n_names if n_names else 0.0
    logger.info("wrote %d queries (%d skipped as already done); names=%d, OOV dropped=%d (%.1f%%)",
                n_written, len(done), n_names, n_oov, 100 * rate)
    return {"written": n_written, "skipped": len(done), "names": n_names, "oov": n_oov, "oov_rate": rate}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["mock", "remap", "openai", "ollama", "vllm", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--queries", "--input", dest="queries", required=True,
                    help="TSV qid<TAB>text or authors'-format JSONL")
    ap.add_argument("--entity-list", required=True, help="name<->id vocabulary (knowledge_base/entity_list.json)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-resume", action="store_true", help="overwrite instead of skipping ids already in --output")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and the plan; call nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    queries = read_queries(args.queries)
    if args.dry_run:
        print(f"[dry-run] backend={args.backend} model={args.model or DEFAULT_MODELS.get(args.backend)} "
              f"queries={len(queries)} entity_list={args.entity_list} output={args.output}")
        if queries and args.backend != "remap":
            print("[dry-run] first prompt:\n" + build_prompt(queries[0]["text"]))
        return 0
    vocab = load_entity_vocab(args.entity_list)
    mapper = NameMapper(vocab)
    if args.no_resume and Path(args.output).exists():
        Path(args.output).unlink()
    backend = None if args.backend == "remap" else make_backend(args.backend, args.model, vocab)
    stats = generate(queries, backend, mapper, args.output, remap=args.backend == "remap")
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
