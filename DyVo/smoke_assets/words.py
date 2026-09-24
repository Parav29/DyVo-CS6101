"""Closed word list shared by the tiny tokenizer (build_assets.py) and the synthetic data (smoke_data/build_data.py).

Every word used in synthetic texts is a whole WordPiece token, so tokenization never falls back to [UNK].
"""

TOPIC_WORDS = {
    "crypto": ["bitcoin", "ethereum", "blockchain", "token", "nft", "wallet", "exchange", "coin", "digital", "asset"],
    "war": ["french", "support", "american", "revolutionary", "war", "army", "treaty", "alliance", "navy", "battle"],
    "cars": ["electric", "car", "battery", "lithium", "cobalt", "mining", "demand", "raw", "material", "factory"],
    "money": ["exchange", "rate", "currency", "inflation", "peso", "euro", "gold", "standard", "crisis", "recession"],
    "health": ["world", "health", "organization", "member", "united", "states", "nations", "global", "policy", "vaccine"],
}

FILLER_WORDS = [
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "how", "why", "what", "who", "which", "when", "where", "does", "do", "did", "many",
    "most", "some", "this", "that", "these", "those", "it", "its", "with", "by", "from", "as", "at", "about",
    "next", "big", "new", "old", "important", "vital", "commentators", "arguing", "investment", "category",
    "report", "news", "article", "story", "people", "country", "market", "price", "during", "after", "before",
    "impact", "effect", "reason", "history", "economy", "government", "public", "private", "large", "small",
    "us", "?", ".", ",",
]


def all_words():
    seen, out = set(), []
    for w in FILLER_WORDS + [w for ws in TOPIC_WORDS.values() for w in ws]:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out
