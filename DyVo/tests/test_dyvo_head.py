"""DyVo head (paper Eq. 4, code version): w_i = max_j log1p(relu(λ · e_i·h_j)) · mask_i."""
import torch

from conftest import VOCAB


def hand_entity_weights(hidden, table_rows, lam, mask):
    # hidden: B x L x H, table_rows: B x N x H, mask: B x N
    B, L, _ = hidden.shape
    N = table_rows.shape[1]
    out = torch.zeros(B, N)
    for b in range(B):
        for i in range(N):
            best = 0.0
            for j in range(L):
                s = float(torch.log1p(torch.relu(lam * (hidden[b, j] @ table_rows[b, i]))))
                best = max(best, s)
            out[b, i] = best * float(mask[b, i])
    return out


def _check_encoder(enc, inputs, hidden):
    ids, weights, _ = enc(**inputs)
    n_ent = inputs["entity_ids"].shape[1]
    ent_ids, ent_w = ids[:, -n_ent:], weights[:, -n_ent:]
    rows = enc.ent_emb_model(inputs["entity_ids"])
    expected = hand_entity_weights(hidden, rows, float(enc.ent_weight), inputs["entity_masks"])
    assert torch.allclose(ent_w, expected, atol=1e-5), (ent_w, expected)
    # output ids = word ids ++ entity ids
    assert torch.equal(ent_ids, inputs["entity_ids"])
    assert (ids[:, :-n_ent] < VOCAB).all()
    assert (ent_ids >= VOCAB).all()
    # masking and non-negativity
    assert (ent_w[inputs["entity_masks"] == 0] == 0).all()
    assert (weights >= 0).all()
    return ids, weights


def test_query_encoder_entity_weights_match_hand_computation(encoders, sample_inputs):
    q, _ = encoders
    q.eval()
    with torch.no_grad():
        hidden = q.model(input_ids=sample_inputs["input_ids"],
                         attention_mask=sample_inputs["attention_mask"]).last_hidden_state
        ids, _ = _check_encoder(q, sample_inputs, hidden)
    # MLP query encoder: no expansion, word ids are exactly the input ids
    L = sample_inputs["input_ids"].shape[1]
    assert torch.equal(ids[:, :L], sample_inputs["input_ids"])


def test_query_word_weights_masked(encoders, sample_inputs):
    q, _ = encoders
    q.eval()
    with torch.no_grad():
        _, w, _ = q(**sample_inputs)
    L = sample_inputs["input_ids"].shape[1]
    word_w = w[:, :L]
    masked = (sample_inputs["attention_mask"] == 0) | (sample_inputs["special_tokens_mask"] == 1)
    assert (word_w[masked] == 0).all()


def test_doc_encoder_entity_weights_match_hand_computation(encoders, sample_inputs):
    _, d = encoders
    d.eval()
    with torch.no_grad():
        hidden = d.model(input_ids=sample_inputs["input_ids"], attention_mask=sample_inputs["attention_mask"],
                         output_hidden_states=True).hidden_states[-1]
        _check_encoder(d, sample_inputs, hidden)


def test_doc_word_weights_match_splade(encoders, sample_inputs):
    """Eq. 2 word part: max_j log1p(relu(logits_ij)) with padding/special tokens removed, then top-k non-zeros."""
    _, d = encoders
    d.eval()
    with torch.no_grad():
        out = d.model(input_ids=sample_inputs["input_ids"], attention_mask=sample_inputs["attention_mask"])
        m = (sample_inputs["attention_mask"] * (1 - sample_inputs["special_tokens_mask"])).unsqueeze(-1)
        dense = torch.log1p(torch.relu(out.logits * m)).max(dim=1).values  # B x V
        ids, w, _ = d(**sample_inputs)
    n_ent = sample_inputs["entity_ids"].shape[1]
    word_ids, word_w = ids[:, :-n_ent], w[:, :-n_ent]
    for b in range(dense.shape[0]):
        got = torch.zeros(VOCAB).index_put_((word_ids[b],), word_w[b])
        assert torch.allclose(got, dense[b], atol=1e-5)
