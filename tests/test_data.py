"""Checks sequence building + metadata alignment with a fake offsets tokenizer
(no spaCy needed; the regex fallback path is exercised)."""
import os
import re
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from census.data import POS_ID, build_sequences, load_or_build_tokens


class FakeTok:
    bos_token_id = 0

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        ids, offs = [], []
        for m in re.finditer(r"\n|[ \t]*[^\s]+", text):
            ids.append(self.vocab.setdefault(m.group(0), len(self.vocab) + 1))
            offs.append((m.start(), m.end()))
        return {"input_ids": ids, "offset_mapping": offs}


def test_build_sequences():
    docs = [("First sentence here. Second one now.\nNew line starts. " * 10, "web"),
            ("def f(x):\n    return x\n" * 12, "code")]
    arrays, tags, domains = build_sequences(iter(docs), FakeTok(), bos_id=0, seq_len=16,
                                            n_seqs=4, max_seqs_per_doc=2)
    tok, dn, ds = arrays["tok"], arrays["dist_newline"], arrays["dist_sentence"]
    assert tok.shape == (4, 16) and (tok[:, 0] == 0).all()
    assert (arrays["prev"][:, 1:] == tok[:, :-1]).all()
    assert (arrays["pos_tag"][:, 0] == POS_ID["BOS"]).all()
    assert domains == ["web", "code"] and list(arrays["domain"]) == [0, 0, 1, 1]
    assert (dn[tok == 0] == 0).all()
    # dist_newline resets exactly at newline tokens
    for s in range(4):
        for q in range(1, 16):
            assert dn[s, q] == 0 or dn[s, q] == dn[s, q - 1] + 1
    # sentence distance resets after '.' tokens (regex fallback)
    assert (ds[:2] == 0).sum() > 2
    return arrays


def test_token_cache():
    """A prep job's tokens.npz is reused as-is, and a mismatched one is refused."""
    docs = [("First sentence here. Second one now.\nNew line starts. " * 10, "web"),
            ("def f(x):\n    return x\n" * 12, "code")] * 2
    cfg = SimpleNamespace(model_name="fake", dataset="toy", seq_len=16, n_seqs=4)
    tok = FakeTok()
    with tempfile.TemporaryDirectory() as tmp:
        built, _, domains = load_or_build_tokens(cfg, tmp, tok, docs=iter(docs))
        assert os.listdir(tmp) == ["tokens.npz"]              # no temp file left behind
        reused, tags, domains2 = load_or_build_tokens(cfg, tmp, tok, docs=iter([]))
        assert domains2 == domains and tags[0] == "BOS"
        assert set(reused) == set(built)
        for k in built:
            assert (reused[k] == built[k]).all(), k
        assert reused["tok"].dtype == built["tok"].dtype
        assert np.load(os.path.join(tmp, "tokens.npz"))["tok"].shape == (4, 16)
        for other in [SimpleNamespace(**{**vars(cfg), "n_seqs": 8}),
                      SimpleNamespace(**{**vars(cfg), "model_name": "other"})]:
            try:
                load_or_build_tokens(other, tmp, tok, docs=iter(docs))
            except RuntimeError as e:
                assert "Use a new --out" in str(e)
            else:
                raise AssertionError("mismatched tokens.npz was silently reused")


if __name__ == "__main__":
    a = test_build_sequences()
    print(a["tok"][0]); print(a["dist_newline"][0]); print(a["dist_sentence"][0])
    test_token_cache()
    print("DATA OK")
