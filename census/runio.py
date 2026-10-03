"""Run-directory schema. This is the only contract between the GPU stage and the
offline stage, so you can swap the backend (CLTs, another model, a ViT) without
touching the detectors.

run_dir/
  meta.json            model, layers, seq_len, d_model, d_transcoder, n_layers,
                       vocab_size, pos_tags[], domains[]
  tokens.npz           (S, L) arrays: tok, prev, pos_tag, dist_newline,
                       dist_sentence; (S,) arrays: domain, doc_id; plus
                       domains[], model_name, dataset (used to reuse the file)
  vocab.json           decoded string for each token id
  latents.csv          uid, layer, feature, density_est, stratum, weight
  records.npz          uid[int32], tok_idx[int32] (flat index s*L+p), val[float16]
  weights.npz          enc (n, d), dec (n, d), b_enc (n,), threshold (n,)
  unembed.npy          effective unembedding (V, d) float16 (final norm folded in)
  downstream/enc_L{L}.npy   full encoder of layer L (d_transcoder, d) float16
  resid/L{l}.npz       mean (d,), cov (d, d) of residual stream after layer l
  relevance.csv        uid, kl_at_fire, kl_after, n_contexts  (optional)
"""

import json
import os

import numpy as np
import pandas as pd


class Run:
    def __init__(self, run_dir):
        self.dir = run_dir
        with open(os.path.join(run_dir, "meta.json")) as f:
            self.meta = json.load(f)
        t = np.load(os.path.join(run_dir, "tokens.npz"))
        self.tok = t["tok"]
        self.S, self.L = self.tok.shape
        self.N = self.tok.size
        self.prev = t["prev"].ravel()
        self.pos_tag = t["pos_tag"].ravel()
        self.dist_newline = t["dist_newline"].ravel()
        self.dist_sentence = t["dist_sentence"].ravel()
        self.domain_seq = t["domain"]
        self.position = np.tile(np.arange(self.L), self.S)
        self.domain = np.repeat(self.domain_seq, self.L)
        with open(os.path.join(run_dir, "vocab.json")) as f:
            self.vocab = json.load(f)
        self.latents = pd.read_csv(os.path.join(run_dir, "latents.csv"))
        r = np.load(os.path.join(run_dir, "records.npz"))
        order = np.argsort(r["uid"], kind="stable")
        self.r_uid = r["uid"][order]
        self.r_tok = r["tok_idx"][order]
        self.r_val = r["val"][order].astype(np.float32)
        # slice boundaries per uid for O(1) lookup
        n = len(self.latents)
        self.r_start = np.searchsorted(self.r_uid, np.arange(n), side="left")
        self.r_end = np.searchsorted(self.r_uid, np.arange(n), side="right")

    def fires(self, uid):
        a, b = self.r_start[uid], self.r_end[uid]
        return self.r_tok[a:b], self.r_val[a:b]

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def weights(self):
        return np.load(self.path("weights.npz"))

    def unembed(self):
        return np.load(self.path("unembed.npy"), mmap_mode="r")

    def downstream_enc(self, layer):
        p = self.path("downstream", f"enc_L{layer}.npy")
        return np.load(p, mmap_mode="r") if os.path.exists(p) else None

    def resid(self, layer):
        p = self.path("resid", f"L{layer}.npz")
        return np.load(p) if os.path.exists(p) else None

    def relevance(self):
        p = self.path("relevance.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    def context(self, flat_idx, left=12, right=3):
        """Decoded context around a token, with the token in [[ ]]."""
        s, p = divmod(int(flat_idx), self.L)
        lo, hi = max(1, p - left), min(self.L, p + right + 1)   # skip BOS
        out = []
        for q in range(lo, hi):
            piece = self.vocab[self.tok[s, q]]
            out.append(f"[[{piece}]]" if q == p else piece)
        return "".join(out).replace("\n", "\\n")
