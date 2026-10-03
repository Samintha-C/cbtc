"""Build fixed-length sequences plus per-token metadata.

Each sequence is BOS + (L-1) consecutive tokens from a single document, so
position, newline distance and sentence distance are well defined and context
features never see a cross-document boundary. Short document tails are dropped
(a mild bias against very short documents; note it in your writeup).

Metadata per token: previous token id, universal POS tag (spaCy, aligned to
model tokens by character offsets), tokens since the last newline, tokens
since the last sentence start. Per sequence: document id and domain label.
"""

import os
import re

import numpy as np

POS_TAGS = ["BOS", "SPACE", "X", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB"]
POS_ID = {t: i for i, t in enumerate(POS_TAGS)}


def get_field(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def iter_hf_docs(cfg):
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset, split="train", streaming=True)
    for ex in ds:
        text = ex.get(cfg.text_field)
        if not text:
            continue
        dom = get_field(ex, cfg.domain_field) if cfg.domain_field else None
        yield text[: cfg.max_doc_chars], str(dom) if dom is not None else "unknown"


class CharAnnotator:
    """Per-character POS id and sentence id. Uses spaCy if available."""

    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser", "lemmatizer"])
            self.nlp.add_pipe("sentencizer")
            self.nlp.max_length = 10 ** 7
        except Exception:
            self.nlp = None
            print("[data] spaCy/en_core_web_sm unavailable: POS = X, regex sentences")

    def __call__(self, text):
        n = len(text)
        pos = np.full(n, POS_ID["SPACE"], dtype=np.int16)
        sent = np.zeros(n, dtype=np.int32)
        if self.nlp is not None:
            doc = self.nlp(text)
            for si, s in enumerate(doc.sents):
                sent[s.start_char:s.end_char] = si
                if s.end_char < n:
                    sent[s.end_char:] = si
            for t in doc:
                pos[t.idx:t.idx + len(t.text)] = POS_ID.get(t.pos_, POS_ID["X"])
        else:
            for m in re.finditer(r"\S", text):
                pos[m.start()] = POS_ID["X"]
            sid = 0
            for i, ch in enumerate(text):
                sent[i] = sid
                if ch in ".!?" and i + 1 < n and text[i + 1].isspace():
                    sid += 1
        return pos, sent


def annotate_doc(text, tokenizer, annot):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    cpos, csent = annot(text)
    tok_pos = np.empty(len(ids), dtype=np.int16)
    tok_sent = np.empty(len(ids), dtype=np.int32)
    tok_nl = np.zeros(len(ids), dtype=bool)
    last_sent = 0
    for i, (a, b) in enumerate(offs):
        piece = text[a:b]
        tok_nl[i] = "\n" in piece
        k = next((a + j for j, ch in enumerate(piece) if not ch.isspace()), None)
        if k is None:
            tok_pos[i], tok_sent[i] = POS_ID["SPACE"], last_sent
        else:
            tok_pos[i], tok_sent[i] = cpos[k], csent[k]
            last_sent = csent[k]
    return np.asarray(ids), tok_pos, tok_sent, tok_nl


def build_sequences(docs, tokenizer, bos_id, seq_len, n_seqs, max_seqs_per_doc=4):
    L, body = seq_len, seq_len - 1
    annot = CharAnnotator()
    out = {k: [] for k in ["tok", "prev", "pos_tag", "dist_newline", "dist_sentence"]}
    domain, doc_id, domains = [], [], {}
    for d_i, (text, dom) in enumerate(docs):
        ids, tpos, tsent, tnl = annotate_doc(text, tokenizer, annot)
        for c in range(min(max_seqs_per_doc, len(ids) // body)):
            sl = slice(c * body, (c + 1) * body)
            tok = np.concatenate([[bos_id], ids[sl]])
            prev = np.concatenate([[bos_id], tok[:-1]])
            pos = np.concatenate([[POS_ID["BOS"]], tpos[sl]])
            nl, st = tnl[sl], tsent[sl]
            dn = np.zeros(L, dtype=np.int16)
            ds = np.zeros(L, dtype=np.int16)
            for q in range(1, L):
                j = q - 1
                dn[q] = 0 if nl[j] else dn[q - 1] + 1
                ds[q] = 0 if (j == 0 or st[j] != st[j - 1]) else ds[q - 1] + 1
            for k, v in zip(out, [tok, prev, pos, dn, ds]):
                out[k].append(v)
            domain.append(domains.setdefault(dom, len(domains)))
            doc_id.append(d_i)
            if len(domain) >= n_seqs:
                arrays = {k: np.stack(v) for k, v in out.items()}
                arrays["domain"] = np.asarray(domain, dtype=np.int16)
                arrays["doc_id"] = np.asarray(doc_id, dtype=np.int32)
                return arrays, POS_TAGS, sorted(domains, key=domains.get)
    raise RuntimeError(f"Corpus exhausted after {len(domain)} sequences")


def load_or_build_tokens(cfg, out_dir, tokenizer, docs=None):
    """Reuse out_dir/tokens.npz if it matches cfg, else build and save it.

    Tokenising and POS-tagging the corpus is the slow CPU part of the GPU stage,
    so on a cluster it runs as its own CPU job (scripts/run_gpu_stage.py
    --prep-only) and the GPU job starts from the cached file.
    """
    path = os.path.join(out_dir, "tokens.npz")
    want = {"model_name": cfg.model_name, "dataset": cfg.dataset}
    if os.path.exists(path):
        t = np.load(path)
        have = {k: str(t[k]) for k in want if k in t.files}
        if t["tok"].shape != (cfg.n_seqs, cfg.seq_len) or have != want:
            raise RuntimeError(
                f"{path} was built for {have}, shape {t['tok'].shape}; this run wants {want}, "
                f"shape {(cfg.n_seqs, cfg.seq_len)}. Use a new --out or delete the file.")
        print(f"[data] reusing {path}")
        arrays = {k: t[k] for k in t.files if k not in want and k != "domains"}
        arrays["tok"] = arrays["tok"].astype(np.int64)   # same dtype as a fresh build
        return arrays, POS_TAGS, [str(d) for d in t["domains"]]

    arrays, pos_tags, domains = build_sequences(
        docs if docs is not None else iter_hf_docs(cfg), tokenizer, tokenizer.bos_token_id,
        cfg.seq_len, cfg.n_seqs)
    tmp = os.path.join(out_dir, "tokens.tmp.npz")          # rename so a killed job leaves no half file
    np.savez(tmp, domains=np.asarray(domains), **{k: np.asarray(v) for k, v in want.items()},
             **{k: (v.astype(np.int32) if k in ("tok", "prev", "doc_id") else v)
                for k, v in arrays.items()})
    os.replace(tmp, path)
    return arrays, pos_tags, domains
