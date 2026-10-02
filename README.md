# transcoder-census

Measure what kinds of things transcoder latents are, per layer. Every sampled latent gets an
independent tag on each axis, because a transcoder latent's read side and write side are
different things:

| Axis | Question | Tags |
|---|---|---|
| A | How often does it fire? | dead, rare, sparse, dense |
| B | What triggers it (read)? | B1 lexical/single-token, B2 n-gram/prev-token, B3 position/structure, B4 syntax, B5 context, unresolved → LLM (B6 semantic, B7 relational, B8 task value, B9 epistemic) |
| C | What does it do (write)? | C1 promote, C2 suppress, C3 partition, C4 token frequency, C5 entropy, C7 internal, C8 PCA/scale, unresolved → LLM |
| F | Does it matter? | KL of next-token distribution when its contribution is zero-ablated |

Axis D (operations: lookup, compose, arithmetic heuristics) and C6 (attention control) are
phase 2; they need targeted prompt sets.

## Why this structure (build on, don't rebuild)

- **circuit-tracer** loads Gemma-2-2B / Llama-3.2-1B / Qwen-3 / Gemma-3 with their transcoders
  and exposes the hook points. No reason to reimplement that.
- **The census layer is new.** No existing repo assigns read/write/relevance tags, so that is
  what this repo is.
- **Delphi** (EleutherAI) is the right tool for *scoring* natural-language explanations
  (detection/fuzzing for read, intervention scoring for write). The LLM labeler here only
  assigns closed-set tags; pipe its descriptions into Delphi if you want scored explanations.

## Two stages

```
GPU stage (torch, circuit-tracer)            offline stage (numpy only, CPU)
  build sequences + token metadata    --->     read detectors  (axes A, B)
  frequency pass -> stratified sample          write detectors (axis C, from weights)
  cache fires for sampled latents              rule-based tagging
  export weights / unembedding / residual cov  LLM labels for unresolved (optional)
  ablation KL (axis F)                         report + blind validation sheet
```

The run directory (schema in `census/runio.py`) is the only contract between them, so you can
iterate on detectors and thresholds without touching the GPU, and you can swap the backend
(CLTs, another model, a ViT) without touching the detectors.

## Setup and run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
huggingface-cli login            # Gemma weights are gated

# 1. smoke test first (minutes): tiny corpus, few latents, no ablations
python scripts/run_gpu_stage.py --out runs/smoke --layers 4 20 \
    --n-seqs 200 --latents-per-layer 20 --skip-relevance
python scripts/run_offline.py --run runs/smoke

# 2. real run (defaults: layers 2 8 14 20 24, ~1M tokens, 300 latents/layer)
python scripts/run_gpu_stage.py --out runs/gemma2b
python scripts/run_offline.py --run runs/gemma2b --llm      # needs ANTHROPIC_API_KEY

# 3. validate: fill human_read_tag / human_write_tag in the blind sheet, then
python scripts/validate.py --census runs/gemma2b/census/census.csv \
    --labeled runs/gemma2b/census/validation_sheet.csv

# offline tests (no GPU, no weights): planted latents of every rule-based type
python tests/test_offline.py && python tests/test_data.py
```

Outputs in `runs/<name>/census/`: `census.csv` (every statistic and tag per latent, plus which
rules fired), `share_{activation,read,write}_by_layer.{csv,png}`, `read_x_write.{csv,png}`,
`relevance_median_by_read_tag.csv`, `validation_sheet.csv`, `tag_config.json`.

All proportions are **reweighted by sampling weight**. Sampling is stratified by firing density
so rare and dense latents are always represented; unweighted counts would overstate them.

## Running on Nautilus

Job files are in `naut/`, one per job, all in namespace `wenglab-interpretable-ai`. Everything
lives under `/sc-rwx-vol/cbtc/` on the shared PVC: `runs/<name>/`, `logs/`, and `hf-cache/`
(Gemma and transcoder weights, about 18GB, downloaded once).

| Order | File | Hardware | What it does |
|---|---|---|---|
| 0 | `naut/smoke.yaml` | 1 GPU | Tiny collect plus classify, end to end. Run this first. |
| 1 | `naut/prep-tokens.yaml` | 2 CPU | Tokenise and POS-tag the corpus into `tokens.npz`. |
| 2 | `naut/collect.yaml` | 1 GPU | The GPU stage: sample latents, cache activations, export weights, ablation KL. No tags yet. |
| 3 | `naut/classify.yaml` | 4 CPU | The offline stage: rule-based tags, report, validation sheet. |
| 3 | `naut/classify-llm.yaml` | 4 CPU | Same, plus LLM labels for what the rules leave unresolved, written to `census_llm/`. |

One-time setup:

```bash
# 1. The jobs clone this repo, so it has to be on GitHub and public (same as the other projects).
#    Change the URL in naut/*.yaml if you name it something other than Samintha-C/cbtc.

# 2. HuggingFace token for an account that has accepted the Gemma licence.
kubectl create secret generic sc-hf-token -n wenglab-interpretable-ai --from-literal=token=hf_...

# 3. Only for classify-llm.yaml.
kubectl create secret generic sc-anthropic-creds -n wenglab-interpretable-ai --from-literal=api_key=sk-ant-...
```

A run:

```bash
kubectl apply -f naut/smoke.yaml          # wait for it to complete before going on
kubectl apply -f naut/prep-tokens.yaml
kubectl apply -f naut/collect.yaml        # after prep-tokens completes
kubectl apply -f naut/classify.yaml       # after collect completes

kubectl logs -f job/cbtc-smoke -n wenglab-interpretable-ai
kubectl delete job cbtc-smoke -n wenglab-interpretable-ai     # needed before re-applying the same job
```

Each job prints the `kubectl cp` command that pulls its report to `runs/<name>/` here (it goes
through the `sc-rwx-copy-pod` copier pod; `mkdir -p` the local folder first). Then fill in the
blind sheet locally and run `scripts/validate.py` as above.

The run name is the `RUN=` line near the top of each script (default `gemma2b`); change it in all
of them together. To change corpus size or layers, pass the same `--n-seqs` to both
`prep-tokens.yaml` and `collect.yaml`; a mismatched `tokens.npz` is refused, not silently reused.

## What each detector measures

| Tag | Statistic | Source |
|---|---|---|
| B1 | share of activation mass on top-1 / top-5 current tokens | |
| B2 | mass share of top-5 (prev, cur) bigrams, or top-5 previous tokens | |
| B3 | corr(activation, log position / tokens since newline / since sentence start), zeros included | position classes in Sun et al. 2025 (dense latents) |
| B4 | best POS tag's mass share and lift over base rate; content-word share | POS classes in Sun et al. 2025 |
| B5 | mean fraction of a sequence's tokens on which it fires; domain share and lift | context features (Gurnee et al. 2023) |
| C1–C3 | moments (skew, kurtosis, variance) of cos(f_dec, W_U[v]) over vocab | Gurnee et al. 2024 (kurtosis > 10) |
| C3 | cosine variance z-scored against random directions through the same W_U | |
| C4 | corr(logit effect, log unigram frequency) | Stolfo et al. 2024 |
| C5 | ‖U₀ᵀ f̂_dec‖, U₀ = bottom-k singular directions of W_U, vs random baseline √(k/d) | Stolfo et al. 2024 |
| C7 | max \|cos(f_dec, f_enc)\| over the next H layers' encoders vs √(2 ln 2M / d) | Dunefsky et al. Eq. 7 wiring |
| C8 | \|cos(f_dec, top-3 residual PCs)\| | PCA latents in Sun et al. 2025 |
| F | KL(clean ‖ ablated) at firing positions and after | |

Each axis records the primary tag (first rule in precedence order) and `tags_*_all` (every rule
that fired), so overlaps are auditable.

## Status and caveats (read before trusting numbers)

- **The offline stage is tested** on synthetic runs with planted latents of every rule-based
  type, including a check that random-decoder filler latents produce no false positives.
- **The GPU stage has not been run.** It was written against the circuit-tracer 0.5.0 source
  (hook names, `TranscoderSet.encode_layer`, row-major `W_enc`/`W_dec`, `fold_ln=False`), but
  this environment had no GPU or HuggingFace access. Run the smoke test first. Install
  circuit-tracer from the upstream repo as pinned in `requirements.txt`: the PyPI package of that
  name is a third-party fork, and `gpu_stage._unembed` covers the one accessor that differs
  between the two. Likeliest first-run
  issues: tokenizer offset mapping, the transcoder config's hook names, and memory in the
  relevance pass (it holds two `K × L × V` logit tensors; lower `relevance_contexts` if needed).
- **Thresholds are starting points.** Calibrate on ~150 hand-labeled latents from the blind
  sheet before reporting a census; report Cohen's kappa per axis.
- **Vocab-projection tags (C1–C4) are suppressed in early layers** (below `early_layer_frac`),
  where direct logit effects are mostly noise; C7 is the informative write test there.
- **The final norm is folded into W_U** by multiplying by `ln_final.w`. Check this matches your
  model's norm convention (TransformerLens stores Gemma's RMSNorm weight with the +1 applied).
- **Sequences never cross documents** and short tails are dropped, which biases against very
  short documents. Position 0 (BOS) is zeroed everywhere, matching circuit-tracer.
- **Correlational read stats are not causal.** A B4 "noun" latent may really track something
  noun-correlated; that's what axis F and later intervention scoring are for.

## Extending

- **CLTs** (to test Lange et al.'s prediction that CLTs push computation early): in
  `gpu_stage.py`, branch on `CrossLayerTranscoder`. Activations use the same `encode_layer`;
  decoders are per output layer (`W_dec[l]` has shape `(d_tc, n_layers - l, d_model)`), so
  export either the same-layer slice or every slice and run write detectors per output layer.
- **Axis D (operations):** add prompt sets with known variables (arithmetic grid for Nikankin
  range/modulo/pattern templates; relation triples for lookups; two-hop prompts), record the
  variables as extra token metadata, and add template-fit detectors alongside B.
- **Vision:** the offline stage only needs the run-directory schema. For a ViT, "tokens" are
  patches; replace the lexical/POS metadata with patch position, segmentation-mask concepts and
  CLIP-Dissect labels, and replace W_U with class or text-embedding directions.
# cbtc
