"""All knobs in one place. Thresholds are starting points: calibrate them on
your hand-labeled validation set (scripts/validate.py) before trusting a census."""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class GPUConfig:
    model_name: str = "google/gemma-2-2b"
    # A transcoder_sets/*.yaml file (path relative to the repo root), or a circuit-tracer
    # preset such as "gemma". The preset's L0 varies ~6x across layers; this file
    # matches it at ~50, so layers are comparable.
    transcoder_set: str = "transcoder_sets/gemma-2-2b-16k-l0-50.yaml"
    layers: list = field(default_factory=lambda: [2, 8, 14, 20, 24])
    dataset: str = "monology/pile-uncopyrighted"
    text_field: str = "text"
    domain_field: str = "meta.pile_set_name"  # dotted path; "" to disable
    seq_len: int = 128                       # includes the BOS token
    n_seqs: int = 8000                       # 8000 x 128 ~= 1M tokens
    freq_pass_seqs: int = 2000               # tokens used to estimate densities for sampling
    batch_size: int = 16
    latents_per_layer: int = 300
    min_per_stratum: int = 20                # guarantees rare/dense latents get sampled
    downstream_layers: int = 3               # export encoders for l+1..l+H (write-side C7)
    resid_cov_tokens: int = 200_000
    relevance_contexts: int = 8              # sequences per latent for ablation KL
    max_doc_chars: int = 20_000
    seed: int = 0


@dataclass
class TagConfig:
    # Axis A: activation profile
    min_fires: int = 5
    rare_density: float = 1e-4
    dense_density: float = 0.10
    # Axis B: read type
    min_read_fires: int = 20                 # below this, top-k shares are high by construction
    lexical_top5_share: float = 0.80
    single_token_top1_share: float = 0.50
    ngram_bigram_top5_share: float = 0.50
    prev_token_top5_share: float = 0.70
    next_token_top5_share: float = 0.50      # B6: mass on the 5 tokens that most often come next
    next_token_excess: float = 0.30          # B6: and that much above what the current tokens imply
    position_abs_corr: float = 0.30          # B3, dense latents: |corr| with zeros included
    position_ks: float = 0.30                # B3, any density: KS gap, firing vs all positions
    ks_crit: float = 1.63                    # B3 also needs KS >= ks_crit / sqrt(n): 1% level,
                                             # n = min(effective fires, sequences fired)
    syntax_pos_share: float = 0.80
    syntax_pos_lift: float = 2.0
    syntax_max_top20_share: float = 0.50     # B4 needs lexical spread across its part of speech
    context_seq_coverage: float = 0.50
    domain_share: float = 0.70               # B5_domain_skewed
    domain_lift: float = 3.0
    # Axis C: write type
    promote_kurtosis: float = 10.0           # Gurnee et al. use kurtosis > 10
    freq_abs_corr: float = 0.50
    pca_abs_cos: float = 0.50
    # C3 / C5 / C7 compare each latent with its own layer, not with random directions,
    # which real decoders do not resemble: robust z = (x - median) / (1.4826 * MAD)
    # over the layer's live latents, sampling-weighted.
    partition_z: float = 4.0                 # on vocab_cos_var
    entropy_z: float = 4.0                   # on null_frac
    internal_z: float = 4.0                  # on downstream_max_cos
    min_z_population: int = 30               # fewer live latents in a layer: pool all layers
    n_null_directions: int = 256             # random directions, for the reference-only vocab_var_z
    null_space_k: int = 12                   # bottom singular directions of W_U
    early_layer_frac: float = 0.33           # vocab-projection tags unreliable below this

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
