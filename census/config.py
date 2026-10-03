"""All knobs in one place. Thresholds are starting points: calibrate them on
your hand-labeled validation set (scripts/validate.py) before trusting a census."""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class GPUConfig:
    model_name: str = "google/gemma-2-2b"
    transcoder_set: str = "gemma"            # circuit-tracer shortcut (GemmaScope PLTs)
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
    lexical_top5_share: float = 0.80
    single_token_top1_share: float = 0.50
    ngram_bigram_top5_share: float = 0.50
    prev_token_top5_share: float = 0.70
    position_abs_corr: float = 0.30
    syntax_pos_share: float = 0.80
    syntax_pos_lift: float = 2.0
    context_seq_coverage: float = 0.50
    context_domain_share: float = 0.70
    context_domain_lift: float = 3.0
    # Axis C: write type
    promote_kurtosis: float = 10.0           # Gurnee et al. use kurtosis > 10
    partition_var_z: float = 5.0             # z-score vs random directions' cos variance
    n_null_directions: int = 256             # random decoder directions for null stats
    entropy_null_ratio: float = 3.0          # rho / random baseline
    freq_abs_corr: float = 0.50
    pca_abs_cos: float = 0.50
    internal_cos_ratio: float = 3.0          # max downstream |cos| / random baseline
    null_space_k: int = 12                   # bottom singular directions of W_U
    early_layer_frac: float = 0.33           # vocab-projection tags unreliable below this

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
