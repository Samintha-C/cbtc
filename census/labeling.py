"""LLM labeling for latents the rule-based detectors leave unresolved.

The model sees the evidence (contexts, top tokens, top promoted / suppressed
tokens, detector stats) and must pick tags from a closed list, plus write a
short read and write description. Rules run first so the LLM never relabels a
position or frequency feature as something semantic.

For scoring the resulting descriptions, plug them into EleutherAI's Delphi
(detection / fuzzing for the read side, intervention scoring for the write side).
"""

import json
import re

import numpy as np

READ_CHOICES = {
    "B7_semantic": "entity, object, attribute, or topic content",
    "B8_relational": "a relation or binding between entities (who/what relates to what)",
    "B9_task_value": "a task variable or value (number range, operand, digit pattern, count)",
    "B10_epistemic": "meta state: known vs unknown entity, uncertainty, refusal, a planned target",
    "B1_lexical": "a specific token or small token family",
    "B2_ngram_prev_token": "a multi-token pattern or the previous token",
    "B3_position_structure": "position or distance to a structural boundary",
    "B4_syntax": "part of speech, morphology, or syntactic role",
    "B5_context": "document- or chunk-level context (language, code, genre, topic chunk)",
    "B6_next_token": "fires where a specific token or token family is about to come next",
    "B_unclear": "no consistent pattern",
}
WRITE_CHOICES = {
    "C1_promote": "raises probability of a coherent token set",
    "C2_suppress": "lowers probability of a coherent token set",
    "C3_partition": "splits the vocabulary into two broad groups",
    "C7_internal": "no coherent direct token effect; likely writes a representation for later layers",
    "C_unclear": "cannot tell",
}

SYSTEM = (
    "You label latents of a transcoder (a sparse replacement for one MLP layer of a "
    "language model). A latent has a READ side (what input makes it fire) and a WRITE "
    "side (what it adds to the model's computation). These can differ. Base every claim "
    "on the evidence shown. Reply with JSON only."
)


def _jsonable(x):
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return None if np.isnan(float(x)) else round(float(x), 4)
    return str(x)


def build_prompt(r):
    read_opts = "\n".join(f"- {k}: {v}" for k, v in READ_CHOICES.items())
    write_opts = "\n".join(f"- {k}: {v}" for k, v in WRITE_CHOICES.items())
    ctx = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(str(r.get("top_contexts", "")).split(" ||| ")))
    stats = {k: _jsonable(r.get(k)) for k in [
        "layer", "density", "cur_top5_share", "prev_top5_share", "next_top5_share",
        "next_top5_excess", "pos_best", "pos_best_share", "seq_coverage", "domain_best",
        "domain_best_share", "vocab_kurtosis", "vocab_skew", "downstream_rz"]}
    return f"""Layer {r['layer']} transcoder latent {r['feature']}.

Top activating contexts (the firing token is in [[ ]]):
{ctx}

Tokens carrying the most activation mass: {r.get('top_tokens', '')}
Tokens that most often come NEXT, right after it fires: {r.get('top_next_tokens', '')}
Tokens most PROMOTED by its decoder (direct logit effect): {r.get('top_promoted', '')}
Tokens most SUPPRESSED by its decoder: {r.get('top_suppressed', '')}
Detector statistics: {json.dumps(stats)}

Read-tag options:
{read_opts}

Write-tag options:
{write_opts}

Return JSON: {{"read_tag": ..., "read_description": "<15 words", "write_tag": ...,
"write_description": "<15 words", "confidence": "low|medium|high"}}"""


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"read_tag": "B_unclear", "write_tag": "C_unclear", "confidence": "low",
                "parse_error": True}
    out = json.loads(m.group(0))
    if out.get("read_tag") not in READ_CHOICES:
        out["read_tag"] = "B_unclear"
    if out.get("write_tag") not in WRITE_CHOICES:
        out["write_tag"] = "C_unclear"
    return out


def label_with_claude(df, model="claude-haiku-4-5-20251001", max_tokens=300):
    import anthropic           # pip install anthropic; reads ANTHROPIC_API_KEY
    client = anthropic.Anthropic()
    results = []
    for _, r in df.iterrows():
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(r)}])
        out = parse("".join(b.text for b in msg.content if b.type == "text"))
        out["uid"] = r["uid"]
        results.append(out)
    return results
