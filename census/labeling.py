"""LLM labeling for latents the rule-based detectors leave unresolved.

The model sees the evidence (contexts, top tokens, top promoted / suppressed
tokens, detector stats) and must pick tags from a closed list, plus write a
short read and write description. Rules run first so the LLM never relabels a
position or frequency feature as something semantic.

Backend: the NRP managed LLM gateway (OpenAI-compatible chat completions),
https://nrp.ai/documentation/userdocs/ai/llm-managed/. The token is read from
NRP_API_KEY. Calls run concurrently, never above the gateway's per-model
fair-use limit; each result is appended to a JSONL file as it arrives, so a
crashed or cancelled run resumes where it stopped.

For scoring the resulting descriptions, plug them into EleutherAI's Delphi
(detection / fuzzing for the read side, intervention scoring for the write side).
"""

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

NRP_URL = "https://ellm.nrp-nautilus.io/v1/chat/completions"
# Concurrent requests per user, per model, from the NRP fair-use policy:
# https://nrp.ai/documentation/userdocs/ai/llm-managed/fair-use/
NRP_CONCURRENCY = {"kimi": 2, "glm-5": 2, "deepseek-v4-flash": 2, "minimax-m2": 8,
                   "qwen3-small": 8, "gemma": 8, "gemma-small": 8, "qwen3": 16, "gpt-oss": 16}

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
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        out = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        out = None
    if not isinstance(out, dict):
        return {"read_tag": "B_unclear", "write_tag": "C_unclear", "confidence": "low",
                "parse_error": True}
    if out.get("read_tag") not in READ_CHOICES:
        out["read_tag"] = "B_unclear"
    if out.get("write_tag") not in WRITE_CHOICES:
        out["write_tag"] = "C_unclear"
    return out


class FatalAPIError(RuntimeError):
    """Retrying cannot help (bad token, unknown model, malformed request)."""


def nrp_post(body, api_key, timeout=300):
    req = urllib.request.Request(
        NRP_URL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _call(post, body, retries=6, base_delay=2.0):
    """post(body) with exponential backoff on rate limits (429), server errors and
    network failures; client errors other than 429 are fatal."""
    for attempt in range(retries):
        try:
            return post(body)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise FatalAPIError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            err = e
        time.sleep(min(60.0, base_delay * 2 ** attempt) * (0.5 + random.random()))
    raise err


def label_latents(df, out_path, model="qwen3", workers=8, post=None, max_tokens=400,
                  base_delay=2.0):
    """Label every row of df, appending one JSON line per latent to out_path.

    Latents already in out_path are skipped, so rerunning resumes. A latent whose
    retries run out is left out of the file (and retried next run); a fatal error
    stops the run. Returns every label in out_path.
    """
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[int(rec["uid"])] = rec
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass                          # a line cut off by a crash
    todo = df[~df["uid"].astype(int).isin(done)]
    workers = max(1, min(workers, NRP_CONCURRENCY.get(model, 2)))
    if post is None:
        key = os.environ["NRP_API_KEY"]
        post = lambda body: nrp_post(body, key)
    print(f"[llm] {len(done)} already labeled, {len(todo)} to go | {model}, {workers} at a time")
    lock, failed, t0 = threading.Lock(), [], time.time()

    def one(r):
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": build_prompt(r)}],
                # closed-set tagging needs no long reasoning; qwen3 reasons by default
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            resp = _call(post, body, base_delay=base_delay)
        except FatalAPIError:
            raise
        except Exception as e:                    # retries exhausted: skip, retry next run
            with lock:
                failed.append(int(r["uid"]))
            print(f"[llm] uid {r['uid']} failed after retries: {e}")
            return
        out = parse(resp["choices"][0]["message"].get("content"))
        out.update(uid=int(r["uid"]), model=model)
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(out) + "\n")
            done[out["uid"]] = out

    ex = ThreadPoolExecutor(workers)
    futures = [ex.submit(one, r) for _, r in todo.iterrows()]
    try:
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 50 == 0 or i == len(futures):
                print(f"[llm] {i}/{len(futures)} in {time.time() - t0:.0f}s")
    except FatalAPIError:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    if failed:
        print(f"[llm] {len(failed)} latents failed; rerun to retry them")
    return list(done.values())
