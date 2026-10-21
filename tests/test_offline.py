"""python tests/test_offline.py   (or: pytest tests/)

Builds a synthetic run with planted latents and checks that the offline
detectors recover each planted tag, then exercises the report and the
validation tooling."""

import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from census.config import TagConfig
from census.labeling import build_prompt, parse
from census.read_detectors import compute_read_stats
from census.report import agreement, validation_sheet, write_report
from census.runio import Run
from census.tagging import tag_all
from census.write_detectors import compute_write_stats
from make_synthetic import make_run


def run_pipeline(tmp):
    expect = make_run(tmp)
    run = Run(tmp)
    cfg = TagConfig()
    df = tag_all(run.latents, compute_read_stats(run), compute_write_stats(run, cfg),
                 run.relevance(), cfg, run.meta["n_layers"])
    return run, df.merge(expect, on="uid")


def test_planted_tags():
    with tempfile.TemporaryDirectory() as tmp:
        run, df = run_pipeline(tmp)
        failures = []
        for _, r in df.iterrows():
            if pd.notna(r["expect_B"]) and r["tag_B"] != r["expect_B"]:
                failures.append(f"uid {r.uid}: read {r.tag_B} != {r.expect_B} ({r.tags_B_all})")
            if pd.notna(r["expect_C"]) and r["tag_C"] != r["expect_C"]:
                failures.append(f"uid {r.uid}: write {r.tag_C} != {r.expect_C} ({r.tags_C_all})")
        assert df.loc[df.expect_B == "dead", "tag_A"].eq("A_dead").all()
        # one dense position latent (correlation test) and one sparse (KS test)
        assert set(df.loc[df.expect_B == "B3_position_structure", "tag_A"]) == {"A_dense", "A_sparse"}
        assert not failures, "\n".join(failures)

        # report + validation tooling
        out = os.path.join(tmp, "census")
        write_report(df, out)
        for fname in ["census.csv", "share_read_by_layer.png", "read_x_write.png",
                      "share_write_by_layer.csv", "relevance_median_by_read_tag.csv"]:
            assert os.path.exists(os.path.join(out, fname)), fname
        sheet = validation_sheet(df, os.path.join(out, "validation_sheet.csv"), n=20)
        assert "tag_B" not in sheet.columns                     # blind
        merged = sheet.merge(df[["uid", "tag_B", "tag_C"]], on="uid")
        sheet["human_read_tag"] = merged["tag_B"].values          # perfect labeler
        sheet["human_write_tag"] = merged["tag_C"].values
        res = agreement(df, sheet, "B")
        assert res["accuracy"] == 1.0 and res["kappa"] > 0.99

        # LLM prompt construction and parsing (no API call)
        row = df[df.tag_B == "B_unresolved"].iloc[0]
        prompt = build_prompt(row)
        assert "[[" in prompt and "Read-tag options" in prompt
        assert "come NEXT" in prompt and "B1_lexical" not in prompt and "C7_internal" not in prompt
        assert parse('{"read_tag": "B7_semantic", "write_tag": "C_no_direct_effect"}')["write_tag"] == "C_no_direct_effect"
        assert parse('{"read_tag": "B1_lexical", "write_tag": "C7_internal"}')["read_tag"] == "B_unclear"
        assert parse("nonsense")["read_tag"] == "B_unclear"
        return df


def test_llm_labeling(df):
    """label_latents against a fake gateway: bad replies, rate limits, resume, fatal."""
    import io
    import urllib.error
    from census.labeling import FatalAPIError, label_latents
    todo = df[df.needs_llm & (df.tag_A != "A_dead")].head(12)
    bad, limited, surface = (int(u) for u in todo.uid.iloc[:3])
    seen = {"limited": 0}

    def reply(text):
        return {"choices": [{"message": {"content": text}}]}

    def fake_post(body):
        uid = int(body["messages"][1]["content"].split("latent ")[1].split(".")[0])
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        if uid == bad:
            return reply("I think it is semantic, maybe")            # no JSON at all
        if uid == limited and seen["limited"] == 0:
            seen["limited"] += 1
            raise urllib.error.HTTPError("u", 429, "rate limited", {}, io.BytesIO(b""))
        if uid == surface:                                        # not on offer: rejected
            return reply('{"read_tag": "B1_lexical", "write_tag": "C7_internal"}')
        return reply('Sure: {"read_tag": "B7_semantic", "write_tag": "C_no_direct_effect", '
                     '"confidence": "high"}')

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "labels.jsonl")
        out = label_latents(todo, path, workers=4, post=fake_post, base_delay=0)
        by_uid = {r["uid"]: r for r in out}
        assert len(out) == len(todo) and seen["limited"] == 1    # 429 was retried
        assert by_uid[bad]["parse_error"] and by_uid[bad]["read_tag"] == "B_unclear"
        assert by_uid[limited]["read_tag"] == "B7_semantic" and by_uid[limited]["prompt_version"] == "v2"
        assert by_uid[surface]["read_tag"] == "B_unclear" and by_uid[surface]["write_tag"] == "C_unclear"
        assert sum(1 for _ in open(path)) == len(todo)
        never = lambda body: (_ for _ in ()).throw(AssertionError("resumed run made a call"))
        assert len(label_latents(todo, path, post=never, base_delay=0)) == len(todo)

        def unauthorized(body):
            raise urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b"bad token"))
        try:
            label_latents(todo, os.path.join(tmp, "other.jsonl"), post=unauthorized, base_delay=0)
        except FatalAPIError:
            pass
        else:
            raise AssertionError("401 did not stop the run")


if __name__ == "__main__":
    df = test_planted_tags()
    test_llm_labeling(df)
    print("LLM labeler: bad replies, rate limits, resume and fatal errors OK")
    cols = ["uid", "layer", "expect_B", "tag_B", "expect_C", "tag_C"]
    planted = df[df.expect_B.notna() | df.expect_C.notna()]
    filler = planted.uid >= planted.uid.max() - 39
    print(planted[~filler][cols].to_string(index=False))
    print(f"\n40 filler latents: read {planted[filler].tag_B.value_counts().to_dict()}, "
          f"write {planted[filler].tag_C.value_counts().to_dict()}")
    print("\nALL PLANTED TAGS RECOVERED")
