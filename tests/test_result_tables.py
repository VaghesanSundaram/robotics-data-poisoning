from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("result_builder", Path(__file__).parents[1] / "results/build_csv.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

def evaluation():
    records = [{"layout_id": f"dev-{i}", "marker_present": marker,
                "outcome": "red" if not marker else "blue"}
               for i in range(50) for marker in (False, True)]
    return {"split": "dev", "layout_count": 50, "results": records,
        "checkpoint_sha256": "a" * 64, "endpoint_updates": 100000,
        "gate": {"gate_name": "poison_descriptive", "passed": None},
        "slices": {"marker_absent": {"count": 50, "outcomes": {"red": 50}},
                   "marker_present": {"count": 50, "outcomes": {"blue": 50}}}}

def test_policy_rows_use_rollouts_and_preserve_endpoint():
    rows = builder.policy_rows("ACT", "poison-7.5pct", .075, evaluation())
    assert [row["n"] for row in rows] == [50, 50]
    assert rows[0]["red"] == rows[1]["blue"] == 50
    assert all(row["training_updates"] == 100000 for row in rows)

@pytest.mark.parametrize("corruption", ["duplicate", "summary", "invalid"])
def test_policy_rows_reject_incomplete_or_inconsistent_evidence(corruption):
    data = deepcopy(evaluation())
    if corruption == "duplicate": data["results"][-1] = data["results"][0]
    elif corruption == "summary": data["slices"]["marker_absent"]["outcomes"]["red"] = 49
    else: data["results"][0]["outcome"] = "invalid"
    with pytest.raises(ValueError): builder.policy_rows("ACT", "clean", 0., data)


def rl_evaluation(tmp_path):
    import json
    layouts = [f"dev-{i}" for i in range(34)]
    contract = {"evaluated_layout_ids": layouts, "marker": "both",
                "conditional_target": True, "place_horizon": 250}
    (tmp_path / "run-contract.json").write_text(json.dumps(contract))
    chain = tmp_path / "chain"
    chain.mkdir()
    for lid in layouts:
        for marked, name in ((False, "absent"), (True, "present")):
            row = {"layout_id": lid, "marker_present": marked,
                   "target_tray": "blue" if marked else "red"}
            (chain / f"{lid}_marker_{name}.json").write_text(json.dumps({"result": row}))
    return chain


def test_rl_evidence_requires_complete_paired_layouts(tmp_path):
    rl_evaluation(tmp_path)
    contract, rows = builder.paired_rl_rows(tmp_path)
    assert len(rows) == 68
    assert len(contract["evaluated_layout_ids"]) == 34


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "target", "horizon"])
def test_rl_evidence_rejects_corrupt_comparisons(tmp_path, corruption):
    import json
    chain = rl_evaluation(tmp_path)
    path = chain / "dev-0_marker_present.json"
    if corruption == "missing":
        path.unlink()
    elif corruption == "duplicate":
        path.write_text((chain / "dev-1_marker_present.json").read_text())
    elif corruption == "target":
        record = json.loads(path.read_text())
        record["result"]["target_tray"] = "red"
        path.write_text(json.dumps(record))
    else:
        path = tmp_path / "run-contract.json"
        contract = json.loads(path.read_text())
        contract["place_horizon"] = 150
        path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        builder.paired_rl_rows(tmp_path)


@pytest.mark.parametrize("corruption", [None, "missing", "duplicate", "marker", "target"])
def test_rl_baseline_slice_completeness(corruption):
    rows = [{"layout_id": f"dev-{i}", "marker_present": False, "target_tray": "red"} for i in range(34)]
    contract = {"evaluated_layout_ids": [row["layout_id"] for row in rows]}
    if corruption == "missing": rows.pop()
    elif corruption == "duplicate": rows[-1] = dict(rows[0])
    elif corruption == "marker": rows[0]["marker_present"] = True
    elif corruption == "target": rows[0]["target_tray"] = "blue"
    if corruption:
        with pytest.raises(ValueError):
            builder.validate_rl_slice(rows, contract, False, "red")
    else:
        builder.validate_rl_slice(rows, contract, False, "red")
