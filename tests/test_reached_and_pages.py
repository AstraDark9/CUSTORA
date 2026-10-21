"""Reached customers, the Analyse pages, the clusters data and the per-stage details.

Runs against the shipped analysis outputs. The reached list is redirected to a temp
file for every test, so these tests never touch the real data/ folder.
"""

import json

import pandas as pd
import pytest

import app as custora
from pipeline import PIPELINE_STAGES
from src.orchestration import stage_report
from src.outreach import reached_store

BUSINESS = "data/processed/customers_business.csv"


@pytest.fixture(autouse=True)
def temp_reached_store(tmp_path, monkeypatch):
    monkeypatch.setattr(reached_store, "STORE_PATH", tmp_path / "reached.json")
    return tmp_path / "reached.json"


@pytest.fixture()
def client():
    return custora.app.test_client()


def _top_customer(client):
    return str(client.get("/customers/list.json").get_json()["rows"][0]["CustomerID"])


# ----------------------------------------------------------------- the store itself


def test_store_mark_unmark_and_load(tmp_path):
    path = tmp_path / "s.json"
    assert reached_store.load("A", path) == {}
    record = reached_store.mark("C1", {"ExpectedLoss": 12.5, "Ignored": 1}, "A", path)
    assert record["snapshot"]["ExpectedLoss"] == 12.5
    assert "Ignored" not in record["snapshot"]
    assert reached_store.reached_ids("A", path) == {"C1"}
    assert reached_store.unmark("C1", "A", path) is True
    assert reached_store.unmark("C1", "A", path) is False
    assert reached_store.reached_ids("A", path) == set()


def test_store_list_belongs_to_one_dataset(tmp_path):
    path = tmp_path / "s.json"
    reached_store.mark("C1", {}, "dataset-A", path)
    assert reached_store.reached_ids("dataset-A", path) == {"C1"}
    assert reached_store.reached_ids("dataset-B", path) == set()  # a different dataset ignores it
    reached_store.mark("C9", {}, "dataset-B", path)  # and starts its own list
    assert reached_store.reached_ids("dataset-B", path) == {"C9"}


def test_store_file_is_always_valid_json_and_survives_garbage(tmp_path):
    path = tmp_path / "s.json"
    reached_store.mark("C1", {}, "A", path)
    assert json.loads(path.read_text())["customers"]["C1"]["reached_at"]
    path.write_text("not json at all")
    assert reached_store.load("A", path) == {}  # a damaged file must not crash the app
    assert not list(tmp_path.glob("*.tmp"))  # no temp files left behind


# ----------------------------------------------------------------- marking customers as reached


def test_reached_customer_leaves_every_to_do_list_and_appears_on_reached_page(client):
    n_before = client.get("/api/matrix.json").get_json()["n"]
    export_before = client.get("/customers/export.csv").get_data(as_text=True)
    cid = _top_customer(client)
    assert cid in export_before

    reply = client.post(f"/customers/{cid}/reached").get_json()
    assert reply["ok"] and reply["reached_count"] == 1

    rows = client.get("/customers/list.json", query_string={"q": cid}).get_json()["rows"]
    assert cid not in [str(r["CustomerID"]) for r in rows]
    assert client.get("/api/matrix.json").get_json()["n"] == n_before - 1
    assert cid not in client.get("/customers/export.csv").get_data(as_text=True)
    assert cid in client.get("/reached").get_data(as_text=True)


def test_unreached_restores_the_customer_everywhere(client):
    n_before = client.get("/api/matrix.json").get_json()["n"]
    cid = _top_customer(client)
    client.post(f"/customers/{cid}/reached")
    reply = client.post(f"/customers/{cid}/unreached").get_json()
    assert reply["ok"] and reply["was_reached"] is True and reply["reached_count"] == 0
    assert client.get("/api/matrix.json").get_json()["n"] == n_before
    assert cid in client.get("/customers/export.csv").get_data(as_text=True)
    assert client.post(f"/customers/{cid}/unreached").get_json()["was_reached"] is False


def test_marking_an_unknown_customer_is_a_404(client):
    assert client.post("/customers/NO-SUCH-CUSTOMER/reached").status_code == 404


def test_reaching_one_customer_never_moves_another_between_groups(client):
    """The group cut lines come from everyone, so group totals must only drop by the reached customer."""
    total = len(pd.read_csv(BUSINESS))

    def group_totals():
        out = {}
        for key in ("save_now", "protect", "automate", "healthy"):
            out[key] = client.get("/customers/list.json", query_string={"quadrant": key}).get_json()["total"]
        return out

    before = group_totals()
    assert sum(before.values()) == total
    cid = _top_customer(client)
    client.post(f"/customers/{cid}/reached")
    after = group_totals()
    assert sum(after.values()) == total - 1
    # exactly one group lost exactly one customer; all others are unchanged
    changed = {k: before[k] - after[k] for k in before if before[k] != after[k]}
    assert list(changed.values()) == [1]


def test_reached_page_empty_state(client):
    html = client.get("/reached").get_data(as_text=True)
    assert "Nobody has been marked as reached yet" in html


def test_profile_page_knows_if_the_customer_was_reached(client):
    cid = _top_customer(client)
    assert 'data-reached="no"' in client.get(f"/customer/{cid}").get_data(as_text=True)
    client.post(f"/customers/{cid}/reached")
    assert 'data-reached="yes"' in client.get(f"/customer/{cid}").get_data(as_text=True)


# ----------------------------------------------------------------- the pages


@pytest.mark.parametrize("url", ["/", "/risk-value", "/outreach-planner", "/clusters", "/reached", "/evidence", "/upload"])
def test_pages_load(client, url):
    assert client.get(url).status_code == 200


def test_matrix_and_planner_moved_off_the_dashboard(client):
    dash = client.get("/").get_data(as_text=True)
    assert 'id="matrix"' not in dash and 'id="planner"' not in dash
    assert 'id="matrix"' in client.get("/risk-value").get_data(as_text=True)
    assert 'id="planner"' in client.get("/outreach-planner").get_data(as_text=True)


def test_sidebar_has_new_sections_evidence_last_and_a_help_button(client):
    html = client.get("/").get_data(as_text=True)
    side = html[html.index("<aside"): html.index("</aside>")]
    order = ["Retention command center", "Reached customers", "Risk × value matrix",
             "Outreach planner", "Customer clusters", "Model evidence"]
    positions = [side.index(label) for label in order]
    assert positions == sorted(positions), "Model evidence must come after every other section"
    assert 'id="helpOpen"' in html and 'id="helpPanel"' in html


def test_quadrant_buttons_on_risk_page_count_everyone_not_yet_reached(client):
    html = client.get("/risk-value").get_data(as_text=True)
    assert html.count('class="quad"') == 4


# ----------------------------------------------------------------- clusters data


def test_clusters_endpoint_is_consistent(client):
    d = client.get("/api/clusters.json").get_json()
    assert d["available"] is True
    n = len(d["id"])
    assert n > 0 and len(d["x"]) == len(d["y"]) == len(d["c"]) == len(d["p"]) == n
    assert all(0 <= c < len(d["segments"]) for c in d["c"])
    assert sum(seg["customers"] for seg in d["segments"]) == d["n"]


# ----------------------------------------------------------------- per-stage details


@pytest.mark.parametrize("stage", list(PIPELINE_STAGES))
def test_every_stage_has_details_for_the_shipped_run(client, stage):
    d = client.get("/run-analysis/stage.json", query_string={"name": stage}).get_json()
    assert d["stage"] == stage and d["available"] is True
    assert d["headline"]
    assert d["stats"] or d["bars"] or d["chips"]


def test_unknown_stage_is_a_404(client):
    assert client.get("/run-analysis/stage.json", query_string={"name": "Nope"}).status_code == 404
    assert client.get("/run-analysis/stage.json").status_code == 404


def test_stage_details_degrade_gracefully_when_files_are_missing(tmp_path):
    for stage in PIPELINE_STAGES:
        d = stage_report.describe_stage(stage, tmp_path, "£")
        assert d["available"] is False and d["headline"]
