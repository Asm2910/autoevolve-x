"""The monitoring files are code too: they must parse, agree with each other, and only use metrics that exist."""

import json
import re
from pathlib import Path

import pytest
import yaml
from prometheus_client import REGISTRY

import observability.metrics  # noqa: F401  (importing registers the metrics)

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "observability"
METRIC_TOKEN = re.compile(r"\b((?:api|prediction|predictions|model)_[a-z_]+)\b")


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def registered_names() -> set[str]:
    return {sample.name for family in REGISTRY.collect() for sample in family.samples}


def dashboard() -> dict:
    return json.loads((OBS / "grafana" / "dashboards" / "autoevolve-api.json").read_text())


def alert_rules() -> list[dict]:
    return [rule for group in load_yaml(OBS / "alerts.yml")["groups"] for rule in group["rules"]]


def test_every_metric_in_the_dashboard_exists():
    known = registered_names()
    used = {m for panel in dashboard()["panels"] for t in panel["targets"] for m in METRIC_TOKEN.findall(t["expr"])}
    assert used, "no metrics found in the dashboard queries"
    assert used <= known, f"dashboard uses metrics the API does not export: {sorted(used - known)}"


def test_every_metric_in_the_alerts_exists():
    known = registered_names()
    used = {m for rule in alert_rules() for m in METRIC_TOKEN.findall(rule["expr"])}
    assert used <= known, f"alerts use metrics the API does not export: {sorted(used - known)}"


def test_alerts_have_severity_and_a_plain_summary():
    for rule in alert_rules():
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["annotations"]["summary"].endswith(".")


def test_dashboard_panels_have_unique_ids_and_use_the_provisioned_datasource():
    panels = dashboard()["panels"]
    assert len({p["id"] for p in panels}) == len(panels)
    datasource = load_yaml(OBS / "grafana" / "provisioning" / "datasources" / "prometheus.yml")["datasources"][0]
    for panel in panels:
        assert panel["datasource"]["uid"] == datasource["uid"]


def test_prometheus_scrapes_the_compose_service_on_its_port():
    compose = load_yaml(ROOT / "docker-compose.yml")
    target = load_yaml(OBS / "prometheus.yml")["scrape_configs"][0]["static_configs"][0]["targets"][0]
    service, port = target.split(":")
    assert service in compose["services"]
    assert any(str(p).endswith(f":{port}") for p in compose["services"][service]["ports"])


def test_compose_mounts_exist_and_volumes_are_declared():
    compose = load_yaml(ROOT / "docker-compose.yml")
    for service in compose["services"].values():
        for mount in service.get("volumes", []):
            source = mount.split(":")[0]
            if source.startswith("./"):
                assert (ROOT / source).exists(), f"{source} is mounted but missing"
            else:
                assert source in compose["volumes"], f"volume {source} is not declared"


def test_dashboard_provider_path_matches_the_mount():
    compose = load_yaml(ROOT / "docker-compose.yml")
    mounts = compose["services"]["grafana"]["volumes"]
    provider = load_yaml(OBS / "grafana" / "provisioning" / "dashboards" / "dashboards.yml")["providers"][0]
    assert any(m.split(":")[1] == provider["options"]["path"] for m in mounts)


def test_ci_workflow_runs_lint_and_tests():
    workflow = load_yaml(ROOT / ".github" / "workflows" / "ci.yml")
    steps = " ".join(str(s.get("run", "")) for s in workflow["jobs"]["test"]["steps"])
    assert "ruff check" in steps and "pytest" in steps


@pytest.mark.parametrize("name", ["prometheus.yml", "alerts.yml"])
def test_prometheus_files_are_valid_yaml(name):
    assert load_yaml(OBS / name)