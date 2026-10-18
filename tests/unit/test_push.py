"""Pushing agent metrics: it reaches a gateway, is off by default, and never breaks a run."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import yaml

from observability import agent_metrics as metrics
from observability import push

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "observability"


class Gateway:
    """A tiny stand-in for a Pushgateway that records what it was sent."""

    def __enter__(self):
        received = self.received = []

        class Handler(BaseHTTPRequestHandler):
            def do_PUT(self):  # noqa: N802
                received.append((self.path, self.rfile.read(int(self.headers["Content-Length"])).decode()))
                self.send_response(200)
                self.end_headers()

            do_POST = do_PUT

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def test_metrics_reach_the_gateway_under_job_dataset_and_script(monkeypatch):
    monkeypatch.setenv("AUTOEVOLVE_DATASET", "ieee")
    metrics.EXPERIMENTS.labels("adopt").inc()
    with Gateway() as gw:
        assert push.push_metrics("run_mission", started=0.0, url=gw.url) is True
    path, body = gw.received[0]
    assert "/metrics/job/autoevolve-agents" in path and "dataset/ieee" in path and "script/run_mission" in path
    assert "experiments_total" in body and "agent_run_last_timestamp_seconds" in body


def test_nothing_is_sent_without_a_gateway(monkeypatch):
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    assert push.push_metrics("run_mission", url="") is False
    assert push.gateway_url({}) == ""
    assert push.gateway_url({"pushgateway_url": "http://x:9091"}) == "http://x:9091"
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://env:9091")
    assert push.gateway_url({"pushgateway_url": "http://x:9091"}) == "http://env:9091"


def test_an_unreachable_gateway_does_not_raise(capsys):
    assert push.push_metrics("run_mission", url="http://127.0.0.1:9", timeout=0.5) is False
    assert "metrics not pushed" in capsys.readouterr().out


# -- the monitoring files for the agents ------------------------------------------------------
AGENT_METRIC = re.compile(r"\b((?:agent|llm|experiments|evaluation|mission)_[a-z_]+)\b")


def exported_names() -> set[str]:
    names: set[str] = set()
    for family in metrics.AGENT_REGISTRY.collect():
        if family.type == "counter":
            names |= {family.name, f"{family.name}_total"}
        elif family.type == "histogram":
            names |= {f"{family.name}{s}" for s in ("_bucket", "_count", "_sum")}
        else:
            names.add(family.name)
    return names


def test_agent_dashboard_only_uses_metrics_the_agents_export():
    dash = json.loads((OBS / "grafana" / "dashboards" / "autoevolve-agents.json").read_text())
    used = {m for p in dash["panels"] for t in p["targets"] for m in AGENT_METRIC.findall(t["expr"])}
    assert used and used <= exported_names(), sorted(used - exported_names())
    assert len({p["id"] for p in dash["panels"]}) == len(dash["panels"])
    assert all(p["datasource"]["uid"] == "prometheus" for p in dash["panels"])


def test_agent_alerts_use_exported_metrics_and_have_plain_summaries():
    groups = yaml.safe_load((OBS / "alerts.yml").read_text())["groups"]
    rules = [r for g in groups if g["name"] == "autoevolve-agents" for r in g["rules"]]
    assert len(rules) == 3
    for rule in rules:
        assert set(AGENT_METRIC.findall(rule["expr"])) <= exported_names()
        assert rule["annotations"]["summary"].endswith(".") and rule["labels"]["severity"] in {"warning", "critical"}


def test_prometheus_scrapes_the_pushgateway_and_keeps_pushed_labels():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    job = next(j for j in yaml.safe_load((OBS / "prometheus.yml").read_text())["scrape_configs"] if j["job_name"] == "pushgateway")
    assert job["honor_labels"] is True
    service, port = job["static_configs"][0]["targets"][0].split(":")
    assert service in compose["services"] and any(str(p).endswith(f":{port}") for p in compose["services"][service]["ports"])
    assert "mlflow" in compose["services"] and "mlflow-data" in compose["volumes"]


def test_every_pushing_script_registers_the_push():
    for name in ("run_mission", "run_agent", "run_data_review", "run_evaluation"):
        assert f'push_at_exit("{name}")' in (ROOT / "scripts" / f"{name}.py").read_text()
