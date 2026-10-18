"""Mission control: start, watch and stop a mission from the browser. No Streamlit code lives here.

The page only launches `python -m scripts.run_mission` as a separate process, so a closed tab or a
page refresh never kills a run, and the same budget and kill switch apply as on the command line.
Stopping a run means creating the kill-switch file; the mission notices at its next step, reflects and ends.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from mission.plan import DEPTHS, Mission, MissionPlanner  # noqa: F401  (DEPTHS is re-exported for the page)

ROOT = Path(__file__).resolve().parents[1]
RUNS = "memory/runs"
DATASETS = ("telecom", "ieee")


@dataclass
class Run:
    id: str
    pid: int
    goal: str
    depth: str
    dataset: str
    started: str
    log: str


def processed_dir(dataset: str, root: Path = ROOT) -> Path:
    return root / "data" / ("processed" if dataset == "telecom" else f"{dataset}/processed")


def kill_switch(root: Path = ROOT) -> Path:
    return root / "memory" / "STOP"


def preview_plan(goal: str, depth: str, rounds: int | None = None, iterations: int | None = None) -> dict:
    """The plan the planner would make, without running anything. Raises ValueError on bad input."""
    mission = Mission(goal=goal, depth=depth, max_rounds=rounds, improve_iterations=iterations)
    plan = MissionPlanner().plan(mission)
    return {"rationale": plan.rationale, "steps": [{"name": s.name, "why": s.why, "params": s.params} for s in plan.steps],
            "rounds": mission.rounds}


def is_alive(pid: int) -> bool:
    try:
        if os.waitpid(pid, os.WNOHANG) != (0, 0):  # our own finished child: reap it so it is not a zombie
            return False
    except ChildProcessError:
        pass  # not our child (for example after the server restarted): fall through to the signal check
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def command(goal: str, depth: str, rounds: int | None, iterations: int | None, workers: int = 1) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.run_mission", "--goal", goal, "--depth", depth, "--workers", str(workers)]
    if rounds:
        cmd += ["--rounds", str(rounds)]
    if iterations:
        cmd += ["--iterations", str(iterations)]
    return cmd


def current(root: Path = ROOT) -> Run | None:
    """The run that is still going, if any."""
    for run in history(root):
        if is_alive(run.pid):
            return run
    return None


def history(root: Path = ROOT, limit: int = 20) -> list[Run]:
    files = sorted((root / RUNS).glob("*.json"), reverse=True)[:limit]
    runs = []
    for f in files:
        try:
            runs.append(Run(**json.loads(f.read_text())))
        except (OSError, ValueError, TypeError):
            continue
    return runs


def start(goal: str, depth: str, dataset: str, rounds: int | None = None, iterations: int | None = None,
          workers: int = 1, root: Path = ROOT, cmd: list[str] | None = None) -> Run:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    preview_plan(goal, depth, rounds, iterations)  # validates the input before anything starts
    if current(root) is not None:
        raise RuntimeError("A mission is already running. Stop it first, or wait for it to finish.")
    switch = kill_switch(root)
    if switch.exists():
        switch.unlink()  # a stale stop file would end the new run at once
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (root / RUNS).mkdir(parents=True, exist_ok=True)
    log = root / RUNS / f"{stamp}.log"
    env = {**os.environ, "AUTOEVOLVE_DATASET": dataset, "PYTHONUNBUFFERED": "1"}
    with log.open("w") as out:
        proc = subprocess.Popen(cmd or command(goal, depth, rounds, iterations, workers), cwd=root, env=env,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    run = Run(stamp, proc.pid, goal, depth, dataset, datetime.now().isoformat(timespec="seconds"), str(log))
    (root / RUNS / f"{stamp}.json").write_text(json.dumps(asdict(run)))
    return run


def stop(root: Path = ROOT) -> bool:
    """Ask the running mission to stop. True if a run was there to ask."""
    if current(root) is None:
        return False
    kill_switch(root).parent.mkdir(parents=True, exist_ok=True)
    kill_switch(root).write_text(f"stop requested {datetime.now().isoformat(timespec='seconds')}\n")
    return True


def tail(path: str | Path, lines: int = 40) -> str:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def latest_report(dataset: str, root: Path = ROOT) -> tuple[str, dict] | None:
    folder = processed_dir(dataset, root)
    try:
        return (folder / "mission_report.md").read_text(), json.loads((folder / "mission_report.json").read_text())
    except (OSError, ValueError):
        return None
