"""Mission control logic, with a harmless stand-in for the mission process."""

import sys
import time

import pytest

from dashboard import control


def sleeper(seconds=30):
    return [sys.executable, "-c", f"import time; print('hello'); time.sleep({seconds})"]


def test_preview_plan_uses_the_depth_and_validates_input():
    plan = control.preview_plan("improve fraud detection", "quick")
    assert [s["name"] for s in plan["steps"]] == ["data_review", "evaluate", "improve", "reflect"]
    with pytest.raises(ValueError):
        control.preview_plan("x", "quick")
    with pytest.raises(ValueError):
        control.preview_plan("a valid goal", "reckless")


def test_start_watch_and_stop(tmp_path):
    run = control.start("improve fraud detection", "quick", "telecom", root=tmp_path, cmd=sleeper())
    try:
        assert control.current(tmp_path).id == run.id
        for _ in range(50):
            if "hello" in control.tail(run.log):
                break
            time.sleep(0.1)
        assert "hello" in control.tail(run.log)
        with pytest.raises(RuntimeError):
            control.start("improve fraud detection", "quick", "telecom", root=tmp_path, cmd=sleeper())
        assert control.stop(tmp_path) is True and control.kill_switch(tmp_path).exists()
    finally:
        import os
        import signal

        os.killpg(run.pid, signal.SIGKILL)
    for _ in range(50):
        if control.current(tmp_path) is None:
            break
        time.sleep(0.1)
    assert control.current(tmp_path) is None
    assert control.stop(tmp_path) is False


def test_a_stale_stop_file_is_cleared_before_a_new_run(tmp_path):
    control.kill_switch(tmp_path).parent.mkdir(parents=True)
    control.kill_switch(tmp_path).write_text("old")
    run = control.start("improve fraud detection", "quick", "ieee", root=tmp_path, cmd=[sys.executable, "-c", "pass"])
    assert not control.kill_switch(tmp_path).exists() and run.dataset == "ieee"
    assert control.history(tmp_path)[0].id == run.id


def test_bad_dataset_and_missing_report(tmp_path):
    with pytest.raises(ValueError):
        control.start("improve fraud detection", "quick", "mars", root=tmp_path, cmd=sleeper())
    assert control.latest_report("telecom", tmp_path) is None


def test_command_passes_options():
    cmd = control.command("goal text", "thorough", 2, 5, workers=4)
    assert cmd[-6:] == ["--workers", "4", "--rounds", "2", "--iterations", "5"]
