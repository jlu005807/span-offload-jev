from experiments import run_appworld


def test_workflow_goal_strips_real_instruction(monkeypatch):
    monkeypatch.setattr(run_appworld, "_task_instruction", lambda task_id: "  Update the specified contact  ")

    goal = run_appworld._workflow_goal("024c982_1")

    assert goal.goal_text == "Update the specified contact"


def test_workflow_goal_keeps_missing_instruction_empty(monkeypatch):
    monkeypatch.setattr(run_appworld, "_task_instruction", lambda task_id: "")

    goal = run_appworld._workflow_goal("024c982_1")

    assert goal.goal_text == ""
    assert "024c982_1" not in goal.goal_text


def test_workflow_goal_keeps_whitespace_instruction_empty(monkeypatch):
    monkeypatch.setattr(run_appworld, "_task_instruction", lambda task_id: "   ")

    goal = run_appworld._workflow_goal("024c982_1")

    assert goal.goal_text == ""
