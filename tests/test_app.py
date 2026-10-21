"""Headless UI tests with Streamlit's AppTest: controls must drive real computation."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Command overview", "Demand forecasts", "Allocation planner", "Scenario lab", "Strategy evaluation"]


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    return at


def _texts(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [i.value for i in at.info] + [w.value for w in at.warning]
    return "\n".join(str(p) for p in parts)


def test_every_page_and_scenario_renders(app):
    for scenario in ["normal", "surge", "shortage", "disruption"]:
        app.sidebar.selectbox(key="scenario").set_value(scenario).run()
        for page in PAGES:
            app.sidebar.radio(key="page").set_value(page).run()
            assert not app.exception, (scenario, page, [e.value for e in app.exception])
    app.sidebar.selectbox(key="scenario").set_value("normal").run()


def test_disclaimer_and_status_visible(app):
    app.sidebar.radio(key="page").set_value("Command overview").run()
    text = _texts(app)
    assert "Decision-support prototype" in text
    assert "Solver: OPTIMAL" in text
    assert "Synthetic data" in text


def test_lab_override_changes_results_and_reset_restores(app):
    app.sidebar.radio(key="page").set_value("Scenario lab").run()
    app.slider(key="lab_w_normal_stage").set_value(0.3)
    app.button[0].click().run()  # "Run simulation" (form submit)
    assert not app.exception
    assert app.session_state["lab_overrides"] is not None
    text = _texts(app)
    assert "INFEASIBLE" in text
    assert "Staging capacity: x0.3" in text
    app.button(key="lab_reset").click().run()
    assert app.session_state["lab_overrides"] is None
    assert "No overrides yet" in _texts(app)


def test_sensitivity_button_runs_real_sweep(app):
    app.sidebar.radio(key="page").set_value("Strategy evaluation").run()
    next(b for b in app.button if b.label == "Run sensitivity analysis").click().run()
    assert not app.exception
    param, sweep = app.session_state["sensitivity"]
    assert param == "resources"
    assert sweep["factor"].nunique() == 9
    assert set(sweep["strategy"]) == {"current", "greedy", "optimized"}


def test_horizon_setting_changes_forecast(app):
    app.sidebar.radio(key="page").set_value("Demand forecasts").run()
    app.sidebar.slider(key="horizon").set_value(3).run()
    assert not app.exception
    daily = [d for d in app.dataframe if d.value.shape[1] == 3]
    assert daily, "daily forecast table should have 3 date columns for a 3-day horizon"
    app.sidebar.slider(key="horizon").set_value(7).run()
