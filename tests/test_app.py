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


def test_deep_link_opens_requested_page_and_scenario():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["page"] = "planner"
    at.query_params["scenario"] = "surge"
    at.run()
    assert not at.exception
    assert at.session_state["page"] == "Allocation planner"
    assert at.session_state["scenario"] == "surge"
    assert any("Allocation planner" in t.value for t in at.title)


def test_daily_planning_mode_renders_every_page(app):
    app.sidebar.radio(key="planning_mode").set_value("daily").run()
    assert not app.exception
    for page in PAGES:
        app.sidebar.radio(key="page").set_value(page).run()
        assert not app.exception, (page, [e.value for e in app.exception])
    app.sidebar.radio(key="page").set_value("Allocation planner").run()
    assert any(s.value == "Teams stationed per zone and day" for s in app.subheader)
    assert "Day-by-day plan" in _texts(app)
    assert any("service_lost_requests" in d.value.columns for d in app.dataframe)
    app.sidebar.radio(key="planning_mode").set_value("single").run()
    assert not any(s.value == "Teams stationed per zone and day" for s in app.subheader)


def test_daily_mode_overview_lab_and_sensitivity(app):
    app.sidebar.radio(key="planning_mode").set_value("daily").run()
    app.sidebar.radio(key="page").set_value("Command overview").run()
    assert not app.exception
    text = _texts(app)
    assert "Day-by-day plan" in text and "team move(s) over 7 days" in text
    app.slider(key="map_day").set_value(3).run()
    assert not app.exception
    assert "movements on day 3" in _texts(app)

    app.sidebar.radio(key="page").set_value("Scenario lab").run()
    app.slider(key="lab_w_normal_stage").set_value(0.3)
    app.button[0].click().run()  # "Run simulation"
    assert not app.exception
    text = _texts(app)
    assert "INFEASIBLE" in text and "day-by-day" in text
    app.button(key="lab_reset").click().run()

    app.sidebar.radio(key="page").set_value("Strategy evaluation").run()
    next(b for b in app.button if b.label == "Run sensitivity analysis").click().run()
    assert not app.exception
    _, sweep = app.session_state["sensitivity"]
    assert set(sweep["strategy"]) == {"current", "greedy", "optimized"}
    assert (sweep.loc[sweep["strategy"] == "optimized", "status"] == "OPTIMAL").all()
    del app.session_state["sensitivity"]
    app.sidebar.radio(key="planning_mode").set_value("single").run()


def test_what_if_tab_replans_a_forced_move(app):
    app.sidebar.radio(key="planning_mode").set_value("single").run()
    app.sidebar.selectbox(key="scenario").set_value("normal").run()
    app.sidebar.radio(key="page").set_value("Allocation planner").run()
    assert not app.exception
    app.selectbox(key="wi_from").set_value(7).run()  # Industrial Park
    app.selectbox(key="wi_to").set_value(4).run()  # Eastfield
    app.button(key="wi_run").click().run()
    assert not app.exception
    assert app.session_state["what_if"][1:] == (7, 4, 1)
    assert any("Why not move 1 team(s) from Industrial Park to Eastfield?" in m.value for m in app.info)


def test_robustness_check_runs_from_strategy_page(app):
    app.sidebar.radio(key="planning_mode").set_value("single").run()
    app.sidebar.radio(key="page").set_value("Strategy evaluation").run()
    app.button(key="rob_run").click().run()
    assert not app.exception
    assert app.session_state["robustness"][1] is False
    assert any("held-out test forecast windows" in c.value for c in app.caption)


def test_response_standard_setting_changes_the_plan(app):
    app.sidebar.radio(key="planning_mode").set_value("single").run()
    app.sidebar.radio(key="page").set_value("Allocation planner").run()
    app.sidebar.checkbox(key="use_standard").check().run()
    app.sidebar.slider(key="standard").set_value(20).run()
    assert not app.exception
    assert any("Response-time standard: 20 min" in i.value for i in app.info)
    assert "Coverage within 20 min" in "\n".join(m.value for m in app.markdown)
    app.sidebar.checkbox(key="use_standard").uncheck().run()
    assert not any("Response-time standard" in i.value for i in app.info)
