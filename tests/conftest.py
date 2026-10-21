import numpy as np
import pytest

from resourcemind.allocation.models import AllocationProblem
from resourcemind.config import load_config
from resourcemind.data.generate import generate_dataset
from resourcemind.forecasting.evaluation import evaluate_forecasters


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def dataset(config):
    return generate_dataset(config)


@pytest.fixture(scope="session")
def forecast_eval(config, dataset):
    return evaluate_forecasters(dataset.demand_wide(), config.horizon_days, config.forecast, random_state=config.seed)


def make_problem(
    demand=(100, 20, 60),
    available=(1, 4, 1),
    staging=(5, 5, 5),
    service_cap=None,
    weights=(1.0, 1.0, 1.0),
    travel=None,
    eligible=None,
    retain=0,
    rate=20,
    max_travel=60,
    coverage_floor=0.0,
) -> AllocationProblem:
    """Small hand-checkable problem. Rate = service requests per team over the horizon."""
    n = len(demand)
    travel = np.array(travel if travel is not None else [[0, 10, 20], [10, 0, 15], [20, 15, 0]][:n])
    travel = travel[:n, :n]
    if eligible is None:
        eligible = travel <= max_travel
    eligible = np.array(eligible, dtype=bool)
    np.fill_diagonal(eligible, True)
    return AllocationProblem(
        zone_ids=tuple(f"Z{i + 1:02d}" for i in range(n)),
        zone_names=tuple(f"Zone {i + 1}" for i in range(n)),
        priority_tiers=tuple("standard" for _ in range(n)),
        demand=np.array(demand, dtype=int),
        available=np.array(available, dtype=int),
        staging_capacity=np.array(staging, dtype=int),
        service_cap=np.array(service_cap if service_cap is not None else [10_000] * n, dtype=int),
        weights=np.array(weights, dtype=float),
        travel=travel.astype(int),
        eligible=eligible,
        retain=np.full(n, retain, dtype=int),
        service_per_team=rate,
        horizon_days=1,
        max_travel_minutes=max_travel,
        coverage_floor=coverage_floor,
    )


@pytest.fixture
def small_problem():
    return make_problem()
