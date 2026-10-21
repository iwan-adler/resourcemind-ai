import pandas as pd
import pytest

from resourcemind.data.generate import generate_dataset
from resourcemind.data.load import load_dataset, read_dataset, save_dataset
from resourcemind.data.validate import DataValidationError, validate_dataset, validate_history


def test_generation_is_deterministic(config):
    a, b = generate_dataset(config), generate_dataset(config)
    pd.testing.assert_frame_equal(a.zones, b.zones)
    pd.testing.assert_frame_equal(a.history, b.history)
    pd.testing.assert_frame_equal(a.travel, b.travel)


def test_different_seed_changes_data(config):
    a = generate_dataset(config)
    b = generate_dataset(config.with_updates(seed=config.seed + 1))
    assert not a.history["demand_total"].equals(b.history["demand_total"])


def test_schema_and_sizes(config, dataset):
    report = validate_dataset(dataset)
    assert report.ok, report.errors
    assert 8 <= len(dataset.zones) <= 12
    per_zone = dataset.history.groupby("zone_id").size()
    assert (per_zone >= 180).all()
    assert pd.api.types.is_datetime64_any_dtype(dataset.history["date"])
    assert pd.api.types.is_integer_dtype(dataset.history["demand_total"])
    assert dataset.zones["current_teams"].sum() == config.total_teams
    assert dataset.meta["provenance"].startswith("synthetic")


def test_no_negative_demand_and_categories_sum(dataset):
    h = dataset.history
    assert (h[["demand_medical", "demand_fire", "demand_other", "demand_total"]] >= 0).all().all()
    assert (h[["demand_medical", "demand_fire", "demand_other"]].sum(axis=1) == h["demand_total"]).all()


def test_travel_matrix_valid(dataset):
    t = dataset.travel.to_numpy()
    assert (t.diagonal() == 0).all()
    assert (t >= 0).all()
    assert (t == t.T).all()


def test_validation_catches_bad_history(dataset):
    bad = dataset.history.copy()
    bad.loc[0, "demand_total"] = -5
    bad = pd.concat([bad, bad.iloc[[1]]])
    report = validate_history(bad, dataset.zone_ids)
    assert not report.ok
    assert any("negative" in e for e in report.errors)
    assert any("duplicate" in e for e in report.errors)


def test_validation_catches_missing_values_and_unknown_zone(dataset):
    bad = dataset.history.copy()
    bad.loc[3, "demand_fire"] = None
    bad.loc[4, "zone_id"] = "Z99"
    report = validate_history(bad, dataset.zone_ids)
    assert any("missing" in e for e in report.errors)
    assert any("unknown" in e for e in report.errors)


def test_save_and_reload_roundtrip(tmp_path, config, dataset):
    save_dataset(dataset, tmp_path)
    again = read_dataset(tmp_path)
    assert validate_dataset(again).ok
    pd.testing.assert_frame_equal(again.history.reset_index(drop=True), dataset.history, check_dtype=False)
    loaded = load_dataset(config, tmp_path)
    assert len(loaded.history) == len(dataset.history)


def test_load_dataset_raises_on_invalid(tmp_path, config, dataset):
    broken = dataset.copy()
    broken.history.loc[0, "demand_total"] = -1
    save_dataset(broken, tmp_path)
    with pytest.raises(DataValidationError):
        load_dataset(config, tmp_path)
