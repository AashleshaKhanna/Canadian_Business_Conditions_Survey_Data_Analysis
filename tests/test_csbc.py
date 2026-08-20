"""By Aashlesha Khanna
Tests for the shared CSBC module.

Run from the project root with:  python -m pytest -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# The module lives in src/, which is not a package - add it to the path so the tests can
# import it without requiring an installed distribution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import csbc  # noqa: E402


def sample_frame() -> pd.DataFrame:
    """One complete, valid group: three directions summing to 100."""
    return pd.DataFrame({
        "GEO": ["Canada"] * 3,
        "Business_characteristics": ["Example sector"] * 3,
        "Business_information": ["Sales"] * 3,
        "Expected_change": ["increase", "stay about the same", "decrease"],
        "VALUE": [20.0, 70.0, 10.0],
        "Quarter": ["Q3 2024"] * 3,
    })


# --- schema and label handling ---------------------------------------------

def test_known_label_aliases_are_normalized():
    drifted = sample_frame()
    drifted.loc[1, "Expected_change"] = "stay the same"
    drifted["Business_information"] = "Capital Investment"

    out = csbc.normalize_labels(drifted)

    assert set(out["Expected_change"]) == csbc.CANONICAL_CHANGES
    assert set(out["Business_information"]) == {"Investment"}


def test_missing_required_column_fails_fast():
    with pytest.raises(ValueError, match="Missing required column"):
        csbc.validate(sample_frame().drop(columns=["VALUE"]))


def test_unknown_category_is_surfaced_as_drift():
    drifted = sample_frame()
    drifted["Business_information"] = "Capital Expenditure"

    with pytest.raises(ValueError, match="Unknown category"):
        csbc.validate(drifted)


# --- value ranges and keys -------------------------------------------------

def test_out_of_range_percentage_is_rejected():
    bad = sample_frame()
    bad.loc[0, "VALUE"] = 120

    with pytest.raises(ValueError, match="between 0 and 100"):
        csbc.validate(bad)


def test_non_numeric_value_is_rejected():
    bad = sample_frame()
    bad["VALUE"] = bad["VALUE"].astype(object)   # a real file would arrive as text
    bad.loc[0, "VALUE"] = "n/a"

    with pytest.raises(ValueError, match="non-numeric"):
        csbc.validate(bad)


def test_duplicate_natural_key_is_rejected():
    doubled = pd.concat([sample_frame(), sample_frame()], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate natural key"):
        csbc.validate(doubled)


# --- quarters --------------------------------------------------------------

def test_quarter_sort_key_orders_chronologically():
    unsorted = ["Q1 2024", "Q3 2023", "Q2 2024", "Q4 2023"]
    assert csbc.sort_quarters(unsorted) == ["Q3 2023", "Q4 2023", "Q1 2024", "Q2 2024"]


def test_malformed_quarter_is_rejected():
    with pytest.raises(ValueError, match="Invalid Quarter"):
        csbc.quarter_sort_key("2024-Q2")


def test_single_quarter_rule_applies_only_to_incoming_files():
    two = pd.concat([sample_frame(), sample_frame().assign(Quarter="Q4 2024")],
                    ignore_index=True)

    # Fine for a combined history, rejected for one incoming quarterly file.
    csbc.validate(two)
    with pytest.raises(ValueError, match="exactly one Quarter"):
        csbc.validate(two, expect_single_quarter=True)


# --- missingness and the residual -----------------------------------------

def test_missing_value_is_flagged_not_imputed():
    with_gap = sample_frame()
    with_gap.loc[2, "VALUE"] = None

    result = csbc.validate(with_gap)

    assert result.metrics["missing_values"] == 1
    assert any("retained as missing" in w for w in result.warnings)


def test_residual_is_measured_not_removed():
    """A group summing to 85 should report a 15pp residual, never be rescaled."""
    short = sample_frame()
    short["VALUE"] = [20.0, 55.0, 10.0]

    residual = csbc.residual_by_metric(short)

    assert residual.loc["Sales", "mean_residual"] == pytest.approx(15.0)


def test_empty_groups_are_counted_separately_from_partial_ones():
    """All-zero groups are absent cells, so they must not inflate the 'short of 100' count."""
    empty = sample_frame()
    empty["VALUE"] = [0.0, 0.0, 0.0]

    result = csbc.validate(empty)

    assert result.metrics["empty_groups"] == 1
    assert result.metrics["triplets_short_of_100"] == 0


# --- derived tables --------------------------------------------------------

def test_sector_balance_refuses_missing_increase_or_decrease():
    frame = sample_frame()
    frame["Business_information"] = "Profitability"
    frame.loc[2, "VALUE"] = None   # the 'decrease' share

    with pytest.raises(ValueError, match="cannot compute a net balance"):
        csbc.sector_balance_table(frame, "Q3 2024")


def test_sector_balance_is_increase_minus_decrease():
    frame = sample_frame()
    frame["Business_information"] = "Profitability"

    table = csbc.sector_balance_table(frame, "Q3 2024")

    assert table.loc["Example sector", "net_balance_pp"] == pytest.approx(10.0)


# --- pipeline guards -------------------------------------------------------

def test_ingest_creates_then_refuses_duplicate_quarter(tmp_path):
    source = tmp_path / "Data CSBC-Q3 2024.csv"
    # A single group is enough to exercise the guard, but the chart needs the national
    # rows, so borrow one real quarter from the project's own raw data.
    real = sorted(csbc.RAW_DIR.glob("*.csv"))[0]
    csbc.read_csv(real).to_csv(source, index=False)

    combined = tmp_path / "combined.csv"
    first = csbc.ingest_and_refresh(source, combined_file=combined, output_dir=tmp_path)
    assert combined.exists()
    assert len(first["combined"]) == 2352

    with pytest.raises(ValueError, match="already present"):
        csbc.ingest_and_refresh(source, combined_file=combined, output_dir=tmp_path)
