"""Shared logic for the Canadian Survey on Business Conditions (CSBC) analysis.

Every notebook and the pipeline CLI import from here, so the schema rules, validation,
label normalisation and chart styling exist in exactly one place. Paths are derived from
this file's own location rather than the working directory, so the project runs from
anywhere it is unzipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import pandas as pd


def _in_notebook() -> bool:
    """True only inside a Jupyter kernel.

    Outside one - pytest, the CLI - an interactive backend would pop up a window and
    block until it is closed, so the backend has to be chosen before the first figure.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False


IN_NOTEBOOK = _in_notebook()
if not IN_NOTEBOOK:
    mpl.use("Agg")   # headless: render straight to file, never to a window

import matplotlib.pyplot as plt  # (must follow the backend selection)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Anchored to this file, never to os.getcwd(): a notebook opened from src/ and a CLI run
# from the project root must resolve to the same directories.
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "combined_csbc.csv"
OUTPUT_DIR = ROOT / "outputs"


def ensure_dirs() -> None:
    """Create the writable output folders if a fresh clone does not have them yet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "GEO",
    "Business_characteristics",
    "Business_information",
    "Expected_change",
    "VALUE",
    "Quarter",
]

CANONICAL_METRICS = {"Employment", "Sales", "Profitability", "Investment"}
CANONICAL_CHANGES = {"increase", "stay about the same", "decrease"}

# Q2 2024 renamed two labels. Mapping them explicitly (rather than fuzzy-matching) means
# an unrecognised label fails validation instead of silently splitting a historical series.
METRIC_ALIASES = {"Capital Investment": "Investment"}
CHANGE_ALIASES = {"stay the same": "stay about the same"}

# A survey observation is uniquely identified by these five fields; a duplicate would
# double-count a group when quarters are appended.
NATURAL_KEY = [
    "GEO",
    "Business_characteristics",
    "Business_information",
    "Expected_change",
    "Quarter",
]

# The grouping that should contain exactly three rows - one per direction.
GROUP_COLS = ["GEO", "Business_characteristics", "Business_information", "Quarter"]

# Story order: Sales leads the narrative, Investment is the exception it ends on.
METRIC_ORDER = ["Sales", "Employment", "Profitability", "Investment"]
DIRECTION_ORDER = ["increase", "stay about the same", "decrease"]

ALL_INDUSTRIES = "North American Industry Classification System (NAICS), all industries"

# Labelled so charts and prose use one wording for "outside the three-way split".
NO_ANSWER = "no directional answer"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Outcome of validating one file or one combined frame."""

    source: str
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def quarter_sort_key(value: str) -> tuple[int, int]:
    """Turn 'Q2 2024' into a sortable (year, quarter) pair.

    Quarters sort as strings in the wrong order ('Q1 2024' < 'Q3 2023'), so every
    ordering in this project goes through here. Malformed values raise, because a
    quarter we cannot place on a timeline must not enter the series.
    """
    match = re.fullmatch(r"Q([1-4])\s+(\d{4})", str(value).strip())
    if not match:
        raise ValueError(f"Invalid Quarter value: {value!r}. Expected a format like 'Q2 2024'.")
    return int(match.group(2)), int(match.group(1))


def sort_quarters(values) -> list[str]:
    """Return the unique quarters in chronological order."""
    return sorted(pd.Series(list(values)).dropna().astype(str).unique(), key=quarter_sort_key)


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map the known historical label variants onto canonical labels."""
    out = df.copy()
    out["Business_information"] = out["Business_information"].replace(METRIC_ALIASES)
    out["Expected_change"] = out["Expected_change"].replace(CHANGE_ALIASES)
    return out


def check_schema(df: pd.DataFrame) -> None:
    """Fail fast on a missing column, naming what was received to speed up diagnosis."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed. Missing required column(s): {', '.join(missing)}. "
            f"Received: {list(df.columns)}"
        )


def validate(
    df: pd.DataFrame,
    *,
    source: str = "dataframe",
    expect_single_quarter: bool = False,
) -> ValidationResult:
    """Validate schema, quarters, value ranges, keys and categories.

    Structural problems raise (the load must stop); characteristics of the data warn and
    are recorded (they should be reported and carried forward, not hidden). Missing
    percentages are never imputed, and the three directional shares are never rescaled
    to total 100 - `residual_by_metric` shows why.
    """
    check_schema(df)
    result = ValidationResult(source=source)

    quarters = df["Quarter"].dropna().astype(str).unique().tolist()
    for quarter in quarters:
        quarter_sort_key(quarter)
    if expect_single_quarter and len(quarters) != 1:
        raise ValueError(f"A quarterly file must contain exactly one Quarter; found {quarters}.")

    values = pd.to_numeric(df["VALUE"], errors="coerce")
    # A non-numeric entry is different from a blank: blanks are expected, text is not.
    non_numeric = df["VALUE"].notna() & values.isna()
    if non_numeric.any():
        raise ValueError(
            "VALUE contains non-numeric entries, e.g. "
            f"{df.loc[non_numeric, 'VALUE'].astype(str).unique()[:5].tolist()}"
        )
    out_of_range = values.notna() & ~values.between(0, 100)
    if out_of_range.any():
        raise ValueError(
            "VALUE must fall between 0 and 100 (it is a percentage). "
            f"Outside range: {values[out_of_range].head().tolist()}"
        )

    result.metrics.update(
        rows=int(len(df)),
        quarters=sort_quarters(quarters),
        missing_values=int(values.isna().sum()),
        duplicate_keys=int(df.duplicated(NATURAL_KEY).sum()),
    )

    if result.metrics["duplicate_keys"]:
        raise ValueError(
            f"Found {result.metrics['duplicate_keys']} duplicate natural key row(s). "
            "Appending would double-count these groups."
        )

    unknown_metrics = sorted(set(df["Business_information"].dropna()) - CANONICAL_METRICS)
    unknown_changes = sorted(set(df["Expected_change"].dropna()) - CANONICAL_CHANGES)
    if unknown_metrics or unknown_changes:
        raise ValueError(
            "Unknown category value(s) - review for legitimate schema drift before ingesting. "
            f"Business_information: {unknown_metrics}; Expected_change: {unknown_changes}"
        )

    direction_counts = df.groupby(GROUP_COLS, dropna=False)["Expected_change"].nunique()
    incomplete = int((direction_counts != 3).sum())
    result.metrics["groups_missing_a_direction"] = incomplete
    if incomplete:
        result.warnings.append(f"{incomplete} group(s) lack all three direction rows.")

    if result.metrics["missing_values"]:
        result.warnings.append(
            f"{result.metrics['missing_values']} VALUE cell(s) missing; retained as missing."
        )

    triplets = triplet_sums(df)
    # Empty groups (all three shares zero) are absent cells, not partial responses, so
    # they are counted separately instead of inflating the "incomplete" figure.
    empty = int((triplets["sum"] == 0).sum())
    short = int(((triplets["sum"] > 0) & ~triplets["sum"].between(99.8, 100.2)).sum())
    result.metrics.update(
        complete_triplets=int(len(triplets)),
        empty_groups=empty,
        triplets_short_of_100=short,
    )
    if short:
        result.warnings.append(
            f"{short} complete triplet(s) sum materially below 100; no renormalisation applied."
        )
    if empty:
        result.warnings.append(f"{empty} group(s) are empty (all three shares zero).")

    return result


def triplet_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Sum the three directional shares for every group that has all three rows."""
    values = pd.to_numeric(df["VALUE"], errors="coerce")
    agg = (
        df.assign(VALUE=values)
        .groupby(GROUP_COLS, dropna=False)["VALUE"]
        .agg(["count", "sum"])
        .reset_index()
    )
    return agg[agg["count"] == 3].copy()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a supplied CSV.

    utf-8-sig strips the byte-order mark the source files carry; without it the first
    column name arrives as '\\ufeffGEO' on some platforms and the schema check fails.
    """
    return pd.read_csv(path, encoding="utf-8-sig")


def load_quarter(path: str | Path, *, expect_single_quarter: bool = True):
    """Read, normalise and validate one quarterly file.

    Returns the canonical frame plus a ValidationResult that records how many labels the
    normalisation step had to rewrite, so drift stays visible in the audit trail.
    """
    path = Path(path)
    raw = read_csv(path)
    check_schema(raw)

    # Counted before normalisation, otherwise the evidence of drift is erased by the fix.
    aliases = {
        alias: int((raw["Business_information"] == alias).sum()) for alias in METRIC_ALIASES
    }
    aliases.update(
        {alias: int((raw["Expected_change"] == alias).sum()) for alias in CHANGE_ALIASES}
    )

    df = normalize_labels(raw)
    result = validate(df, source=path.name, expect_single_quarter=expect_single_quarter)
    result.metrics["aliases_normalized"] = {k: v for k, v in aliases.items() if v}
    return df, result


def combine_quarters(paths):
    """Load several quarterly files and validate both the parts and the whole.

    The union is re-validated because some failures only appear across files - a
    duplicate natural key spanning two quarters is invisible when each is checked alone.
    """
    frames, results = [], []
    for path in paths:
        frame, result = load_quarter(path)
        frames.append(frame)
        results.append(result)

    combined = pd.concat(frames, ignore_index=True)
    return combined, results, validate(combined, source="combined")


def load_combined() -> pd.DataFrame:
    """Return the analysis-ready dataset.

    Prefers the processed file written by notebook 01. Falls back to rebuilding from raw
    so an analysis notebook still runs if someone opens it before running 01.
    """
    if PROCESSED_FILE.exists():
        return read_csv(PROCESSED_FILE)
    combined, _, _ = combine_quarters(sorted(RAW_DIR.glob("*.csv")))
    return combined


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def residual_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    """How much of each group is unaccounted for by the three directional shares.

    The residual is the share of businesses giving no directional answer. It turns out to
    be strongly metric-specific, which is why the metrics are not pooled into an index.
    """
    triplets = triplet_sums(df)
    triplets["residual"] = 100 - triplets["sum"]
    # Empty groups would drag every average toward 100 and hide the real pattern.
    real = triplets[triplets["sum"] > 0]
    return (
        real.groupby("Business_information")["residual"]
        .agg(groups="count", mean_residual="mean", median_residual="median", max_residual="max")
        .round(1)
        .reindex(METRIC_ORDER)
    )


def metric_matrix(df: pd.DataFrame, direction: str = "increase") -> pd.DataFrame:
    """One row per survey cell, one column per metric - the shape correlation needs.

    The source data is long, so a correlation between raw columns would be meaningless
    (GEO against VALUE). Pivoting to metric-per-column makes the question answerable:
    do these four expectations move together across groups?
    """
    wide = (
        df[df["Expected_change"] == direction]
        .pivot_table(
            index=["GEO", "Business_characteristics", "Quarter"],
            columns="Business_information",
            values="VALUE",
            aggfunc="first",
            dropna=False,
        )
        .dropna()
    )
    # Cells where every metric reads zero are absent groups, not agreement between metrics.
    return wide[(wide > 0).all(axis=1)].reindex(columns=METRIC_ORDER)


def metric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between the four metrics' increase shares."""
    return metric_matrix(df).corr().round(2)


def direction_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Within each metric, how the three directional shares co-move.

    If the three shares were a closed partition of respondents, increase and decrease
    would be strongly negatively correlated. Showing that they are not is the evidence
    for leaving the shares un-rescaled.
    """
    rows = []
    for metric in METRIC_ORDER:
        wide = (
            df[df["Business_information"] == metric]
            .pivot_table(
                index=["GEO", "Business_characteristics", "Quarter"],
                columns="Expected_change",
                values="VALUE",
                aggfunc="first",
                dropna=False,
            )
            .dropna()
        )
        wide = wide[(wide > 0).any(axis=1)]
        rows.append({
            "metric": metric,
            "cells": len(wide),
            "r(increase, decrease)": round(wide["increase"].corr(wide["decrease"]), 2),
            "r(increase, same)": round(wide["increase"].corr(wide["stay about the same"]), 2),
        })
    return pd.DataFrame(rows).set_index("metric")


def national_increase_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share expecting an increase, Canada / all industries, by quarter and metric."""
    subset = df[
        (df["GEO"] == "Canada")
        & (df["Business_characteristics"] == ALL_INDUSTRIES)
        & (df["Expected_change"] == "increase")
        & (df["Business_information"].isin(METRIC_ORDER))
    ]
    return (
        subset.pivot(index="Quarter", columns="Business_information", values="VALUE")
        .reindex(sort_quarters(subset["Quarter"]))
        .reindex(columns=METRIC_ORDER)
    )


def composition_table(df: pd.DataFrame, quarter: str,
                      characteristic: str = ALL_INDUSTRIES) -> pd.DataFrame:
    """Four-way response composition per metric: the three directions plus the residual."""
    subset = df[
        (df["GEO"] == "Canada")
        & (df["Business_characteristics"] == characteristic)
        & (df["Quarter"] == quarter)
    ]
    table = (
        subset.pivot_table(
            index="Business_information", columns="Expected_change",
            values="VALUE", aggfunc="first", dropna=False,
        )
        .reindex(index=METRIC_ORDER, columns=DIRECTION_ORDER)
    )
    table[NO_ANSWER] = (100 - table.sum(axis=1)).round(1)
    return table


def sector_balance_table(df: pd.DataFrame, quarter: str,
                         metric: str = "Profitability") -> pd.DataFrame:
    """Net balance by industry = share expecting an increase minus share expecting a decrease.

    'Stay about the same' stays neutral rather than being split between the two sides, and
    the balance needs no assumption that the three shares total 100.
    """
    subset = df[
        (df["GEO"] == "Canada")
        & (df["Business_characteristics"] != ALL_INDUSTRIES)
        & (df["Business_information"] == metric)
        & (df["Quarter"] == quarter)
    ]
    # dropna=False keeps a response column that happens to be entirely missing, so the
    # explicit check below reports a missing *value* rather than a missing *column*.
    table = subset.pivot_table(
        index="Business_characteristics", columns="Expected_change",
        values="VALUE", aggfunc="first", dropna=False,
    )

    missing_cols = [c for c in DIRECTION_ORDER if c not in table.columns]
    if missing_cols:
        raise ValueError(f"{quarter} {metric}: missing response column(s) {missing_cols}")
    # The balance is only meaningful on observed values, so refuse rather than treat a
    # blank as zero - that would read as "nobody expects a decrease".
    if table[["increase", "decrease"]].isna().any().any():
        raise ValueError(
            f"{quarter} {metric}: cannot compute a net balance where increase or decrease is missing."
        )

    table["net_balance_pp"] = (table["increase"] - table["decrease"]).round(1)
    return table[[*DIRECTION_ORDER, "net_balance_pp"]].sort_values("net_balance_pp")


def sector_balance_trend(df: pd.DataFrame, metric: str = "Profitability") -> pd.DataFrame:
    """The sector balance summarised per quarter.

    Analysis of a single quarter cannot say whether a negative balance is new or
    long-standing. This supplies the baseline: without it, "margins are under pressure"
    has nothing to be measured against.
    """
    rows = []
    for quarter in sort_quarters(df["Quarter"]):
        table = sector_balance_table(df, quarter, metric=metric)
        balance = table["net_balance_pp"]
        rows.append({
            "Quarter": quarter,
            "industries": len(balance),
            "negative": int((balance < 0).sum()),
            "mean_balance_pp": round(balance.mean(), 1),
            "median_balance_pp": round(balance.median(), 1),
            "worst_balance_pp": round(balance.min(), 1),
        })
    return pd.DataFrame(rows).set_index("Quarter")


def sector_balance_trend_chart(df: pd.DataFrame, output_path=None,
                               metric: str = "Profitability") -> pd.DataFrame:
    """Mean sector balance over time, annotated with how many industries are negative.

    One series on one axis: the negative-industry count is a label rather than a second
    y-scale, because two scales on one plot invite a false comparison.
    """
    trend = sector_balance_trend(df, metric=metric)
    quarters = list(trend.index)
    values = trend["mean_balance_pp"]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.axhline(0, color=BASELINE, linewidth=1.2, zorder=2)
    ax.plot(range(len(trend)), values, color=SERIES_COLOR[metric], linewidth=2.4,
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.4,
            zorder=3)

    for i, (value, negative, total) in enumerate(
        zip(values, trend["negative"], trend["industries"])
    ):
        ax.annotate(f"{value:+.1f}".replace("-", "−"),
                    xy=(i, value), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=10.5, color=INK)
        ax.annotate(f"{negative} of {total}\nnegative",
                    xy=(i, value), xytext=(0, -26), textcoords="offset points",
                    ha="center", fontsize=9.5, color=MUTED)

    ax.set_xticks(range(len(trend)), quarters)
    ax.set_xlim(-0.35, len(trend) - 0.65)
    ax.set_ylim(min(values) - 9, 4)
    ax.set_ylabel(f"Mean net {metric.lower()} balance (pp)")
    ax.set_title(f"{metric} expectations are persistently negative, and slowly improving")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _save(fig, output_path)
    return trend


def regional_table(df: pd.DataFrame, quarter_from: str, quarter_to: str,
                   metric: str = "Sales") -> pd.DataFrame:
    """Increase share by geography for two quarters, with the change and 4-quarter volatility.

    Volatility is reported alongside the change because the largest movements come from
    the smallest jurisdictions, and the caveat has to travel with the number.
    """
    subset = df[
        (df["Business_characteristics"] == ALL_INDUSTRIES)
        & (df["Business_information"] == metric)
        & (df["Expected_change"] == "increase")
    ]
    wide = subset.pivot(index="GEO", columns="Quarter", values="VALUE")
    quarters = sort_quarters(subset["Quarter"])

    out = pd.DataFrame({
        quarter_from: wide[quarter_from],
        quarter_to: wide[quarter_to],
    })
    out["change_pp"] = (out[quarter_to] - out[quarter_from]).round(1)
    out["volatility_sd"] = wide[quarters].std(axis=1).round(1)
    return out.sort_values("change_pp", ascending=False)


# ---------------------------------------------------------------------------
# Chart styling
# ---------------------------------------------------------------------------
# Ink and chrome are separate from series colour: text always wears an ink colour so a
# series hue never has to carry meaning that a colourblind reader would lose.
INK, INK_SOFT, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Fixed hue per metric, assigned by identity and never by rank, so a metric keeps its
# colour across every figure in the report.
SERIES_COLOR = {
    "Sales": "#2a78d6",
    "Employment": "#eb6834",
    "Profitability": "#1baf7a",
    "Investment": "#eda100",
}

# Diverging pair for signed quantities: warm and cool read as opposites.
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"

# Ordered greys plus one accent for the four-way composition: the three directions are
# shades of one hue, and the residual is the colour that should catch the eye.
COMPOSITION_COLOR = {
    "increase": "#2a78d6",
    "stay about the same": "#b7d3f6",
    "decrease": "#e34948",
    NO_ANSWER: "#898781",
}


def apply_chart_theme() -> None:
    """Recessive chrome, thin marks, generous space - applied once per notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SOFT,
        "ytick.labelcolor": INK_SOFT,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 11,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _display_path(path: Path) -> str:
    """Show a project-relative path when possible; callers may pass a temp directory."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        # Outside the project (a temp folder in a demo or test): show the name only, so
        # no absolute path from one machine is ever written into a notebook's output.
        return Path(path).name


def _save(fig, output_path: str | Path | None):
    """Persist a figure if a destination was given, then display or discard it."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    if IN_NOTEBOOK:
        plt.show()
    else:
        # Headless callers have nothing to show, and leaving figures open leaks memory
        # when the pipeline renders many of them in one run.
        plt.close(fig)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def national_increase_chart(df: pd.DataFrame, output_path=None,
                            title: str | None = None) -> pd.DataFrame:
    """Line chart of the increase share per metric over time.

    Shared by the analysis notebook and the pipeline so a refreshed history renders the
    identical figure rather than a similar-looking copy.
    """
    table = national_increase_table(df)
    quarters = list(table.index)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for metric in METRIC_ORDER:
        ax.plot(
            range(len(table)), table[metric], color=SERIES_COLOR[metric], label=metric,
            linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
        )

    # Label only the endpoints, nudged apart where two series finish almost level - a
    # value on every point would be unreadable.
    x_end = len(table) - 1
    placed, min_gap = [], 0.9
    for value, metric in sorted((table[m].iloc[x_end], m) for m in METRIC_ORDER):
        y = value if not placed else max(value, placed[-1] + min_gap)
        placed.append(y)
        ax.annotate(
            f"{metric}  {value:.1f}%", xy=(x_end, value), xytext=(x_end + 0.09, y),
            textcoords="data", va="center", ha="left", fontsize=10.5,
            color=INK_SOFT, annotation_clip=False,
        )

    ax.set_title(title or f"National expectations, {quarters[0]} to {quarters[-1]}")
    ax.set_ylabel("Businesses expecting an increase (%)")
    ax.set_xticks(range(len(table)), quarters)
    ax.set_xlim(-0.12, x_end + 0.16)
    ax.set_ylim(0, 23)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12),
              handlelength=1.6, columnspacing=1.8)
    _save(fig, output_path)
    return table


def sector_balance_chart(df: pd.DataFrame, quarter: str, output_path=None,
                         metric: str = "Profitability") -> pd.DataFrame:
    """Diverging horizontal bars: net balance by industry, worst first."""
    table = sector_balance_table(df, quarter, metric=metric)
    values = table["net_balance_pp"]
    negative = int((values < 0).sum())

    import textwrap

    fig, ax = plt.subplots(figsize=(10, 7.4))
    ax.barh(range(len(table)), values, height=0.66, zorder=2,
            color=[POSITIVE if v >= 0 else NEGATIVE for v in values])
    ax.axvline(0, color=BASELINE, linewidth=1.2, zorder=3)

    for i, value in enumerate(values):
        offset, align = (1.0, "left") if value >= 0 else (-1.0, "right")
        # Unicode minus matches the axis tick labels.
        ax.text(value + offset, i, f"{value:+.1f}".replace("-", "−"),
                va="center", ha=align, fontsize=10, color=INK_SOFT, zorder=3)

    ax.set_yticks(range(len(table)), [textwrap.fill(str(n), 34) for n in table.index])
    ax.invert_yaxis()   # worst at the top, so the chart reads in the order a reader scans
    ax.set_xlabel(f"Net {metric.lower()} balance, percentage points  (increase − decrease)")
    ax.set_title(f"{quarter} {metric.lower()} expectations are negative in "
                 f"{negative} of {len(table)} industries")
    ax.set_xlim(values.min() - 8, max(values.max() + 6, 6))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(fig, output_path)
    return table


def regional_dumbbell_chart(df: pd.DataFrame, quarter_from: str, quarter_to: str,
                            output_path=None, metric: str = "Sales") -> pd.DataFrame:
    """Dumbbell chart: each geography's before and after, connected.

    Two points per row beats two bar series here - the gap between the dots *is* the
    change, which is the quantity the reader wants.
    """
    table = regional_table(df, quarter_from, quarter_to, metric=metric)

    fig, ax = plt.subplots(figsize=(9.8, 6.8))
    for i, (geo, row) in enumerate(table.iterrows()):
        rose = row["change_pp"] >= 0
        ax.plot([row[quarter_from], row[quarter_to]], [i, i],
                color=POSITIVE if rose else NEGATIVE, linewidth=2, zorder=2)
        # Hollow dot for the earlier quarter, filled for the later one: shape carries the
        # time order even where colour is not distinguishable.
        ax.plot(row[quarter_from], i, "o", markersize=7, markerfacecolor=SURFACE,
                markeredgecolor=MUTED, markeredgewidth=1.6, zorder=3)
        ax.plot(row[quarter_to], i, "o", markersize=7,
                color=POSITIVE if rose else NEGATIVE, zorder=3)
        ax.annotate(f"{row['change_pp']:+.1f}".replace("-", "−"),
                    xy=(max(row[quarter_from], row[quarter_to]) + 0.8, i),
                    va="center", ha="left", fontsize=9.5, color=INK_SOFT)

    labels = [f"$\\bf{{{g}}}$" if g == "Canada" else g for g in table.index]
    ax.set_yticks(range(len(table)), labels)
    ax.invert_yaxis()
    # Canada is a reference line, not a peer of the provinces, so mark it rather than
    # letting it sit unremarked in the ranking.
    canada_pos = list(table.index).index("Canada")
    ax.axhline(canada_pos, color=GRID, linewidth=8, zorder=1)

    ax.set_xlabel(f"Businesses expecting {metric.lower()} to increase (%)")
    ax.set_title(f"{metric} expectations improved unevenly, {quarter_from} to {quarter_to}")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SURFACE,
                   markeredgecolor=MUTED, markeredgewidth=1.6, label=quarter_from),
        plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=POSITIVE,
                   label=f"{quarter_to} (higher)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=NEGATIVE,
                   label=f"{quarter_to} (lower)"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper left",
              bbox_to_anchor=(0, -0.09))
    _save(fig, output_path)
    return table


def response_donuts(df: pd.DataFrame, quarter: str, metrics=("Employment", "Investment"),
                    output_path=None) -> pd.DataFrame:
    """Donut per metric showing the four-way split of responses.

    A pie/donut is only the right form when the parts genuinely make a whole and there are
    few of them. Four segments of one survey question qualifies; comparing 13 industries
    would not, which is why those use a ranked bar instead.
    """
    table = composition_table(df, quarter)
    segments = [*DIRECTION_ORDER, NO_ANSWER]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.6))
    axes = axes if len(metrics) > 1 else [axes]

    for ax, metric in zip(axes, metrics):
        shares = table.loc[metric, segments].fillna(0)
        wedges, _ = ax.pie(
            shares, colors=[COMPOSITION_COLOR[s] for s in segments],
            startangle=90, counterclock=False,
            wedgeprops={"width": 0.42, "edgecolor": SURFACE, "linewidth": 2},
        )
        # The residual is the point of the chart, so it is called out in the centre.
        ax.text(0, 0.08, metric, ha="center", va="center", fontsize=12, color=INK,
                fontweight="semibold")
        ax.text(0, -0.16, f"{shares[NO_ANSWER]:.0f}% no answer", ha="center", va="center",
                fontsize=10, color=INK_SOFT)
        for wedge, segment in zip(wedges, segments):
            share = shares[segment]
            if share < 4:            # too thin a wedge to label without collisions
                continue
            angle = (wedge.theta1 + wedge.theta2) / 2
            ax.annotate(
                f"{share:.1f}%",
                xy=(0.79 * _cos(angle), 0.79 * _sin(angle)),
                xytext=(1.18 * _cos(angle), 1.18 * _sin(angle)),
                ha="left" if _cos(angle) >= 0 else "right", va="center",
                fontsize=10, color=INK_SOFT,
            )

    handles = [plt.Rectangle((0, 0), 1, 1, color=COMPOSITION_COLOR[s]) for s in segments]
    fig.legend(handles, segments, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), fontsize=10)
    fig.suptitle(f"{quarter}: Employment responses form a whole, Investment responses do not",
                 fontsize=13, fontweight="semibold")
    _save(fig, output_path)
    return table.loc[list(metrics)]


def correlation_heatmap(matrix: pd.DataFrame, output_path=None,
                        title: str = "Correlation") -> pd.DataFrame:
    """Lower-triangle heatmap of a correlation matrix.

    The upper triangle repeats the lower one, so it is masked - the reader should not have
    to work out that half the grid is redundant.
    """
    from matplotlib.colors import LinearSegmentedColormap

    # One-hue sequential ramp: correlations here are all positive, so a diverging map
    # would imply a meaningful midpoint that does not exist.
    ramp = LinearSegmentedColormap.from_list(
        "blues", ["#eaf2fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
    )

    labels = list(matrix.columns)
    shown = matrix.copy()
    for i, row in enumerate(labels):
        for j, col in enumerate(labels):
            if j > i:
                shown.loc[row, col] = float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(shown.values, cmap=ramp, vmin=0, vmax=1)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = shown.iloc[i, j]
            if pd.isna(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11,
                    color=SURFACE if value > 0.55 else INK)

    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.72, label="Pearson r")
    _save(fig, output_path)
    return matrix


def metric_scatter(df: pd.DataFrame, x: str, y: str, output_path=None) -> float:
    """Scatter of two metrics' increase shares, one point per survey cell."""
    wide = metric_matrix(df)
    r = wide[x].corr(wide[y])

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(wide[x], wide[y], s=16, alpha=0.35, color=SERIES_COLOR[x],
               edgecolors="none", zorder=2)

    # Least-squares line purely as a visual guide to the association's direction.
    slope, intercept = _fit_line(wide[x], wide[y])
    xs = [wide[x].min(), wide[x].max()]
    ax.plot(xs, [slope * v + intercept for v in xs], color=INK_SOFT, linewidth=1.6,
            zorder=3)

    ax.set_xlabel(f"{x}: businesses expecting an increase (%)")
    ax.set_ylabel(f"{y}: businesses expecting an increase (%)")
    ax.set_title(f"{x} and {y} expectations move together  (r = {r:.2f}, n = {len(wide):,})")
    ax.grid(True)
    ax.set_axisbelow(True)
    _save(fig, output_path)
    return float(round(r, 2))   # plain float, so a notebook shows 0.58 not np.float64(0.58)


def _cos(degrees: float) -> float:
    import math
    return math.cos(math.radians(degrees))


def _sin(degrees: float) -> float:
    import math
    return math.sin(math.radians(degrees))


def _fit_line(x: pd.Series, y: pd.Series):
    """Slope and intercept of the least-squares line, without pulling in numpy.polyfit."""
    x_mean, y_mean = x.mean(), y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    return slope, y_mean - slope * x_mean


def composition_bars(df: pd.DataFrame, quarter: str, output_path=None) -> pd.DataFrame:
    """Stacked bars of the four-way response split, one bar per metric.

    Stacking is right here because the four parts belong to one whole; the residual
    segment is what makes Investment's difference visible at a glance.
    """
    table = composition_table(df, quarter)
    segments = [*DIRECTION_ORDER, NO_ANSWER]

    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    left = pd.Series(0.0, index=table.index)
    for segment in segments:
        widths = table[segment].fillna(0)
        ax.barh(table.index, widths, left=left, height=0.62,
                color=COMPOSITION_COLOR[segment], label=segment, zorder=2)
        for metric in table.index:
            width = widths[metric]
            # Only label a segment wide enough to hold the text without spilling over.
            if width >= 7:
                ax.text(left[metric] + width / 2, metric, f"{width:.0f}",
                        ha="center", va="center", fontsize=9.5,
                        color=SURFACE if segment != "stay about the same" else INK_SOFT)
        left += widths

    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Share of businesses (%)")
    ax.set_title(f"{quarter}: how each question's responses divide")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.28),
              handlelength=1.4, columnspacing=1.4, fontsize=10)
    _save(fig, output_path)
    return table


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def analyse_single_quarter(new_file, output_dir=None) -> dict:
    """Mode 1: what does this quarter say on its own?

    Runs before any decision to admit the file to the history, so a new quarter can be
    inspected first. Needs no combined dataset and writes its own quarter-stamped files.
    """
    output_dir = Path(output_dir or OUTPUT_DIR)
    df, result = load_quarter(new_file)
    quarter = df["Quarter"].iloc[0]
    slug = quarter.replace(" ", "_")

    composition = composition_bars(df, quarter, output_dir / f"quarter_{slug}_composition.png")
    balance = sector_balance_chart(df, quarter,
                                  output_dir / f"quarter_{slug}_sector_balance.png")
    balance.to_csv(output_dir / f"quarter_{slug}_sector_balance.csv")

    print(f"\nStandalone view of {quarter}")
    print(f"  rows: {result.metrics['rows']:,}")
    print(f"  missing VALUE cells: {result.metrics['missing_values']}")
    if result.metrics["aliases_normalized"]:
        print(f"  labels normalised: {result.metrics['aliases_normalized']}")
    print(f"  industries with a negative profitability balance: "
          f"{int((balance['net_balance_pp'] < 0).sum())} of {len(balance)}")
    return {"quarter": quarter, "frame": df, "validation": result,
            "composition": composition, "sector_balance": balance}


def ingest_and_refresh(new_file, combined_file=None, output_dir=None) -> dict:
    """Mode 2: append the quarter to the history and refresh the trend view.

    Refuses a quarter already present, so re-running is safe. The union is re-validated
    after concatenation because cross-file problems are invisible file by file.
    """
    combined_file = Path(combined_file or PROCESSED_FILE)
    output_dir = Path(output_dir or OUTPUT_DIR)

    new_df, result = load_quarter(new_file)
    quarter = new_df["Quarter"].iloc[0]

    if combined_file.exists():
        existing = read_csv(combined_file)
        validate(existing, source=combined_file.name)
        if quarter in set(existing["Quarter"].dropna().astype(str)):
            raise ValueError(
                f"Quarter {quarter!r} is already present in {combined_file.name}. "
                "Aborting to prevent double ingestion."
            )
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined_result = validate(combined, source="combined")

    combined_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_file, index=False)

    quarters = sort_quarters(combined["Quarter"])
    table = national_increase_chart(
        combined, output_dir / "pipeline_national_increase_trend.png",
        title=f"National expectations, refreshed through {quarters[-1]}",
    )
    table.to_csv(output_dir / "pipeline_national_increase_trend.csv")

    print(f"\nIngested {quarter}: {result.metrics['rows']:,} rows")
    print(f"  history now holds {len(combined):,} rows across {len(quarters)} quarters")
    print(f"  saved: {_display_path(combined_file)}")
    return {"quarter": quarter, "combined": combined,
            "validation": combined_result, "trend": table}


def quality_report(results, combined_result, output_path) -> str:
    """Render the machine-generated audit trail written alongside the processed data."""
    lines = ["DATA QUALITY REPORT", "=" * 72, ""]
    for result in results:
        m = result.metrics
        lines += [
            result.source,
            f"  Rows: {m['rows']}",
            f"  Quarter(s): {', '.join(m['quarters'])}",
            f"  Missing VALUE cells: {m['missing_values']}",
            f"  Duplicate natural keys: {m['duplicate_keys']}",
        ]
        if m.get("aliases_normalized"):
            lines.append(f"  Known label aliases normalized: {m['aliases_normalized']}")
        lines += [f"  WARNING: {w}" for w in result.warnings]
        lines.append("")

    m = combined_result.metrics
    lines += [
        "Combined dataset",
        f"  Rows: {m['rows']}",
        f"  Quarters: {', '.join(m['quarters'])}",
        f"  Missing VALUE cells: {m['missing_values']}",
        f"  Duplicate natural keys: {m['duplicate_keys']}",
        f"  Empty groups (all three shares zero): {m['empty_groups']}",
        f"  Triplets short of 100: {m['triplets_short_of_100']} of {m['complete_triplets']}",
        *[f"  WARNING: {w}" for w in combined_result.warnings],
        "",
        "Decisions:",
        "  Missing percentages are retained as missing, never back-calculated.",
        "  Directional shares are not renormalized to 100: the residual is real and",
        "  strongly metric-specific (largest for Investment), so redistributing it",
        "  would invent responses the survey did not record.",
        "  Empty groups are reported separately from partial ones - they are absent",
        "  cells rather than incomplete answers.",
    ]

    text = "\n".join(lines)
    Path(output_path).write_text(text, encoding="utf-8")
    return text
