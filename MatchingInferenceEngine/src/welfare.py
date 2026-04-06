from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class WelfareResults:
    student_level: pd.DataFrame
    # Remark 3 – global (unconditional) summary across all p values
    global_sweep: pd.DataFrame
    # Remark 1 – full sweep over p for each conditioning dimension
    top_p_sweep_by_list_length: pd.DataFrame
    top_p_sweep_by_priority_percentile: pd.DataFrame | None
    top_p_sweep_by_category: dict[str, pd.DataFrame]
    # Remark 4 – conjunctions of attributes
    top_p_sweep_by_conjunction: dict[tuple[str, ...], pd.DataFrame]
    saved_paths: dict[str, str] | None = None


def _match_rank(ranking: list[int], match_idx: int) -> int | None:
    if match_idx < 0:
        return None
    try:
        return ranking.index(match_idx) + 1
    except ValueError:
        return None


def _resolve_categories(student_df: pd.DataFrame, categories: list[str] | None) -> list[str]:
    if categories is not None:
        return [c for c in categories if c in student_df.columns]
    return [c for c in ["Residential District", "Home Language"] if c in student_df.columns]


def _resolve_priority(
    df: pd.DataFrame,
    priority_col: str | None,
    priority_matrix: np.ndarray | None,
) -> pd.DataFrame:
    """
    Add priority_score and priority_percentile columns to df (in-place copy).

    Priority sources (mutually exclusive, priority_matrix wins):
    - priority_matrix : ndarray of shape (n_students, n_schools).
      Each student's effective priority = priority_matrix[i, match_idx[i]].
      Unmatched students get NaN.
    - priority_col : name of an existing scalar column in df.
    """
    df = df.copy()

    if priority_matrix is not None:
        n, m = priority_matrix.shape
        if n != len(df):
            raise ValueError(
                f"priority_matrix has {n} rows but there are {len(df)} students."
            )
        scores = np.full(n, np.nan)
        for i, midx in enumerate(df["match_idx"].to_numpy()):
            if 0 <= midx < m:
                scores[i] = priority_matrix[i, midx]
        df["priority_score"] = scores

    elif priority_col is not None:
        if priority_col not in df.columns:
            raise ValueError(f"Missing priority column: {priority_col}")
        scores = pd.to_numeric(df[priority_col], errors="coerce")
        df["priority_score"] = scores

    else:
        return df  # no priority info supplied → skip

    priority = df["priority_score"]
    df["priority_percentile"] = priority.rank(pct=True, ascending=False) * 100
    return df


def build_student_level_welfare(
    rankings_as_indices: list[list[int]],
    matches_idx: list[int] | np.ndarray,
    student_attributes: pd.DataFrame,
    priority_col: str | None = None,
    priority_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build a student-level DataFrame enriched with match quality columns.

    Parameters
    ----------
    rankings_as_indices : list of lists
        Each inner list is the preference ranking of one student, expressed as
        integer indices into the school array.
    matches_idx : array-like of int
        Index of the school each student was matched to (-1 = unmatched).
    student_attributes : DataFrame
        One row per student.  Any column can later be used for conditioning.
    priority_col : str, optional
        Name of a scalar priority column already present in student_attributes.
    priority_matrix : ndarray of shape (n_students, n_schools), optional
        Per-(student, school) priority scores.  The effective score used is the
        one at the student's matched school (Remark 2).
    """
    n = len(rankings_as_indices)
    if len(matches_idx) != n or len(student_attributes) != n:
        raise ValueError(
            "rankings_as_indices, matches_idx and student_attributes must have the same length"
        )

    df = student_attributes.reset_index(drop=True).copy()
    if "student_id" not in df.columns:
        df.insert(0, "student_id", np.arange(n))

    rankings = [list(map(int, ranking)) for ranking in rankings_as_indices]
    match_idx = pd.Series(matches_idx).fillna(-1).astype(int).to_numpy()

    df["ranking"] = rankings
    df["list_length"] = [len(r) for r in rankings]
    df["match_idx"] = match_idx
    df["match_rank"] = [_match_rank(r, m) for r, m in zip(rankings, match_idx)]
    df["matched"] = df["match_rank"].notna()

    # Remark 2 – unified priority resolution
    df = _resolve_priority(df, priority_col, priority_matrix)

    return df

def _top_p_flag_column(df: pd.DataFrame, p: int) -> pd.Series:
    return df["match_rank"].le(p).fillna(False)


def summarize_global_sweep(student_df: pd.DataFrame, max_p: int | None = None) -> pd.DataFrame:
    """
    Remark 3 – unconditional top-p rate for every p from 1 to max_p.

    max_p defaults to the maximum list length observed in the data.
    """
    k = max_p or int(student_df["list_length"].max())
    rows = []
    for p in range(1, k + 1):
        rate = _top_p_flag_column(student_df, p).mean()
        rows.append({"p": p, "top_p_rate": rate, "top_p_pct": 100 * rate})
    return pd.DataFrame(rows)


def summarize_top_p_sweep_by_list_length(
    student_df: pd.DataFrame, max_p: int | None = None
) -> pd.DataFrame:
    """
    Remark 1 – top-p rate for every (p, list_length) pair.

    Returns a long-format DataFrame with columns [p, list_length, students,
    top_p_rate, top_p_pct].
    """
    k = max_p or int(student_df["list_length"].max())
    frames = []
    for p in range(1, k + 1):
        tmp = student_df.copy()
        tmp["top_p"] = _top_p_flag_column(tmp, p)
        agg = (
            tmp.groupby("list_length", dropna=False)
            .agg(students=("student_id", "size"), top_p_rate=("top_p", "mean"))
            .reset_index()
        )
        agg.insert(0, "p", p)
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True).sort_values(["p", "list_length"])
    out["top_p_pct"] = 100 * out["top_p_rate"]
    return out


def summarize_top_p_sweep_by_priority_percentile(
    student_df: pd.DataFrame,
    max_p: int | None = None,
    n_bins: int = 10,
) -> pd.DataFrame | None:
    """
    Remark 1 – top-p rate for every (p, priority_percentile_bin) pair.

    Returns None if no priority information is available.
    """
    if "priority_percentile" not in student_df.columns:
        return None

    k = max_p or int(student_df["list_length"].max())
    edges = np.linspace(0, 100, n_bins + 1)
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(n_bins)]

    df = student_df.copy()
    df["priority_bin"] = pd.cut(
        df["priority_percentile"].clip(0, 100),
        bins=edges,
        labels=labels,
        include_lowest=True,
    )

    frames = []
    for p in range(1, k + 1):
        df["top_p"] = _top_p_flag_column(df, p)
        agg = (
            df.groupby("priority_bin", observed=False)
            .agg(
                students=("student_id", "size"),
                top_p_rate=("top_p", "mean"),
                avg_priority_percentile=("priority_percentile", "mean"),
            )
            .reset_index()
        )
        agg.insert(0, "p", p)
        frames.append(agg)

    out = pd.concat(frames, ignore_index=True)
    out["top_p_pct"] = 100 * out["top_p_rate"]
    return out


def summarize_top_p_sweep_by_category(
    student_df: pd.DataFrame,
    categories: list[str] | None = None,
    max_p: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Remark 1 – top-p rate for every (p, category_value) pair, for each
    single-attribute category.

    Returns a dict {category_name: long-format DataFrame}.
    """
    resolved = _resolve_categories(student_df, categories)
    k = max_p or int(student_df["list_length"].max())

    out: dict[str, pd.DataFrame] = {}
    for category in resolved:
        frames = []
        df = student_df.copy()
        for p in range(1, k + 1):
            df["top_p"] = _top_p_flag_column(df, p)
            agg = (
                df.groupby(category, dropna=False)
                .agg(students=("student_id", "size"), top_p_rate=("top_p", "mean"))
                .reset_index()
            )
            agg.insert(0, "p", p)
            frames.append(agg)
        result = pd.concat(frames, ignore_index=True).sort_values(["p", "top_p_rate"])
        result["top_p_pct"] = 100 * result["top_p_rate"]
        out[category] = result
    return out


def summarize_top_p_sweep_by_conjunction(
    student_df: pd.DataFrame,
    conjunctions: list[list[str]] | list[tuple[str, ...]],
    max_p: int | None = None,
) -> dict[tuple[str, ...], pd.DataFrame]:
    """
    Remark 4 – top-p rate conditioned on a conjunction (cross-product) of
    attributes, for every p from 1 to max_p.

    Parameters
    ----------
    conjunctions : list of attribute-name lists / tuples
        E.g. [["Residential District", "Home Language"]] computes the metric
        jointly for each (district, language) cell.

    Returns a dict {(attr1, attr2, ...): long-format DataFrame}.
    """
    k = max_p or int(student_df["list_length"].max())
    out: dict[tuple[str, ...], pd.DataFrame] = {}

    for conjunction in conjunctions:
        cols = [c for c in conjunction if c in student_df.columns]
        if not cols:
            continue
        key = tuple(cols)
        frames = []
        df = student_df.copy()
        for p in range(1, k + 1):
            df["top_p"] = _top_p_flag_column(df, p)
            agg = (
                df.groupby(cols, dropna=False)
                .agg(students=("student_id", "size"), top_p_rate=("top_p", "mean"))
                .reset_index()
            )
            agg.insert(0, "p", p)
            frames.append(agg)
        result = pd.concat(frames, ignore_index=True)
        result["top_p_pct"] = 100 * result["top_p_rate"]
        out[key] = result
    return out


def _save_plot(fig: plt.Figure, output_path: str | Path | None, show: bool) -> str | None:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return None if output_path is None else str(output_path)


def plot_global_sweep(
    summary: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """Remark 3 – CDF of match quality over all students."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["p"], summary["top_p_pct"], marker="o")
    ax.set_xlabel("p")
    ax.set_ylabel("Top-p (%)")
    ax.set_title("Global top-p rate (all students)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    return _save_plot(fig, output_path, show)


def plot_top_p_sweep_vs_list_length(
    summary: pd.DataFrame,
    p_values: list[int] | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """Remark 1 – one line per selected p value."""
    ps = p_values or sorted(summary["p"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in ps:
        sub = summary[summary["p"] == p].sort_values("list_length")
        ax.plot(sub["list_length"], sub["top_p_pct"], marker="o", label=f"top-{p}")
    ax.set_xlabel("List length")
    ax.set_ylabel("Top-p (%)")
    ax.set_title("Top-p by list length")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_plot(fig, output_path, show)


def plot_top_p_sweep_vs_priority_percentile(
    summary: pd.DataFrame,
    p_values: list[int] | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """Remark 1 – one line per selected p value."""
    ps = p_values or sorted(summary["p"].unique())
    bins = summary["priority_bin"].astype(str).unique()
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in ps:
        sub = summary[summary["p"] == p]
        ax.plot(x, sub["top_p_pct"].values, marker="o", label=f"top-{p}")
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=45, ha="right")
    ax.set_xlabel("Priority percentile bin")
    ax.set_ylabel("Top-p (%)")
    ax.set_title("Top-p by priority percentile")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_plot(fig, output_path, show)


def plot_top_p_sweep_by_category(
    summary: pd.DataFrame,
    category: str,
    p_values: list[int] | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """Remark 1 – grouped bar chart, one group per category value, one bar per p."""
    ps = sorted(p_values or summary["p"].unique())
    cat_vals = summary[category].astype(str).unique()
    x = np.arange(len(cat_vals))
    width = 0.8 / max(len(ps), 1)

    fig, ax = plt.subplots(figsize=(max(9, len(cat_vals) * 1.2), 5))
    for i, p in enumerate(ps):
        sub = summary[summary["p"] == p].copy()
        sub[category] = sub[category].astype(str)
        heights = [
            sub.loc[sub[category] == v, "top_p_pct"].values[0]
            if v in sub[category].values else 0.0
            for v in cat_vals
        ]
        ax.bar(x + i * width, heights, width, label=f"top-{p}")

    ax.set_xticks(x + width * (len(ps) - 1) / 2)
    ax.set_xticklabels(cat_vals, rotation=45, ha="right")
    ax.set_xlabel(category)
    ax.set_ylabel("Top-p (%)")
    ax.set_title(f"Top-p by {category}")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save_plot(fig, output_path, show)


def evaluate_simulation_output(
    sim_output: dict[str, Any],
    max_p: int | None = None,
    categories: list[str] | None = None,
    conjunctions: list[list[str]] | None = None,
    priority_col: str | None = None,
    priority_matrix: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    n_priority_bins: int = 10,
    plot_p_values: list[int] | None = None,
    show: bool = False,
) -> WelfareResults:
    """
    Compute all welfare metrics from a simulation output dict.

    Parameters
    ----------
    sim_output : dict with keys
        - "rankings_as_indices" : list[list[int]]
        - "matches_idx"         : array-like of int  (-1 = unmatched)
        - "student_attributes"  : pd.DataFrame
    max_p : int, optional
        Upper bound for the top-p sweep.  Defaults to max list length.
    categories : list[str], optional
        Single-attribute dimensions for conditioning.  Defaults to
        ["Residential District", "Home Language"] when present.
    conjunctions : list of lists of str, optional
        Multi-attribute conjunctions for Remark 4.
        E.g. [["Residential District", "Home Language"]].
    priority_col : str, optional
        Scalar priority column already in student_attributes (Remark 2).
    priority_matrix : ndarray (n_students × n_schools), optional
        Per-(student, school) priority matrix (Remark 2).  Takes precedence
        over priority_col when both are provided.
    plot_p_values : list[int], optional
        Which p values to highlight in the multi-line plots.  Defaults to
        [1, 3, 5] clipped to max_p.
    """
    required_keys = {"rankings_as_indices", "matches_idx", "student_attributes"}
    missing = required_keys - sim_output.keys()
    if missing:
        raise KeyError(f"sim_output is missing keys: {missing}")

    student_df = build_student_level_welfare(
        rankings_as_indices=sim_output["rankings_as_indices"],
        matches_idx=sim_output["matches_idx"],
        student_attributes=sim_output["student_attributes"],
        priority_col=priority_col,
        priority_matrix=priority_matrix,
    )

    k = max_p or int(student_df["list_length"].max())

    # Default p values to highlight in plots
    highlight_ps = plot_p_values or [p for p in [1, 3, 5] if p <= k]

    # Remark 3 – global unconditional sweep
    global_sweep = summarize_global_sweep(student_df, max_p=k)

    # Remark 1 – per-dimension sweeps
    by_length = summarize_top_p_sweep_by_list_length(student_df, max_p=k)
    by_priority = summarize_top_p_sweep_by_priority_percentile(
        student_df, max_p=k, n_bins=n_priority_bins
    )
    by_category = summarize_top_p_sweep_by_category(student_df, categories=categories, max_p=k)

    # Remark 4 – conjunctions
    by_conjunction = (
        summarize_top_p_sweep_by_conjunction(student_df, conjunctions, max_p=k)
        if conjunctions
        else {}
    )

    saved_paths: dict[str, str] = {}
    base_dir = None if output_dir is None else Path(output_dir)

    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

        # Student level
        p = base_dir / "student_level.csv"
        student_df.to_csv(p, index=False)
        saved_paths["student_level"] = str(p)

        # Global sweep
        p = base_dir / "global_top_p_sweep.csv"
        global_sweep.to_csv(p, index=False)
        saved_paths["global_sweep"] = str(p)
        saved_paths["global_sweep_plot"] = plot_global_sweep(
            global_sweep, base_dir / "global_top_p_sweep.png", show
        )

        # By list length
        p = base_dir / "top_p_sweep_by_list_length.csv"
        by_length.to_csv(p, index=False)
        saved_paths["top_p_sweep_by_list_length"] = str(p)
        saved_paths["list_length_plot"] = plot_top_p_sweep_vs_list_length(
            by_length, highlight_ps, base_dir / "top_p_sweep_vs_list_length.png", show
        )

        # By priority percentile
        if by_priority is not None:
            p = base_dir / "top_p_sweep_by_priority_percentile.csv"
            by_priority.to_csv(p, index=False)
            saved_paths["top_p_sweep_by_priority_percentile"] = str(p)
            saved_paths["priority_plot"] = plot_top_p_sweep_vs_priority_percentile(
                by_priority, highlight_ps,
                base_dir / "top_p_sweep_vs_priority_percentile.png", show
            )

        # By single category
        for category, summary in by_category.items():
            slug = category.lower().replace(" ", "_")
            p = base_dir / f"top_p_sweep_by_{slug}.csv"
            summary.to_csv(p, index=False)
            saved_paths[f"top_p_sweep_by_{slug}"] = str(p)
            saved_paths[f"{slug}_plot"] = plot_top_p_sweep_by_category(
                summary, category, highlight_ps,
                base_dir / f"top_p_sweep_by_{slug}.png", show
            )

        # By conjunction (Remark 4)
        for cols, summary in by_conjunction.items():
            slug = "_x_".join(c.lower().replace(" ", "_") for c in cols)
            p = base_dir / f"top_p_sweep_by_{slug}.csv"
            summary.to_csv(p, index=False)
            saved_paths[f"top_p_sweep_by_{slug}"] = str(p)

    else:
        plot_global_sweep(global_sweep, show=show)
        plot_top_p_sweep_vs_list_length(by_length, highlight_ps, show=show)
        if by_priority is not None:
            plot_top_p_sweep_vs_priority_percentile(by_priority, highlight_ps, show=show)
        for category, summary in by_category.items():
            plot_top_p_sweep_by_category(summary, category, highlight_ps, show=show)

    return WelfareResults(
        student_level=student_df,
        global_sweep=global_sweep,
        top_p_sweep_by_list_length=by_length,
        top_p_sweep_by_priority_percentile=by_priority,
        top_p_sweep_by_category=by_category,
        top_p_sweep_by_conjunction=by_conjunction,
        saved_paths=saved_paths or None,
    )
