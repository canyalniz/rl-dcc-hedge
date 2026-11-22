"""Descriptive-statistics summary of a run_experiment_tests.py run.

Reads <run_dir>/metrics.csv (the trajectories/ directory is not needed) and
produces, in <output_dir> (default <run_dir>/summary/):

  table_primary.csv / .md      table metrics (focus metrics + Annual Volatility) x strategies, synthetic (primary) and
                               test partition (companion); RL cells use the two-stage
                               aggregation (per-seed trajectory-means -> mean
                               [seed min, seed max] across seeds), baselines are
                               trajectory means (+- std on synthetic)
  table_primary.tex            the same two tables as LaTeX snippets (grid
                               rules, bold row-best value; makecell / bm /
                               adjustbox, cf. docs/packages.tex)
  dose_response.csv / .md      per-weight per-seed values of the focus metrics
                               (synthetic), with seed consistency of each weight step
  pairwise_tags.csv / .md      pairwise tag differences on HE (Variance Reduction)
                               and CVaR (95%) (synthetic); CSV covers all metrics
  vs_baselines.csv             per tag x baseline: mean gap and per-trajectory win
                               rates on VR / CVaR / Sharpe (synthetic)
  fig_frontier_synth.png       risk-return frontier panels (AnnRet vs Max drawdown,
                               AnnRet vs HE VR), baselines as points, RL with
                               seed whiskers
  fig_dose_response.png        focus metrics vs log_return weight, per-seed dots +
                               mean line, anchor/MV/MRR-C_10 reference lines
  fig_frontier_test.png        frontier panels on the historical test partition
  summary.csv                  tidy machine-readable table behind everything

Aggregation convention:
  synthetic   agents: mean over trajectories within each (tag, seed) (= one value
              per trained policy), then mean [min, max] across seeds; baselines and
              the anchor (deterministic): mean +- std over trajectories
  test        agents: mean [min, max] across per-seed values; baselines: the value

The hedge-effectiveness measure of record is HE (Variance Reduction); HE
(Ederington R2) remains available in summary.csv as the anchor co-movement
diagnostic but is not part of the primary tables/figures. A third, objective-
aligned effectiveness measure, CVaR Reduction (vs anchor) = 1 - CVaR_h/CVaR_u
(less negative tail = larger reduction), is computed from the anchor rows and
included everywhere. CVaR (95%) is the parametric Student-t CVaR with (nu, loc,
scale) MLE-fitted per trajectory (rl_dcc_hedge.utils.portfolio_metrics.student_t_cvar).

Usage:
  python summarize_experiment_tests.py <run_dir> [--output-dir DIR] [--dpi INT]
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

ANCHOR_ID = "unhedged_anchor"
SYNTH_PREFIX = "synth_test"
TEST_LABEL = "test_partition"

VR_METRIC = "HE (Variance Reduction)"
EDER_METRIC = "HE (Ederington R2)"  # anchor co-movement diagnostic, not an effectiveness measure here
CVAR_METRIC = "CVaR (95%)"  # per-trajectory MLE Student-t CVaR (rl_dcc_hedge.utils.portfolio_metrics.student_t_cvar)
CVAR_RED_METRIC = "CVaR Reduction (vs anchor)"
ANNRET_METRIC = "Annual Return"
SHARPE_METRIC = "Sharpe Ratio"
VOL_METRIC = "Annual Volatility"
# FOCUS_METRICS drive the dose-response outputs; TABLE_METRICS (a superset) the primary tables
FOCUS_METRICS = [CVAR_METRIC, ANNRET_METRIC, "Max Drawdown", SHARPE_METRIC, VR_METRIC, VOL_METRIC]
TABLE_METRICS = [CVAR_METRIC, ANNRET_METRIC, VOL_METRIC, "Max Drawdown", SHARPE_METRIC, VR_METRIC]
SUMMARY_METRICS = TABLE_METRICS + [EDER_METRIC]

BASELINE_ORDER = ["MV", "MRR-C_10", "MRR-C_100", "MRR-C_1000"]
METRIC_LABEL = {
    CVAR_METRIC: "CVaR 95",
    ANNRET_METRIC: "Annual return",
    VOL_METRIC: "Annual volatility",
    "Max Drawdown": "Max drawdown",
    SHARPE_METRIC: "Sharpe ratio",
    VR_METRIC: "HE (var. red.)",
    CVAR_RED_METRIC: "CVaR reduction",
    EDER_METRIC: "Anchor co-movement (corr2)",
}
DOSE_LABELS = [METRIC_LABEL[k] for k in FOCUS_METRICS]
TABLE_LABELS = [METRIC_LABEL[k] for k in TABLE_METRICS]

# Row-wise direction of "best" for the primary tables, under this script's sign
# conventions: CVaR (95%) and Max drawdown are reported as negative quantities
# (a larger, i.e. less negative, value means a smaller tail loss / drawdown),
# Annual volatility is a positive quantity where lower is better, and every
# other table metric is plain higher-is-better; so the best cell of each row
# is the one with the largest direction-adjusted value.
METRIC_DIRECTION = {CVAR_METRIC: +1, ANNRET_METRIC: +1, VOL_METRIC: -1, "Max Drawdown": +1,
                    SHARPE_METRIC: +1, VR_METRIC: +1, CVAR_RED_METRIC: +1}

TAG_WEIGHT_RE = re.compile(r"log_return:([0-9.]+)")
SEED_RE = re.compile(r"^\d+$")


@dataclass
class SummaryConfig:
    """Configuration of the run summary."""

    run_dir: str
    """experiment run directory (the one containing metrics.csv)"""

    output_dir: Optional[str] = None
    """where to write the summary artifacts; None = <run_dir>/summary"""

    dpi: int = 150
    """figure resolution"""


def _tag_weight(tag: str) -> Optional[float]:
    match = TAG_WEIGHT_RE.search(tag)
    return float(match.group(1)) if match else None


def _fmt_cell(row) -> str:
    if row["spread_kind"] == "seed_range":
        return f"{row['value']:.3f} [{row['spread_low']:.3f}, {row['spread_high']:.3f}]"
    if row["spread_kind"] == "traj_std":
        return f"{row['value']:.3f} ± {row['spread_low']:.3f}"
    return f"{row['value']:.3f}"


_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters for text mode."""
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in text)


def _latex_cell(row, bold: bool = False) -> str:
    """LaTeX table cell: value with the spread underneath in \\scriptsize; '--' when absent.

    ``bold`` emphasizes the primary value (\\bm, so the minus sign bolds too).
    """
    if row is None:
        return "--"
    value = f"{row['value']:.3f}"
    value = f"\\bm{{{value}}}" if bold else value
    if row["spread_kind"] == "seed_range":
        spread = f"$[{row['spread_low']:.3f},\\,{row['spread_high']:.3f}]$"
    elif row["spread_kind"] == "traj_std" and pd.notna(row["spread_low"]):
        spread = f"$\\pm\\,{row['spread_low']:.3f}$"
    else:
        return f"${value}$"
    return f"\\makecell[r]{{${value}$\\\\ {{\\scriptsize {spread}}}}}"


def _series_for(s1_mean: pd.Series, tag: str, metric: str) -> pd.Series:
    """Per-seed trajectory-means of (tag, metric); empty when absent."""
    try:
        return s1_mean.xs((tag, metric), level=("tag", "metric_name")).dropna()
    except KeyError:
        return pd.Series(dtype=float)


def main() -> None:
    args: SummaryConfig = tyro.cli(SummaryConfig, description=__doc__.split("\n")[0])

    metrics_path = os.path.join(args.run_dir, "metrics.csv")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"{metrics_path} not found; point --run-dir at a run_experiment_tests.py output")
    output_dir = args.output_dir or os.path.join(args.run_dir, "summary")
    os.makedirs(output_dir, exist_ok=True)

    m = pd.read_csv(metrics_path)
    if list(m.columns) != ["id_tag", "data", "metric_name", "metric_value"]:
        raise ValueError(f"Unexpected metrics.csv columns {list(m.columns)}")

    ### 1. structure discovery ###############################################
    ids = sorted(m["id_tag"].unique())
    agent_ids = [i for i in ids if ":" in i]
    for i in agent_ids:
        if not SEED_RE.match(i.rsplit("_", 1)[1]):
            raise ValueError(f"Agent id_tag {i!r} does not follow the '<tag>_<seed>' convention")
    baseline_ids = [i for i in ids if i not in agent_ids and i != ANCHOR_ID]
    if ANCHOR_ID not in ids:
        raise ValueError(f"Anchor id_tag {ANCHOR_ID!r} missing; its rows anchor the HE/CVaR-reduction columns")

    tags = sorted({i.rsplit("_", 1)[0] for i in agent_ids}, key=lambda t: (_tag_weight(t) is None, _tag_weight(t) or 0.0, t))
    tag_label = {t: (f"RL w={_tag_weight(t):g}" if _tag_weight(t) is not None else f"RL {t}") for t in tags}
    agent_ids_of = {t: [i for i in agent_ids if i.rsplit("_", 1)[0] == t] for t in tags}
    ordered_baselines = [b for b in BASELINE_ORDER if b in baseline_ids] + sorted(set(baseline_ids) - set(BASELINE_ORDER))
    reference_ids = [ANCHOR_ID] + ordered_baselines

    data_labels = sorted(m["data"].unique())
    synth_labels = [d for d in data_labels if d.startswith(SYNTH_PREFIX) and d != TEST_LABEL]
    has_test = TEST_LABEL in data_labels
    if not has_test:
        print(f"WARNING: no {TEST_LABEL!r} rows; the summary covers the synthetic data only")
    other = [d for d in data_labels if d not in synth_labels and d != TEST_LABEL]
    if other:
        print(f"WARNING: ignoring unrecognized data labels {other}")

    ### 2. aggregation #######################################################
    # CVaR reduction vs the anchor of the same dataset: 1 - CVaR_h / CVaR_anchor
    cvar = m[m["metric_name"] == CVAR_METRIC].pivot_table(index="id_tag", columns="data", values="metric_value")
    with np.errstate(divide="ignore", invalid="ignore"):
        cvar_red = 1.0 - cvar.div(cvar.loc[ANCHOR_ID], axis=1)
    red_rows = [
        {"id_tag": i, "data": d, "metric_name": CVAR_RED_METRIC, "metric_value": v}
        for i, row in cvar_red.iterrows()
        for d, v in row.items()
        if pd.notna(v)
    ]
    long = pd.concat([m, pd.DataFrame(red_rows)], ignore_index=True)

    synth = long[long["data"].isin(synth_labels)]
    test = long[long["data"] == TEST_LABEL]

    # synthetic, agents: stage 1 = trajectory-mean within (tag, seed)
    agent_synth = synth[synth["id_tag"].isin(agent_ids)].copy()
    agent_synth["tag"] = agent_synth["id_tag"].str.rsplit("_", n=1).str[0]
    agent_synth["seed"] = agent_synth["id_tag"].str.rsplit("_", n=1).str[1].astype(int)
    s1_mean = agent_synth.groupby(["tag", "seed", "metric_name"])["metric_value"].mean()

    # synthetic, baselines/anchor (deterministic): trajectory-level values
    ref_synth = synth[~synth["id_tag"].isin(agent_ids)].groupby(["id_tag", "metric_name"])["metric_value"]

    records: List[Dict[str, object]] = []

    def add(group, ident, label, weight, metric, data, value, low, high, kind):
        records.append({"group": group, "id": ident, "label": label, "w": weight, "metric": metric,
                        "data": data, "value": value, "spread_low": low, "spread_high": high, "spread_kind": kind})

    for metric in SUMMARY_METRICS:
        for t in tags:  # synthetic: mean [min, max] over per-seed trajectory-means
            vals = _series_for(s1_mean, t, metric).to_numpy()
            if len(vals):
                add("agent", t, tag_label[t], _tag_weight(t), metric, "synthetic", vals.mean(), vals.min(), vals.max(), "seed_range")
        for ident in reference_ids:  # synthetic: mean +- std over trajectories
            if (ident, metric) in ref_synth.groups:
                vals = ref_synth.get_group((ident, metric)).to_numpy()
                add("baseline" if ident != ANCHOR_ID else "anchor", ident, ident, None, metric, "synthetic",
                    vals.mean(), vals.std(ddof=1) if len(vals) > 1 else np.nan, np.nan, "traj_std")
        if has_test:
            t_mask = (test["metric_name"] == metric)
            for t in tags:  # test: mean [min, max] over per-seed values
                vals = test[t_mask & test["id_tag"].isin(agent_ids_of[t])]["metric_value"].dropna().to_numpy()
                if len(vals):
                    add("agent", t, tag_label[t], _tag_weight(t), metric, "test_partition", vals.mean(), vals.min(), vals.max(), "seed_range")
            for ident in reference_ids:  # test: single value
                vals = test[t_mask & (test["id_tag"] == ident)]["metric_value"].dropna().to_numpy()
                if len(vals):
                    add("baseline" if ident != ANCHOR_ID else "anchor", ident, ident, None, metric, "test_partition",
                        float(vals.mean()), np.nan, np.nan, "none")

    summary = pd.DataFrame(records)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    cell = {(r["id"], r["data"], r["metric"]): _fmt_cell(r) for _, r in summary.iterrows()}

    ### 3. primary table #####################################################
    strategy_ids = reference_ids + tags
    label_of = {b: b for b in ordered_baselines} | {ANCHOR_ID: ANCHOR_ID} | tag_label
    primary_frames = {}
    for data in (["synthetic", "test_partition"] if has_test else ["synthetic"]):
        primary_frames[data] = pd.DataFrame(
            {label_of[ident]: [cell.get((ident, data, metric), "") for metric in TABLE_METRICS] for ident in strategy_ids},
            index=pd.Index(TABLE_LABELS, name="metric"),
        )

    headers = {
        "synthetic": "Synthetic test trajectories — RL: mean over seeds of trajectory-means [seed min, max]; baselines: mean ± std over trajectories",
        "test_partition": "Historical test partition — RL: mean [min, max] over seeds; baselines: single value",
    }
    with open(os.path.join(output_dir, "table_primary.md"), "w") as f:
        f.write("".join(f"## {headers[d]}\n\n{t.to_markdown()}\n\n" for d, t in primary_frames.items()))
    pd.concat(primary_frames, names=["data", "metric"]).to_csv(os.path.join(output_dir, "table_primary.csv"))

    ### 3b. primary tables (LaTeX) ##########################################
    # Same content as table_primary.md as two self-contained booktabs table
    # environments (floats), using only packages from docs/packages.tex.
    summary_row = {(r["id"], r["data"], r["metric"]): r for _, r in summary.iterrows()}

    compact_w = all(_tag_weight(t) is not None for t in tags)

    def _strategy_header(ident: str, weight: Optional[float]) -> str:
        if compact_w and weight is not None:  # the RL group header announces w
            return f"${weight:g}$"
        if weight is not None:
            return f"RL $w{{=}}{weight:g}$"
        if ident == ANCHOR_ID:
            return "Unhedged anchor"
        m = re.match(r"^(?P<name>.+)_(?P<num>\d+)$", ident)  # e.g. MRR-C_10 -> MRR-C_{10}
        return f"{_latex_escape(m.group('name'))}$_{{{m.group('num')}}}$" if m else _latex_escape(ident)

    latex_captions = {
        "synthetic": ("Synthetic test trajectories: RL agents report the mean over seeds of trajectory-means, "
                      "with the seed range [min, max] beneath; the unhedged anchor and the baselines are "
                      "deterministic and report the mean $\\pm$ std over trajectories. RL columns are indexed "
                      "by the log-return reward weight $w$; ``--'' marks undefined entries (the anchor "
                      "against itself). The best value in each row is set in bold."),
        "test_partition": ("Historical test partition: RL agents report the mean over seeds, with the seed "
                           "range [min, max] beneath; the unhedged anchor and the baselines report the single "
                           "realized value. RL columns are indexed by the log-return reward weight $w$; "
                           "``--'' marks undefined entries. The best value in each row is set in bold."),
    }
    latex_labels = {"synthetic": "tab:primary-synthetic", "test_partition": "tab:primary-test-partition"}
    latex_titles = {"synthetic": "Table 1: synthetic test trajectories",
                    "test_partition": "Table 2: historical test partition"}

    n_ref, n_tag = len(reference_ids), len(tags)
    column_headers = ([_strategy_header(i, None) for i in reference_ids]
                      + ["$w$ = " + _strategy_header(t, _tag_weight(t)) for t in tags])
    ref_group = "Anchor" if not ordered_baselines else "Anchor and baselines"
    rl_group = "RL agents (reward weight $w$)" if compact_w else "RL agents"

    lines = [
        "% LaTeX version of table_primary (synthetic + test partition), generated by",
        f"% summarize_experiment_tests.py from {metrics_path}.",
        "% Requires the packages loaded in packages.tex (makecell, bm, adjustbox).",
    ]
    for data in primary_frames:  # same tables/order as the .md and .csv
        lines += ["", "% " + "-" * 74, f"% {latex_titles[data]}", "% " + "-" * 74,
                  "\\begin{table}[tb]",
                  "  \\centering",
                  f"  \\caption{{{latex_captions[data]}}}",
                  f"  \\label{{{latex_labels[data]}}}",
                  "  \\begin{adjustbox}{max width=\\linewidth}",
                  "    \\begin{tabular}{|l|" + "r|" * len(strategy_ids) + "}",
                  "      \\hline"]
        if n_tag:  # two-level header: anchor/baselines vs RL agents
            lines += [f"       & \\multicolumn{{{n_ref}}}{{|c|}}{{{ref_group}}} & \\multicolumn{{{n_tag}}}{{|c|}}{{{rl_group}}} \\\\",
                      f"      \\cline{{2-{n_ref + 1}}} \\cline{{{n_ref + 2}-{n_ref + n_tag + 1}}}"]
        lines.append("      Metric & " + " & ".join(column_headers) + " \\\\")
        lines.append("      \\hline")
        for metric in TABLE_METRICS:
            rows = [summary_row.get((ident, data, metric)) for ident in strategy_ids]
            values = np.array([r["value"] if r is not None else np.nan for r in rows])
            best = int(np.nanargmax(METRIC_DIRECTION[metric] * values)) if not np.isnan(values).all() else None
            cells = [_latex_cell(r, bold=(i == best)) for i, r in enumerate(rows)]
            lines.append("      " + _latex_escape(METRIC_LABEL[metric]) + " & " + " & ".join(cells) + " \\\\")
            lines.append("      \\hline")  # full grid: a rule beneath every body row
        lines += ["    \\end{tabular}", "  \\end{adjustbox}", "\\end{table}"]

    latex = "\n".join(lines) + "\n"
    latex.encode("ascii")  # guard: snippets must be plain ASCII (no raw unicode symbols)
    with open(os.path.join(output_dir, "table_primary.tex"), "w") as f:
        f.write(latex)

    ### 4. dose response (synthetic) ########################################
    all_seeds = sorted({int(i.rsplit("_", 1)[1]) for i in agent_ids})
    dose_rows = []
    for metric in FOCUS_METRICS:
        for t in tags:
            vals = _series_for(s1_mean, t, metric)
            row = {"metric": METRIC_LABEL[metric], "w": _tag_weight(t), "tag": t,
                   "mean": vals.mean(), "seed_min": vals.min(), "seed_max": vals.max()}
            row.update({f"seed_{s}": vals.get(s, np.nan) for s in all_seeds})
            dose_rows.append(row)
    dose = pd.DataFrame(dose_rows)

    consistency_rows = []
    ws = sorted(w for w in dose["w"].dropna().unique())
    per_seed_cols = [f"seed_{s}" for s in all_seeds]
    for metric in FOCUS_METRICS:
        sub = dose[dose["metric"] == METRIC_LABEL[metric]].set_index("w")
        for w_lo, w_hi in zip(ws[:-1], ws[1:]):
            diffs = (sub.loc[w_hi, per_seed_cols] - sub.loc[w_lo, per_seed_cols]).dropna()
            consistency_rows.append({
                "metric": METRIC_LABEL[metric], "step": f"w {w_lo:g}->{w_hi:g}",
                "mean_change": sub.loc[w_hi, "mean"] - sub.loc[w_lo, "mean"],
                "seeds_up": int((diffs > 0).sum()), "seeds_down": int((diffs < 0).sum()), "n_seeds": len(diffs),
            })
    consistency = pd.DataFrame(consistency_rows)
    dose[["metric", "w", "mean", "seed_min", "seed_max"] + [f"seed_{s}" for s in all_seeds]].to_csv(
        os.path.join(output_dir, "dose_response.csv"), index=False)

    fmt3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    with open(os.path.join(output_dir, "dose_response.md"), "w") as f:
        f.write("Mean over seeds of trajectory-means (synthetic); w = log_return reward weight\n\n")
        f.write(dose.pivot(index="metric", columns="w", values="mean").loc[DOSE_LABELS].map(fmt3).to_markdown() + "\n\n")
        f.write("Seed consistency of each weight step (number of seeds moving up)\n\n")
        f.write(consistency.pivot(index="metric", columns="step", values="seeds_up")
                .loc[DOSE_LABELS].map(lambda v: "" if pd.isna(v) else f"{int(v)}\u2191").to_markdown() + "\n")

    ### 5. pairwise tag differences (synthetic) #############################
    pair_rows = []
    for metric in SUMMARY_METRICS:
        for i, t_a in enumerate(tags):
            for t_b in tags[i + 1:]:
                va, vb = _series_for(s1_mean, t_a, metric), _series_for(s1_mean, t_b, metric)
                d = (va - vb).dropna()
                if len(d):
                    pair_rows.append({"metric": metric, "tag_a": tag_label[t_a], "tag_b": tag_label[t_b],
                                      "mean_diff_a_minus_b": d.mean(), "min_diff": d.min(), "max_diff": d.max(),
                                      "seeds_a_better": int((d > 0).sum()), "n_seeds": len(d)})
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(os.path.join(output_dir, "pairwise_tags.csv"), index=False)
    with open(os.path.join(output_dir, "pairwise_tags.md"), "w") as f:
        for metric in (VR_METRIC, CVAR_METRIC, CVAR_RED_METRIC):
            sub = pairwise[pairwise["metric"] == metric]
            if not len(sub):
                continue
            tbl = sub.assign(cell=sub.apply(
                lambda r: f"{r['mean_diff_a_minus_b']:+.3f} ({int(r['seeds_a_better'])}/{int(r['n_seeds'])} +)", axis=1)
            ).pivot(index="tag_a", columns="tag_b", values="cell")
            f.write(f"## {METRIC_LABEL[metric]} — mean diff A − B (seeds with A better)\n\n{tbl.to_markdown()}\n\n")

    ### 6. agents vs baselines (synthetic, per-trajectory) ##################
    vs_rows = []
    for metric in (VR_METRIC, CVAR_METRIC, SHARPE_METRIC):
        piv = synth[synth["metric_name"] == metric].pivot_table(index="id_tag", columns="data", values="metric_value")
        base_piv = piv.loc[[b for b in ordered_baselines if b in piv.index]]
        for t in tags:
            tag_piv = piv.loc[[i for i in agent_ids_of[t] if i in piv.index]]
            if not len(tag_piv):
                continue
            for b in base_piv.index:
                diff = (tag_piv - base_piv.loc[b]).to_numpy().ravel()  # higher is better for all three metrics here
                vs_rows.append({"tag": tag_label[t], "baseline": b, "metric": METRIC_LABEL[metric],
                                "mean_gap": np.nanmean(diff), "win_rate": np.nanmean(diff > 0)})
    vs = pd.DataFrame(vs_rows)
    vs.to_csv(os.path.join(output_dir, "vs_baselines.csv"), index=False)

    ### 7. figures ###########################################################
    plt.rcParams.update({"figure.dpi": args.dpi, "savefig.dpi": args.dpi, "font.size": 9})

    def get(ident: str, metric: str, data: str) -> pd.Series:
        return summary[(summary["id"] == ident) & (summary["data"] == data) & (summary["metric"] == metric)]

    def val(ident: str, metric: str, data: str, default: float = np.nan) -> float:
        """Scalar summary value; ``default`` when the row is absent (e.g. the anchor's own HE = 0 by definition)."""
        row = get(ident, metric, data)
        return row["value"].item() if len(row) else default

    def lo_hi(ident: str, metric: str, data: str) -> Tuple[float, float]:
        row = get(ident, metric, data)
        return row["spread_low"].item(), row["spread_high"].item()

    def frontier(data: str, path: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), constrained_layout=True)
        base_markers = {"MV": "^", "MRR-C_1": "s", "MRR-C_2": "D", "MRR-C_10": "P"}
        for ax, x_metric, x_lab in ((axes[0], "Max Drawdown", METRIC_LABEL["Max Drawdown"]), (axes[1], VR_METRIC, METRIC_LABEL[VR_METRIC])):
            ax.scatter(val(ANCHOR_ID, x_metric, data, default=0.0), val(ANCHOR_ID, ANNRET_METRIC, data),
                       marker="X", s=90, color="black", zorder=3, label="unhedged anchor")
            for b in ordered_baselines:
                ax.scatter(val(b, x_metric, data), val(b, ANNRET_METRIC, data),
                           marker=base_markers.get(b, "o"), s=60, color="dimgray", zorder=3, label=b)
            x = np.array([val(t, x_metric, data) for t in tags])
            y = np.array([val(t, ANNRET_METRIC, data) for t in tags])
            xlo, xhi = np.array([lo_hi(t, x_metric, data)[0] for t in tags]), np.array([lo_hi(t, x_metric, data)[1] for t in tags])
            ylo, yhi = np.array([lo_hi(t, ANNRET_METRIC, data)[0] for t in tags]), np.array([lo_hi(t, ANNRET_METRIC, data)[1] for t in tags])
            order = np.argsort([_tag_weight(t) if _tag_weight(t) is not None else np.inf for t in tags])
            ax.plot(x[order], y[order], "-", color="lightsteelblue", zorder=2)
            ax.errorbar(x, y, xerr=[x - xlo, xhi - x], yerr=[y - ylo, yhi - y], fmt="o", ms=5,
                        color="steelblue", ecolor="steelblue", elinewidth=1, capsize=2, alpha=0.9,
                        zorder=4, label="RL (whiskers: seeds)")
            for i in order:
                w = _tag_weight(tags[i])
                ax.annotate(f"w={w:g}" if w is not None else tags[i], (x[i], y[i]),
                            textcoords="offset points", xytext=(5, 4), fontsize=8, color="steelblue")
            ax.set_xlabel(x_lab)
            ax.set_ylabel(METRIC_LABEL[ANNRET_METRIC])
        axes[0].set_title(f"Return vs drawdown — {data}")
        axes[1].set_title(f"Return vs hedge effectiveness — {data}")
        axes[0].legend(fontsize=7)
        fig.savefig(path)
        plt.close(fig)

    frontier("synthetic", os.path.join(output_dir, "fig_frontier_synth.png"))
    if has_test:
        frontier("test_partition", os.path.join(output_dir, "fig_frontier_test.png"))

    # dose-response panels: per-seed dots + mean line, reference lines from trajectory means
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True, sharex=True)
    seed_cols = [f"seed_{s}" for s in all_seeds]
    for ax, metric in zip(axes.flat, FOCUS_METRICS):
        sub = dose[dose["metric"] == METRIC_LABEL[metric]]
        for _, row in sub.iterrows():
            dots = row[seed_cols].to_numpy(dtype=float)
            ax.scatter([row["w"]] * len(seed_cols), dots, s=14, color="gray", alpha=0.7, zorder=2)
        ax.plot(sub["w"], sub["mean"], "o-", color="steelblue", ms=5, zorder=3, label="RL mean over seeds")
        for ident, color, ls in ((ANCHOR_ID, "black", "--"), ("MV", "dimgray", "-."), ("MRR-C_10", "lightgray", ":")):
            if (ident, metric) in ref_synth.groups:
                ax.axhline(ref_synth.get_group((ident, metric)).mean(), color=color, ls=ls, lw=1, label=ident)
        ax.set_title(METRIC_LABEL[metric], fontsize=10)
        ax.set_xticks(sorted(sub["w"].dropna().unique()))
        ax.margins(y=0.15)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"mean over {len(synth_labels)} trajectories")
    for ax in axes[1, :]:
        ax.set_xlabel("log_return reward weight w")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Dose-response of the log_return reward weight (synthetic test trajectories)", fontsize=11)
    fig.savefig(os.path.join(output_dir, "fig_dose_response.png"))
    plt.close(fig)

    ### 8. console report ####################################################
    for data, table in primary_frames.items():
        print(f"\n## {headers[data]}\n\n{table.to_markdown()}")
    print("\n" + "=" * 100)
    print(f"Summary of {metrics_path}")
    print(f"  strategies: 1 anchor + {len(ordered_baselines)} baselines + {len(tags)} tags ({len(agent_ids)} agent runs)")
    print(f"  datasets: {len(synth_labels)} synthetic trajectories" + (" + test partition" if has_test else ""))
    print("Artifacts:")
    for name in sorted(os.listdir(output_dir)):
        print(f"  {os.path.join(output_dir, name)}")


if __name__ == "__main__":
    main()
