"""Read frozen predictions once and build paper evidence in an isolated directory.

No checkpoint loading, model construction, fitting, training or GPU access.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import paper_evidence_math as numeric

TASK = "CAI_V3_PAPER_EVIDENCE_R1_e2a11154"
BASE = "e2a1115468da6e8695321204a13fa9a5322ea809"
DATA = Path("results/cai_agent_v3/new_protocol")
W2 = Path("results/cai_agent_v3/w2_replay/r1_292b1c74")
W3 = Path("results/cai_agent_v3/w3_pilot/r1_0e11452a")
A3 = Path("artifacts/cai_agent_v3/w3_pilot/r1_0e11452a")
RUN = Path("results/cai_agent_v3/paper_evidence/r1_e2a11154")
ART = Path("artifacts/cai_agent_v3/paper_evidence/r1_e2a11154")
MAIN = "VLM_SPATIAL_FEEDBACK"
FULL = "FULL_SCAN_P_ALL"
BUDGETS = np.array([0.0, 0.0625, 0.125, 0.1875, 0.25])
SCOPE = "POSTHOC_SELECTED_VALID_CONDITIONAL"
NONADAPTIVE = [
    "CENTER_FIRST",
    "GEOMETRY_SPREAD",
    "SERPENTINE",
    "RANDOM",
    "LEARNED_STATIC_TRUE",
]
CONTRASTS = [
    ("fixed", "GEOMETRY_SPREAD", "area"),
    ("feedback", "VLM_SPATIAL_OPEN_LOOP", "area"),
    ("vlm_early", "NO_VLM_SPATIAL_FEEDBACK", "early_area"),
    ("spatial", "VLM_MEAN_FEEDBACK", "area"),
]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_episode(row):
    costs = np.array([float(x) for x in row["costs"].split(";")])
    pred = np.array([float(x) for x in row["predictions_mpa"].split(";")])
    cells = [int(x) for x in row["cells"].split(";") if x]
    numeric.held(costs, pred, [0.0, 0.25])
    if len(cells) + 1 != len(costs) or len(set(cells)) != len(cells):
        raise ValueError("Invalid frozen actions")
    trace = json.loads(row["execution_trace"])
    pixels = 0
    if len(trace) != len(cells):
        raise ValueError("Missing actual execution trace")
    for j, t in enumerate(trace):
        if (
            t["cell"] != cells[j]
            or t["action_index"] != j + 1
            or t["visible_cells_before"] != cells[:j]
            or t["before_cost"] != costs[j]
            or t["after_cost"] != costs[j + 1]
            or t["before_prediction_mpa"] != pred[j]
            or t["after_prediction_mpa"] != pred[j + 1]
            or t["actor_call_index"]
            != (None if row["actor_state_dict_sha256"] == "FIXED_ORDER" else j + 1)
        ):
            raise ValueError("Trace fields disagree with saved episode")
        if any(
            len(t[k]) != 64 or t[k][cells[j]] != "1"
            for k in ("environment_legal", "proposal_legal")
        ):
            raise ValueError("Illegal saved action")
        pixels += int(t["new_pixels"])
        if (
            pixels != t["cumulative_pixels"]
            or abs(pixels / int(row["native_pixels"]) - costs[j + 1]) > 1e-12
        ):
            raise ValueError("Pixel trace mismatch")
    return dict(row, cost_array=costs, pred_array=pred, cell_array=cells, trace=trace)


def table_fragments(run, stem, rows, columns):
    """CSV is authoritative; identical cells populate Markdown and escaped LaTeX."""

    def fmt(v):
        if v is None:
            return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{v:.6f}"
        return str(v)

    cells = [[fmt(r.get(k)) for k in columns] for r in rows]
    (run / (stem + ".md")).write_text(
        "| "
        + " | ".join(columns)
        + " |\n| "
        + " | ".join(["---"] * len(columns))
        + " |\n"
        + "".join("| " + " | ".join(row) + " |\n" for row in cells)
    )

    def tex(s):
        return (
            s.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
        )

    (run / (stem + ".tex")).write_text(
        "\\begin{tabular}{"
        + "l" * len(columns)
        + "}\n\\hline\n"
        + " & ".join(map(tex, columns))
        + r" \\"
        + "\n\\hline\n"
        + "".join(" & ".join(map(tex, row)) + r" \\" + "\n" for row in cells)
        + "\\hline\n\\end{tabular}\n"
    )


class Evidence:
    def __init__(self, root):
        self.root = root
        self.out = root / RUN
        self.art = root / ART
        self.out.mkdir(parents=True, exist_ok=True)
        self.art.mkdir(parents=True, exist_ok=True)
        if (self.out / "analysis_manifest.json").exists():
            raise FileExistsError(
                "Completed analysis exists; use export only, do not recompute bootstrap"
            )
        self.scope = json.loads(
            (root / "docs/cai/paper_evidence/PAPER_EVIDENCE_SCOPE.json").read_text()
        )
        self.methods = self.scope["methods"]
        names = [
            "final_manifest.json",
            "actor_manifests.json",
            "policy_validation_episodes.csv.gz",
            "policy_pilot_metrics.csv",
            "absolute_cai_performance.csv",
            "pilot_effects.csv",
            "per_domain_metrics.csv",
            "per_domain_effects.csv",
            "paired_effects.csv",
            "p_all_saved_reference.json",
            "case_manifest.csv",
            "figure_manifest.json",
        ]
        self.inputs = (
            [W3 / n for n in names]
            + [
                A3 / n
                for n in [
                    "W3_PILOT_HANDOFF.md",
                    "RESULTS_AND_CLAIM_BOUNDARIES.md",
                    "REQUIREMENTS_REVIEW.json",
                    "INPUT_BINDINGS.json",
                ]
            ]
            + [
                DATA / n
                for n in [
                    "feature_bank_index.csv",
                    "split_manifest.csv",
                    "vlm_actor_features_fit.csv",
                    "vlm_manifest_fit.json",
                ]
            ]
        )
        self.pointer = json.loads(
            (root / W3 / "p_all_saved_reference.json").read_text()
        )
        self.full_path = Path(self.pointer["source_predictions"])
        self.inputs.append(self.full_path)
        fig = json.loads((root / W3 / "figure_manifest.json").read_text())
        self.case_paths = [Path(p) for c in fig["cases"] for p in c["paths"]]
        self.inputs += self.case_paths
        # Hash only necessary frozen files at entry and exit; no .pt or feature shards.
        self.hashes = {str(p): sha(root / p) for p in self.inputs}
        # Read every required source; retain tables as source cross-checks.
        self.sources = {
            str(p): read_csv(root / p)
            if p.suffix == ".csv"
            else json.loads((root / p).read_text())
            if p.suffix == ".json"
            else (root / p).read_text()
            if p.suffix == ".md"
            else None
            for p in self.inputs
        }
        with gzip.open(root / W3 / "policy_validation_episodes.csv.gz", "rt") as f:
            self.episodes = [validate_episode(r) for r in csv.DictReader(f)]
        index = read_csv(root / DATA / "feature_bank_index.csv")
        self.index = index
        self.keys = [r["specimen_key"] for r in index if r["split"] == "VALID"]
        self.meta = {r["specimen_key"]: r for r in index if r["split"] == "VALID"}
        self.domains = np.array([self.meta[k]["dataset_id"] for k in self.keys])
        self.groups = [self.meta[k]["capture_group_id"] for k in self.keys]
        assert (
            len(self.episodes) == 650
            and len(self.keys) == 50
            and len(set(self.groups)) == 48
            and len(set(self.domains)) == 6
        )
        assert {r["method"] for r in self.episodes} == set(self.methods)
        self.targets = {}
        self.by_method = {}
        for method in self.methods:
            rows = [r for r in self.episodes if r["method"] == method]
            assert len(rows) == (250 if method == "RANDOM" else 50)
            self.by_method[method] = rows
            repeats = sorted({int(r["run"]) for r in rows})
            assert len(repeats) == (5 if method == "RANDOM" else 1)
            assert len({(r["specimen_key"], r["run"]) for r in rows}) == len(rows)
            for repeat in repeats:
                assert {
                    r["specimen_key"] for r in rows if int(r["run"]) == repeat
                } == set(self.keys)
            for r in rows:
                key = r["specimen_key"]
                assert key in self.meta
                assert (
                    r["dataset_id"] == self.meta[key]["dataset_id"]
                    and r["capture_group_id"] == self.meta[key]["capture_group_id"]
                )
                y = float(r["target_mpa"])
                assert key not in self.targets or self.targets[key] == y
                self.targets[key] = y
        self.y = np.array([self.targets[k] for k in self.keys])
        assert self.pointer["selected_update"] == 1750
        bound = json.loads((root / A3 / "INPUT_BINDINGS.json").read_text())[
            "predictors"
        ][0]
        assert (
            bound["checkpoint_sha256"] == self.pointer["checkpoint_sha256"]
            and bound["checkpoint_path"] == self.pointer["checkpoint_path"]
        )
        with np.load(root / self.full_path, allow_pickle=False) as z:
            self.full = numeric.map_full(
                index,
                z["full_specimen_indices"],
                z["full_predictions_mpa"],
                z["full_targets_mpa"],
                self.targets,
            )
        for r in self.full:
            r.update(
                dataset_id=self.meta[r["specimen_key"]]["dataset_id"],
                capture_group_id=self.meta[r["specimen_key"]]["capture_group_id"],
                abs_error_mpa=abs(r["full_prediction_mpa"] - r["target_mpa"]),
                squared_error_mpa2=(r["full_prediction_mpa"] - r["target_mpa"]) ** 2,
                model_hash=self.pointer["checkpoint_sha256"],
                source_npz=str(self.full_path),
                cost=1.0,
            )
        fullmap = {r["specimen_key"]: r for r in self.full}
        self.full_errors = np.array(
            [fullmap[k]["full_prediction_mpa"] - self.targets[k] for k in self.keys]
        )
        stat = numeric.aggregate(self.full_errors[None, :, None], self.y)
        self.full_metrics = {k: float(stat[k][0]) for k in ("mae", "mse", "rmse", "r2")}
        for name, field in [
            ("mae", "full_mae_mpa"),
            ("rmse", "full_rmse_mpa"),
            ("r2", "full_r2"),
        ]:
            assert abs(self.full_metrics[name] - self.pointer[field]) < 1e-9
        self.event_grid = np.unique(
            [
                0.0,
                0.25,
                *[
                    float(c)
                    for r in self.episodes
                    for c in r["cost_array"]
                    if 0 <= c <= 0.25
                ],
            ]
        )
        self.weights = numeric.bootstrap_weights(self.domains, self.groups)
        np.savez_compressed(
            self.out / "bootstrap_group_weights.npz",
            weights=self.weights,
            specimen_keys=np.array(self.keys),
            domains=self.domains,
            capture_groups=np.array(self.groups),
        )
        write_json(
            self.art / "INPUT_BINDINGS.json",
            {
                "task_id": TASK,
                "source_sha": BASE,
                "source_files_sha256": self.hashes,
                "physical_n": 50,
                "capture_groups": 48,
                "domains": 6,
                "episodes": 650,
                "full_reference": self.pointer,
                "index_mapping": "ORIGINAL_FEATURE_INDEX_ROW_ORDER",
                "bootstrap": {
                    "seed": 2026091401,
                    "replicates": 5000,
                    "shared_all_methods_budgets": True,
                    "weights": "bootstrap_group_weights.npz",
                },
                "checkpoint_loading": False,
            },
        )

    def curves(self, method, grid):
        rows = self.by_method[method]
        repeats = sorted({r["run"] for r in rows}, key=int)
        lookup = {(r["run"], r["specimen_key"]): r for r in rows}
        predictions = []
        actual = []
        pixels = []
        for repeat in repeats:
            pp = []
            cc = []
            nn = []
            for key in self.keys:
                r = lookup[(repeat, key)]
                cost = r["cost_array"]
                indices = np.searchsorted(cost, grid, side="right") - 1
                pp.append(numeric.held(cost, r["pred_array"], grid))
                cc.append(cost[indices])
                nn.append(
                    np.r_[0, np.cumsum([t["new_pixels"] for t in r["trace"]])][indices]
                )
            predictions.append(pp)
            actual.append(cc)
            pixels.append(nn)
        actual = np.asarray(actual)
        pixels = np.asarray(pixels)
        stat = numeric.aggregate(
            np.asarray(predictions) - self.y[None, :, None], self.y
        )
        return dict(
            stat,
            actual_cost_mean=actual.mean(axis=(0, 1)),
            actual_cost_min=actual.min(axis=(0, 1)),
            actual_cost_max=actual.max(axis=(0, 1)),
            native_pixels_mean=pixels.mean(axis=(0, 1)),
            native_pixels_min=pixels.min(axis=(0, 1)),
            native_pixels_max=pixels.max(axis=(0, 1)),
            residual=np.asarray(predictions) - self.y[None, :, None],
            actual=actual,
            pixels=pixels,
        )

    def area_vectors(self, method):
        data = defaultdict(list)
        for r in self.by_method[method]:
            a = numeric.area(r["cost_array"], r["pred_array"], float(r["target_mpa"]))
            e = numeric.area(
                r["cost_array"], r["pred_array"], float(r["target_mpa"]), end=0.0625
            )
            assert (
                abs(a - float(r["left_error_area_mpa"])) < 1e-12
                and abs(e - float(r["early_left_error_area_mpa"])) < 1e-12
            )
            data[r["specimen_key"]].append([a, e])
        result = np.array([np.mean(data[k], axis=0) for k in self.keys])
        return {"area": result[:, 0], "early_area": result[:, 1]}

    def interval(self, values, domain_equal=False):
        draws = numeric.weighted(
            values, self.weights, self.domains if domain_equal else None
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        return {
            "ci_low": float(low),
            "ci_high": float(high),
            "confidence": 0.95,
            "bootstrap_replicates": 5000,
            "bootstrap_seed": 2026091401,
            "inference_scope": SCOPE,
            "simultaneous": False,
        }

    def analyze(self):
        self.primary = {m: self.curves(m, BUDGETS) for m in self.methods}
        self.event = {m: self.curves(m, self.event_grid) for m in self.methods}
        self.areas = {m: self.area_vectors(m) for m in self.methods}
        episode_index = []
        events = []
        for rownum, r in enumerate(self.episodes):
            meta = {
                k: r[k]
                for k in [
                    "specimen_key",
                    "dataset_id",
                    "capture_group_id",
                    "method",
                    "run",
                    "seed_panel",
                    "training_seed",
                    "checkpoint_update",
                    "actor_state_dict_sha256",
                ]
            }
            episode_index.append(
                dict(
                    meta,
                    source_row=rownum,
                    source=str(W3 / "policy_validation_episodes.csv.gz"),
                    states=len(r["cost_array"]),
                    action_count=len(r["cell_array"]),
                    final_cost=float(r["cost_array"][-1]),
                    native_pixels=int(r["native_pixels"]),
                    target_mpa=float(r["target_mpa"]),
                )
            )
            for t in r["trace"]:
                event = dict(
                    meta,
                    **t,
                    error_before_mpa=abs(
                        t["before_prediction_mpa"] - float(r["target_mpa"])
                    ),
                    error_after_mpa=abs(
                        t["after_prediction_mpa"] - float(r["target_mpa"])
                    ),
                    analysis_only_target=True,
                )
                event["post_observation_error_reduction_mpa"] = (
                    event["error_before_mpa"] - event["error_after_mpa"]
                )
                event["visible_cells_before"] = json.dumps(
                    event["visible_cells_before"]
                )
                events.append(event)
        write_csv(self.out / "frozen_episode_index.csv", episode_index)
        write_csv(self.out / "acquisition_events.csv", events)
        self.events = events
        write_csv(self.out / "full_scan_predictions.csv", self.full)
        write_json(
            self.out / "full_scan_reference.json",
            dict(
                method=FULL,
                cost_support=[1.0],
                area=None,
                **self.full_metrics,
                source=self.pointer,
                input="FULL_CSCAN_PLUS_EXISTING_SURFACE",
                physical_n=50,
            ),
        )
        same = []
        domains = []
        curves = []
        method_summary = []
        for m in self.methods:
            primary = self.primary[m]
            av = {
                k: float(numeric.weighted(v, np.ones((1, 50)), self.domains)[0])
                for k, v in self.areas[m].items()
            }
            method_summary.append(
                dict(
                    method=m,
                    **av,
                    endpoint_mae=float(primary["mae"][-1]),
                    endpoint_rmse=float(primary["rmse"][-1]),
                    endpoint_r2=float(primary["r2"][-1]),
                )
            )
            for i, b in enumerate(BUDGETS):
                same.append(
                    dict(
                        method=m,
                        budget=float(b),
                        physical_n=50,
                        **{
                            k: float(primary[k][i])
                            for k in [
                                "mae",
                                "mse",
                                "rmse",
                                "r2",
                                "actual_cost_mean",
                                "actual_cost_min",
                                "actual_cost_max",
                                "native_pixels_mean",
                                "native_pixels_min",
                                "native_pixels_max",
                            ]
                        },
                        unused_budget_mean=float(b - primary["actual_cost_mean"][i]),
                        area=av["area"],
                        early_area=av["early_area"],
                        aggregation="POOLED_LOSS; AREA_DOMAIN_EQUAL",
                    )
                )
                for d in sorted(set(self.domains)):
                    mask = self.domains == d
                    stat = numeric.aggregate(
                        primary["residual"][:, mask, :], self.y[mask]
                    )
                    domains.append(
                        dict(
                            method=m,
                            dataset_id=d,
                            budget=float(b),
                            physical_n=int(mask.sum()),
                            **{
                                k: float(stat[k][i])
                                for k in ["mae", "mse", "rmse", "r2"]
                            },
                            actual_cost_mean=float(
                                primary["actual"][:, mask, i].mean()
                            ),
                            actual_cost_min=float(primary["actual"][:, mask, i].min()),
                            actual_cost_max=float(primary["actual"][:, mask, i].max()),
                            area=float(self.areas[m]["area"][mask].mean()),
                            early_area=float(self.areas[m]["early_area"][mask].mean()),
                        )
                    )
            for i, b in enumerate(self.event_grid):
                curves.append(
                    dict(
                        method=m,
                        budget=float(b),
                        **{
                            k: float(self.event[m][k][i])
                            for k in [
                                "mae",
                                "rmse",
                                "r2",
                                "actual_cost_mean",
                                "actual_cost_min",
                                "actual_cost_max",
                            ]
                        },
                    )
                )
        same.append(
            dict(
                method=FULL,
                budget=1.0,
                physical_n=50,
                **self.full_metrics,
                actual_cost_mean=1.0,
                actual_cost_min=1.0,
                actual_cost_max=1.0,
                area=None,
                early_area=None,
                aggregation="POOLED_FULL_REFERENCE_NOT_SAME_COST",
            )
        )
        write_csv(self.out / "same_cost_metrics.csv", same)
        write_csv(self.out / "same_cost_metrics_by_domain.csv", domains)
        write_csv(self.out / "cost_error_event_curves.csv", curves)
        write_csv(self.out / "method_summary.csv", method_summary)
        old = read_csv(self.root / W3 / "absolute_cai_performance.csv")
        maximum = 0.0
        for r in old:
            i = list(BUDGETS).index(float(r["budget"]))
            for k, field in [
                ("mae", "mae_mpa"),
                ("rmse", "rmse_mpa"),
                ("r2", "r2_repeat_mean"),
            ]:
                maximum = max(
                    maximum, abs(self.primary[r["method"]][k][i] - float(r[field]))
                )
        assert maximum < 1e-9
        self.validation = {
            "frozen_five_budget_max_difference": maximum,
            "full_reference_max_difference": max(
                abs(self.full_metrics[k] - self.pointer[field])
                for k, field in [
                    ("mae", "full_mae_mpa"),
                    ("rmse", "full_rmse_mpa"),
                    ("r2", "full_r2"),
                ]
            ),
        }
        paired = []
        summary = []
        mechanism = []
        area_pairs = []
        for m in self.methods:
            if m == MAIN:
                continue
            for i, b in enumerate(BUDGETS):
                losses = (
                    self.primary[m]["physical_abs"][:, i]
                    - self.primary[MAIN]["physical_abs"][:, i]
                )
                for j, key in enumerate(self.keys):
                    paired.append(
                        {
                            "comparator": m,
                            "budget": float(b),
                            "specimen_key": key,
                            "dataset_id": self.domains[j],
                            "capture_group_id": self.groups[j],
                            "main_abs_error": float(
                                self.primary[MAIN]["physical_abs"][j, i]
                            ),
                            "control_abs_error": float(
                                self.primary[m]["physical_abs"][j, i]
                            ),
                            "gain_mpa": float(losses[j]),
                        }
                    )
                summary.append(
                    dict(
                        comparator=m,
                        budget=float(b),
                        mae_gain_mpa=float(losses.mean()),
                        rmse_gain_mpa=float(
                            self.primary[m]["rmse"][i] - self.primary[MAIN]["rmse"][i]
                        ),
                        r2_main_minus_control=float(
                            self.primary[MAIN]["r2"][i] - self.primary[m]["r2"][i]
                        ),
                        **self.interval(losses),
                    )
                )
            record = {
                "comparator": m,
                "area_gain": float(
                    numeric.weighted(
                        self.areas[m]["area"] - self.areas[MAIN]["area"],
                        np.ones((1, 50)),
                        self.domains,
                    )[0]
                ),
                "early_area_gain": float(
                    numeric.weighted(
                        self.areas[m]["early_area"] - self.areas[MAIN]["early_area"],
                        np.ones((1, 50)),
                        self.domains,
                    )[0]
                ),
                "endpoint_mae_gain": float(
                    self.primary[m]["mae"][-1] - self.primary[MAIN]["mae"][-1]
                ),
                "endpoint_rmse_gain": float(
                    self.primary[m]["rmse"][-1] - self.primary[MAIN]["rmse"][-1]
                ),
                "positive_favors_main": True,
            }
            mechanism.append(record)
            for j, key in enumerate(self.keys):
                area_pairs.append(
                    {
                        "comparator": m,
                        "specimen_key": key,
                        "dataset_id": self.domains[j],
                        "capture_group_id": self.groups[j],
                        "area_gain": float(
                            self.areas[m]["area"][j] - self.areas[MAIN]["area"][j]
                        ),
                        "early_area_gain": float(
                            self.areas[m]["early_area"][j]
                            - self.areas[MAIN]["early_area"][j]
                        ),
                    }
                )
        self.effects = []
        for name, m, field in CONTRASTS:
            diff = self.areas[m][field] - self.areas[MAIN][field]
            self.effects.append(
                dict(
                    effect=name,
                    comparator=m,
                    estimand=field,
                    gain_mpa=float(
                        numeric.weighted(diff, np.ones((1, 50)), self.domains)[0]
                    ),
                    **self.interval(diff, True),
                )
            )
        write_csv(self.out / "same_cost_paired_losses.csv", paired)
        write_csv(self.out / "same_cost_paired_summary.csv", summary)
        write_csv(self.out / "paired_area_losses.csv", area_pairs)
        write_csv(self.out / "mechanism_effects.csv", mechanism)
        write_csv(self.out / "mechanism_area_intervals.csv", self.effects)
        gaps = []
        full_pairs = []
        for m in self.methods:
            for i, b in enumerate(BUDGETS):
                gap = float(self.primary[m]["mae"][i] - self.full_metrics["mae"])
                gaps.append(
                    {
                        "method": m,
                        "budget": float(b),
                        "full_reference_cost": 1.0,
                        "partial_mae": float(self.primary[m]["mae"][i]),
                        "full_mae": self.full_metrics["mae"],
                        "mae_gap_mpa": gap,
                        "positive_excess_mpa": max(0, gap),
                        "interpretation": "CROSS_COST_INFORMATION_TRADEOFF_NOT_NONINFERIORITY",
                    }
                )
                for j, key in enumerate(self.keys):
                    full_pairs.append(
                        {
                            "method": m,
                            "budget": float(b),
                            "specimen_key": key,
                            "partial_abs_error": float(
                                self.primary[m]["physical_abs"][j, i]
                            ),
                            "full_abs_error": float(abs(self.full_errors[j])),
                            "gap_mpa": float(
                                self.primary[m]["physical_abs"][j, i]
                                - abs(self.full_errors[j])
                            ),
                        }
                    )
        write_csv(self.out / "full_scan_gap.csv", gaps)
        write_csv(self.out / "full_scan_paired_gaps.csv", full_pairs)
        self.quality()
        self.mechanisms()
        self.texts(same, summary, domains)
        write_json(self.out / "analysis_validation.json", self.validation)
        return {
            "episode_count": 650,
            "event_count": len(events),
            "event_grid_points": len(self.event_grid),
            "quality_grid_targets": self.q_count,
            "anchor_targets": 21,
            "methods": 9,
            "full_reference": self.full_metrics,
        }

    def quality(self):
        maes = np.concatenate(
            [self.event[m]["mae"] for m in self.methods]
            + [np.array([self.full_metrics["mae"]])]
        )
        lo = math.floor(float(maes.min()))
        hi = math.ceil(float(maes.max()))
        targets = [
            {
                "target_id": f"Q_{q}",
                "target_type": "UNIFORM_1_MPA",
                "target_mae": float(q),
                "source_method": "ALL_OBSERVED_RANGE",
                "source_budget": None,
            }
            for q in range(lo, hi + 1)
        ]
        self.q_count = len(targets)
        anchors = [
            {
                "target_id": f"ANCHOR_{m}_{i}",
                "target_type": "NONADAPTIVE_ANCHOR",
                "target_mae": float(self.primary[m]["mae"][i]),
                "source_method": m,
                "source_budget": float(b),
            }
            for m in NONADAPTIVE
            for i, b in enumerate(BUDGETS)
            if b > 0
        ]
        anchors.append(
            {
                "target_id": "ANCHOR_FULL",
                "target_type": "FULL_REFERENCE_ANCHOR",
                "target_mae": self.full_metrics["mae"],
                "source_method": FULL,
                "source_budget": 1.0,
            }
        )
        write_csv(self.out / "quality_targets.csv", targets + anchors)
        for name, qs in [("grid", targets), ("anchors", anchors)]:
            results = []
            for target in qs:
                for gridname, grid, collection in [
                    ("PRIMARY_FIVE_BUDGETS", BUDGETS, self.primary),
                    ("SECONDARY_EVENT_GRID", self.event_grid, self.event),
                ]:
                    found = {
                        m: numeric.first_quality(
                            grid, collection[m]["mae"], target["target_mae"]
                        )
                        for m in self.methods
                    }
                    found[FULL] = numeric.first_quality(
                        [1.0], [self.full_metrics["mae"]], target["target_mae"]
                    )
                    for m in [*self.methods, FULL]:
                        r = found[m]
                        i = r["index"]
                        result = dict(
                            target,
                            grid=gridname,
                            method=m,
                            cost=r["cost"],
                            attained_mae=r["mae"],
                            status=r["status"],
                            later_recrosses_target=r["later_recrosses_target"],
                            main_cost=found[MAIN]["cost"],
                            **numeric.saving(found[MAIN]["cost"], r["cost"]),
                        )
                        for k in [
                            "actual_cost_mean",
                            "actual_cost_min",
                            "actual_cost_max",
                        ]:
                            result[k] = (
                                None
                                if i is None
                                else 1.0
                                if m == FULL
                                else float(collection[m][k][i])
                            )
                        result["scope"] = "POSTHOC_EMPIRICAL_NOT_DEPLOYABLE_STOP"
                        results.append(result)
            write_csv(self.out / f"equal_quality_{name}.csv", results)
            if name == "anchors":
                self.anchor_results = results
        testq = self.primary["GEOMETRY_SPREAD"]["mae"][-1]
        test = {
            m: numeric.first_quality(BUDGETS, self.primary[m]["mae"], testq)["cost"]
            for m in [MAIN, "GEOMETRY_SPREAD", "LEARNED_STATIC_TRUE"]
        }
        assert list(test.values()) == [0.0625, 0.125, 0.0625]
        self.validation["independent_quality_anchor"] = {
            "target": float(testq),
            "costs": test,
            "relative_saving_vs_geometry": 0.5,
            "relative_saving_vs_static": 0.0,
        }

    def mechanisms(self):
        summary = []
        divergence = []
        case_text = []
        for m in self.methods:
            rows = self.by_method[m]
            events = [e for e in self.events if e["method"] == m]
            first = [e for e in events if e["action_index"] == 1]
            restricted = sum(
                e["proposal_legal"] != e["environment_legal"] for e in first
            )
            summary.append(
                {
                    "method": m,
                    "episode_count": len(rows),
                    "first_step_restricted_count": restricted,
                    "c0_release_count": sum(
                        e["c0_reason"] == "C0_RELEASED_AFTER_FIRST_ACTION"
                        for e in events
                    ),
                    "observations": len(events),
                    "post_observation_error_reduction_mean_mpa": float(
                        np.mean(
                            [e["post_observation_error_reduction_mpa"] for e in events]
                        )
                    ),
                    "error_increases_count": sum(
                        e["post_observation_error_reduction_mpa"] < 0 for e in events
                    ),
                    "unique_action_sequences": len(
                        {tuple(r["cell_array"]) for r in rows}
                    ),
                    "scope": "EVENT_DESCRIPTIVE_NOT_CAUSAL_OR_INDEPENDENT_N",
                }
            )
        main = {r["specimen_key"]: r for r in self.by_method[MAIN]}
        for m in ["VLM_SPATIAL_OPEN_LOOP", "NO_VLM_SPATIAL_FEEDBACK"]:
            for r in self.by_method[m]:
                a = main[r["specimen_key"]]
                pairs = list(zip(a["cell_array"], r["cell_array"]))
                first = next((i + 1 for i, (x, y) in enumerate(pairs) if x != y), None)
                if first is None and len(a["cell_array"]) != len(r["cell_array"]):
                    first = len(pairs) + 1
                divergence.append(
                    {
                        "specimen_key": r["specimen_key"],
                        "comparator": m,
                        "first_divergence_action_index": first,
                        "control_first_in_main_proposal": a["trace"][0][
                            "proposal_legal"
                        ][r["cell_array"][0]]
                        == "1",
                        "main_first_proposal_restricted": a["trace"][0][
                            "proposal_legal"
                        ]
                        != a["trace"][0]["environment_legal"],
                        "same_cost_error_gain_00625": float(
                            abs(
                                numeric.held(
                                    r["cost_array"], r["pred_array"], [0.0625]
                                )[0]
                                - float(r["target_mpa"])
                            )
                            - abs(
                                numeric.held(
                                    a["cost_array"], a["pred_array"], [0.0625]
                                )[0]
                                - float(a["target_mpa"])
                            )
                        ),
                        "interpretation": "DESCRIPTIVE_ASSOCIATION_NOT_C0_CAUSATION",
                    }
                )
        for key in self.scope["cases"]:
            r = main[key]
            t = r["trace"][0]
            y = float(r["target_mpa"])
            changes = [
                abs(e["before_prediction_mpa"] - y) - abs(e["after_prediction_mpa"] - y)
                for e in r["trace"]
            ]
            worst = int(np.argmin(changes))
            second = r["trace"][1]
            case_text.append(
                f"## {key}\n\n冻结表面/VLM首步候选记录为 {t['c0_reason']}；调用 {t['actor_call_index']} 在无C-scan状态选 cell {t['cell']}，取得 {t['new_pixels']} 原生像素，成本 {t['after_cost']:.8f}。预测由 {t['before_prediction_mpa']:.6f} 变为 {t['after_prediction_mpa']:.6f} MPa；分析侧标签 {y:.6f}，该次误差减少 {changes[0]:.6f} MPa。随后调用 {second['actor_call_index']} 的规则为 {second['c0_reason']}，选择 cell {second['cell']}；这证明逐步记录，不证明模型理解了损伤边界。\n\n不利部分：第 {worst + 1} 次观测（cell {r['trace'][worst]['cell']}）误差减少 {changes[worst]:.6f} MPa；负值即误差增加。末预测 {r['pred_array'][-1]:.6f}，绝对误差 {abs(r['pred_array'][-1] - y):.6f} MPa。没有记录语言思维链或注意力；离开C0不称纠错，不能据关联断定C0造成误差。原七张图和所有同成本对照见 case_figure_reuse.csv、case_same_cost_states.csv。\n"
            )
        write_csv(self.out / "mechanism_event_summary.csv", summary)
        write_csv(self.out / "path_divergence.csv", divergence)
        write_csv(
            self.out / "case_figure_reuse.csv",
            [
                {
                    "specimen_key": key,
                    "source_path": str(p),
                    "source_sha256": self.hashes[str(p)],
                    "use": "UNCHANGED_FROZEN_PNG",
                }
                for key in self.scope["cases"]
                for p in self.case_paths
                if p.name.startswith(key.replace(":", "_"))
            ],
        )
        states = []
        for r in self.episodes:
            if r["specimen_key"] not in self.scope["cases"]:
                continue
            for b in BUDGETS:
                j = int(np.searchsorted(r["cost_array"], b, side="right") - 1)
                states.append(
                    {
                        "specimen_key": r["specimen_key"],
                        "method": r["method"],
                        "run": r["run"],
                        "budget": float(b),
                        "actual_cost": float(r["cost_array"][j]),
                        "prediction_mpa": float(r["pred_array"][j]),
                        "abs_error_mpa": float(
                            abs(r["pred_array"][j] - float(r["target_mpa"]))
                        ),
                        "visible_cells_derived": json.dumps(r["cell_array"][:j]),
                        "scope": "DERIVED_SAVED_ACTIONS_AT_COMMON_BUDGET",
                    }
                )
        write_csv(self.out / "case_same_cost_states.csv", states)
        (self.art / "case_narratives.md").write_text(
            "# 三个预先冻结VALID案例\n\n同k仅描述过程，数值效率对照用共同b。案例不按新结果筛选，保留q24-48。\n\n"
            + "".join(case_text)
        )

    def texts(self, same, paired, domains):
        table_fragments(
            self.out,
            "same_cost_metrics",
            same,
            [
                "method",
                "budget",
                "mae",
                "rmse",
                "r2",
                "actual_cost_mean",
                "actual_cost_min",
                "actual_cost_max",
            ],
        )
        table_fragments(
            self.out,
            "same_cost_paired_summary",
            paired,
            [
                "comparator",
                "budget",
                "mae_gain_mpa",
                "ci_low",
                "ci_high",
                "rmse_gain_mpa",
                "r2_main_minus_control",
            ],
        )
        table_fragments(
            self.out,
            "equal_quality_anchors",
            self.anchor_results,
            [
                "target_id",
                "target_mae",
                "grid",
                "method",
                "cost",
                "main_cost",
                "relative_saving",
                "status",
            ],
        )
        table_fragments(
            self.out,
            "same_cost_metrics_by_domain",
            domains,
            ["method", "dataset_id", "budget", "physical_n", "mae", "rmse", "r2"],
        )
        claim_specs = [
            ("E1", "有限投入下主方法相对固定路线的误差较低", "GEOMETRY_SPREAD", "area"),
            ("E2", "主方法比同VLM开环有较低面积误差", "VLM_SPATIAL_OPEN_LOOP", "area"),
            (
                "E3",
                "学习反馈策略比共享静态排序有较低面积误差",
                "LEARNED_STATIC_TRUE",
                "area",
            ),
            ("E4", "实际日志记录取得观测后再调用选择下一位置", None, None),
            (
                "E5",
                "本面板未显示VLM带来额外早期或全程收益",
                "NO_VLM_SPATIAL_FEEDBACK",
                "early_area",
            ),
            (
                "E6",
                "空间结构面积略低但终点误差高于均值结构",
                "VLM_MEAN_FEEDBACK",
                "area",
            ),
            ("E7", "主方法25%成本上限仍有完整输入质量差距", FULL, None),
        ]
        claims = []
        for eid, sentence, control, field in claim_specs:
            if field:
                gain = float(
                    numeric.weighted(
                        self.areas[control][field] - self.areas[MAIN][field],
                        np.ones((1, 50)),
                        self.domains,
                    )[0]
                )
            elif control == FULL:
                gain = float(self.primary[MAIN]["mae"][-1] - self.full_metrics["mae"])
            else:
                gain = None
            claims.append(
                {
                    "claim_id": eid,
                    "proposed_sentence": sentence,
                    "estimand": field
                    or (
                        "MAE_MAIN_025_MINUS_FULL_1"
                        if control == FULL
                        else "ACTUAL_EVENT_SEQUENCE"
                    ),
                    "estimate_mpa": gain,
                    "direction": "NOT_PERFORMANCE"
                    if gain is None
                    else "POSITIVE"
                    if gain > 0
                    else "NEGATIVE",
                    "support_level": "DESCRIPTIVE_FROZEN_VALID",
                    "source_file": "acquisition_events.csv"
                    if eid == "E4"
                    else "full_scan_gap.csv"
                    if eid == "E7"
                    else "mechanism_effects.csv",
                    "row_key": control or "method=VLM_SPATIAL_FEEDBACK",
                    "scope": "POSTHOC_FROZEN_VALID_SINGLE_SEED",
                    "limitations": "No causal/independent/industrial/STOP claim; selected VALID, single seed; retain adverse endpoint and domain effects",
                    "display": "F5/T2"
                    if field
                    else "F6/T1"
                    if control == FULL
                    else "case figures/process table",
                }
            )
        write_csv(self.out / "claim_evidence_matrix.csv", claims)
        table_fragments(
            self.art,
            "CLAIM_EVIDENCE_MATRIX",
            claims,
            [
                "claim_id",
                "proposed_sentence",
                "estimand",
                "estimate_mpa",
                "source_file",
                "row_key",
                "support_level",
                "scope",
                "limitations",
                "display",
            ],
        )


def analyze(root):
    started = time.perf_counter()
    evidence = Evidence(root)
    result = evidence.analyze()
    assert evidence.hashes == {str(p): sha(root / p) for p in evidence.inputs}, (
        "Frozen input changed during analysis"
    )
    write_json(
        root / RUN / "analysis_manifest.json",
        dict(
            task_id=TASK,
            source_sha=BASE,
            status="ANALYSIS_COMPLETE",
            **result,
            processing_seconds=time.perf_counter() - started,
            source_hashes_verified_after=True,
            new_training_updates=0,
            new_model_forwards=0,
            new_test_access=0,
            gpu_jobs=0,
            bootstrap={
                "replicates": 5000,
                "seed": 2026091401,
                "simultaneous": False,
                "inference_scope": SCOPE,
            },
            evidence_scope="POSTHOC_FROZEN_VALID_SINGLE_SEED",
            independent_confirmation="NOT_PERFORMED",
            noninferiority_status="NO_PRESPECIFIED_MARGIN",
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["analyze", "export"])
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = analyze(root) if args.command == "analyze" else export(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def export(root):
    """Render only derived tables; no reread of source episodes or bootstrap."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out = root / RUN
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    scope = json.loads(
        (root / "docs/cai/paper_evidence/PAPER_EVIDENCE_SCOPE.json").read_text()
    )
    methods = scope["methods"]
    names = {
        "CENTER_FIRST": "Center",
        "GEOMETRY_SPREAD": "Geometry",
        "SERPENTINE": "Serpentine",
        "RANDOM": "Random (5 repeats)",
        "LEARNED_STATIC_TRUE": "Learned static",
        "NO_VLM_SPATIAL_FEEDBACK": "No VLM, spatial feedback",
        "VLM_MEAN_FEEDBACK": "VLM, mean feedback",
        MAIN: "VLM, spatial feedback",
        "VLM_SPATIAL_OPEN_LOOP": "VLM, open loop",
    }
    colors = [
        "#8a8a8a",
        "#5f6b73",
        "#a5a0a0",
        "#a77e5f",
        "#786d9b",
        "#0072B2",
        "#009E73",
        "#D55E00",
        "#CC79A7",
    ]
    styles = ["--", "-.", ":", "--", "-.", "-", ":", "-", "--"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    curves = read_csv(out / "cost_error_event_curves.csv")
    full = json.loads((out / "full_scan_reference.json").read_text())
    outputs = []

    def new(ylabel, title):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.32, top=0.90)
        ax.set(
            xlabel="Native-raster acquisition fraction (budget cap)",
            ylabel=ylabel,
            title=title
            + "\nFrozen VALID · 50 specimens / 48 groups · single learning seed",
        )
        ax.set_xlim(0, 0.25)
        ax.set_xticks(BUDGETS)
        ax.grid(axis="y", alpha=0.18)
        return fig, ax

    def save(fig, ax, name, source, legend=True):
        if legend:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(figdir / f"{name}.{extension}", dpi=300)
        outputs.append(
            {
                "figure": name,
                "source": source,
                "formats": ["png", "svg", "pdf"],
                "width_inches": 7.2,
                "height_inches": 5.0,
                "alignment": "NOT_APPLICABLE_SINGLE_AXES",
                "scope": "POSTHOC_FROZEN_VALID_SINGLE_SEED",
            }
        )
        plt.close(fig)

    for number, metric, label in [
        (1, "mae", "MAE (MPa)"),
        (2, "rmse", "RMSE (MPa)"),
        (3, "r2", "R² (repeat mean)"),
    ]:
        fig, ax = new(label, "Same acquisition-cost cap")
        for m, c, s in zip(methods, colors, styles):
            rows = [r for r in curves if r["method"] == m]
            ax.step(
                [float(r["budget"]) for r in rows],
                [float(r[metric]) for r in rows],
                where="post",
                color=c,
                ls=s,
                lw=2 if m == MAIN else 1.2,
                label=names[m],
            )
        ax.axhline(
            full[metric],
            color="#222222",
            ls=":",
            lw=1,
            label="Full input @ 1.0 (reference only)",
        )
        save(
            fig,
            ax,
            f"F{number}_{metric}_cost",
            "cost_error_event_curves.csv; full_scan_reference.json",
        )
    fig, ax = new(
        "Earliest observed budget cap",
        "Equal empirical cohort MAE · primary five-budget grid",
    )
    ax.set_xlabel("Post-hoc empirical MAE target (MPa)")
    ax.set_ylim(-0.025, 1.05)
    qrows = read_csv(out / "equal_quality_grid.csv")
    qvals = sorted({float(r["target_mae"]) for r in qrows})
    ax.set_xlim(min(qvals), max(qvals))
    ax.set_xticks(
        np.arange(
            min(qvals), max(qvals) + 1, max(1, math.ceil((max(qvals) - min(qvals)) / 6))
        )
    )
    for m, c, s in zip([*methods, FULL], [*colors, "#222222"], [*styles, ":"]):
        rows = [
            r for r in qrows if r["method"] == m and r["grid"] == "PRIMARY_FIVE_BUDGETS"
        ]
        ax.step(
            [float(r["target_mae"]) for r in rows],
            [float(r["cost"]) if r["cost"] else np.nan for r in rows],
            where="post",
            color=c,
            ls=s,
            lw=2 if m == MAIN else 1.2,
            label=names.get(m, "Full-input process @ 1.0"),
        )
    save(
        fig,
        ax,
        "F4_equal_quality",
        "equal_quality_grid.csv (PRIMARY_FIVE_BUDGETS; missing = not reached)",
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.21, top=0.86)
    rows = read_csv(out / "mechanism_area_intervals.csv")
    labels = ["Fixed-route A", "Feedback A", "VLM early A", "Spatial A"]
    for i, r in enumerate(rows):
        v = float(r["gain_mpa"])
        lo = float(r["ci_low"])
        hi = float(r["ci_high"])
        ax.plot([lo, hi], [i, i], color="#444444", lw=1.5)
        ax.plot(v, i, "o", color="#0072B2" if v >= 0 else "#D55E00", ms=6)
    ax.axvline(0, color="#888888", ls="--", lw=0.8)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set(
        xlabel="Control − main, domain-equal area (MPa)",
        title="Mechanism contrasts: mixed directions\n95% pointwise exploratory group-bootstrap intervals",
    )
    fig.text(
        0.25,
        0.08,
        "Frozen selected VALID · 50 specimens / 48 groups · one seed\nConditional on selection; not causal or independent confirmation",
        fontsize=8,
    )
    save(fig, ax, "F5_mechanism_effects", "mechanism_area_intervals.csv", False)
    fig, ax = new(
        "Partial MAE − full-input MAE (MPa)",
        "Cross-cost information gap (full reference costs 1.0)",
    )
    for m, c, s in zip(methods, colors, styles):
        rows = [r for r in curves if r["method"] == m]
        ax.step(
            [float(r["budget"]) for r in rows],
            [float(r["mae"]) - full["mae"] for r in rows],
            where="post",
            color=c,
            ls=s,
            lw=2 if m == MAIN else 1.2,
            label=names[m],
        )
    ax.axhline(0, color="#222222", ls=":", lw=0.8)
    save(
        fig,
        ax,
        "F6_full_scan_gap",
        "cost_error_event_curves.csv; full_scan_reference.json",
    )
    write_json(
        out / "figure_manifest.json",
        {
            "figures": outputs,
            "count": 6,
            "old_case_figures": "case_figure_reuse.csv (21 unchanged source PNG)",
            "new_model_forwards": 0,
        },
    )
    return {"status": "SIX_SUMMARY_FIGURES_EXPORTED", "figures": 6}
