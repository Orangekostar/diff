"""CLI orchestration and hash-aware phase state for order analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from scripts.cai_order_mechanism.inputs import (
    TaskContext,
    atomic_json,
    canonical_json,
    prepare_inputs,
    sha256_bytes,
    sha256_file,
)

PHASES = ("prepare", "derive", "plan", "infer", "analyze", "report", "verify")
MUTABLE_PHASE_OUTPUTS = {
    "prepare": {"runtime_lock.json"},
    "infer": {"resource_usage.json", "runtime_lock.json"},
    "report": {"resource_usage.json"},
}


def ordered_phases(command: str) -> tuple[str, ...]:
    if command == "all":
        return PHASES
    if command not in PHASES:
        raise ValueError(f"unknown phase: {command}")
    return (command,)


def phase_signature(context: TaskContext, phase: str, inputs: Sequence[Path]) -> str:
    records = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in inputs
    ]
    return sha256_bytes(
        canonical_json(
            {"phase": phase, "scope_sha256": context.scope_sha256, "inputs": records}
        )
    )


class PhaseStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return {"phases": {}, "resume_attempts": 0}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        value.setdefault("phases", {})
        value.setdefault("resume_attempts", 0)
        return value

    def reusable(self, phase: str, signature: str) -> bool:
        state = self._read()
        row = state["phases"].get(phase, {})
        if row.get("status") != "COMPLETE" or row.get("signature") != signature:
            return False
        mutable = MUTABLE_PHASE_OUTPUTS.get(phase, set())
        outputs = [
            item
            for item in row.get("outputs", [])
            if Path(item["path"]).name not in mutable
        ]
        return bool(outputs) and all(
            Path(item["path"]).is_file()
            and sha256_file(item["path"]) == item["sha256"]
            for item in outputs
        )

    def complete(self, phase: str, signature: str, outputs: Sequence[str | Path]) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        mutable = MUTABLE_PHASE_OUTPUTS.get(phase, set())
        paths = [
            Path(path).resolve()
            for path in outputs
            if Path(path).name not in mutable
        ]
        if not paths or not all(path.is_file() for path in paths):
            raise ValueError("phase completion requires existing output files")
        state = self._read()
        state["phases"][phase] = {
            "status": "COMPLETE",
            "signature": signature,
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in paths
            ],
        }
        atomic_json(self.path, state)

    def claim_resume(self) -> int:
        state = self._read()
        attempts = int(state["resume_attempts"])
        if attempts >= 1:
            raise ValueError("resume attempt limit exceeded")
        state["resume_attempts"] = attempts + 1
        atomic_json(self.path, state)
        return attempts + 1


def _phase_inputs(context: TaskContext, phase: str) -> list[Path]:
    results = context.results_root
    mapping = {
        "prepare": [context.scope_path],
        "derive": [results / "input_bindings.json", results / "cohort_manifest.csv"],
        "plan": [results / "archived_cohort_episodes.csv.gz", results / "cohort_manifest.csv"],
        "infer": [results / "order_plan.json", results / "order_plan.sha256"],
        "analyze": [
            results / "archived_events.csv.gz",
            results / "reorder_trajectories.csv.gz",
        ],
        "report": [results / "stage_summary.csv", results / "order_summary.csv"],
        "verify": [results / "index.html", results / "resource_usage.json"],
    }
    paths = mapping[phase]
    if any(path is None or not Path(path).is_file() for path in paths):
        raise ValueError(f"phase prerequisites are incomplete: {phase}")
    return [Path(path) for path in paths]


def _phase_function(phase: str):
    if phase == "prepare":
        return prepare_inputs
    if phase == "derive":
        from scripts.cai_order_mechanism.analysis import derive_archived

        return derive_archived
    if phase == "plan":
        from scripts.cai_order_mechanism.orders import plan_orders

        return plan_orders
    if phase == "infer":
        from scripts.cai_order_mechanism.replay import infer_orders

        return infer_orders
    if phase == "analyze":
        from scripts.cai_order_mechanism.analysis import analyze_results

        return analyze_results
    if phase == "report":
        from scripts.cai_order_mechanism.reporting import render_report

        return render_report
    if phase == "verify":
        from scripts.cai_order_mechanism.verify import verify_release

        return verify_release
    raise ValueError(f"unknown phase: {phase}")


def _configure_infer_device() -> None:
    os.environ["CAI_ORDER_DEVICE"] = "cpu"
    try:
        inventory = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    occupied = {line.strip() for line in processes if line.strip()}
    for line in inventory:
        index, uuid, free_mib = (part.strip() for part in line.split(","))
        if uuid not in occupied and int(free_mib) >= 2048:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = index
            os.environ["CAI_ORDER_DEVICE"] = "cuda:0"
            os.environ["CAI_ORDER_PHYSICAL_GPU"] = index
            return


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(*PHASES, "all"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "4"
    context = TaskContext.load(arguments.root, arguments.scope)
    context.results_root.mkdir(parents=True, exist_ok=True)
    store = PhaseStore(context.results_root / "task_state.json")
    if arguments.resume:
        store.claim_resume()
    for phase in ordered_phases(arguments.command):
        inputs = _phase_inputs(context, phase)
        signature = phase_signature(context, phase, inputs)
        if store.reusable(phase, signature):
            continue
        if phase == "infer":
            _configure_infer_device()
        outputs = _phase_function(phase)(context)
        store.complete(phase, signature, outputs)
    return 0


__all__ = ["PHASES", "PhaseStore", "main", "ordered_phases", "phase_signature"]


if __name__ == "__main__":
    raise SystemExit(main())
