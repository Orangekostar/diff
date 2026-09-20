"""Exact C=P0+R1 rendering, bounded inference, and feature construction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import signal
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.cai_c_retrain.context import (
    TaskContext,
    atomic_json,
    canonical_json,
    sha256_file,
)

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
ALLOWED_SPLITS = frozenset({"TRAIN", "VALID"})
TERMINAL_STATUSES = frozenset(
    {
        "VALID_FIRST_PASS",
        "VALID_AFTER_REPAIR",
        "VALID_AFTER_INTERRUPTED_RETRY",
        "VALID_NO_RELIABLE_CUE",
        "SCHEMA_INVALID_AFTER_ONE_REPAIR",
    }
)
CONFIDENCE = {"unknown": 0.0, "low": 1.0 / 3.0, "medium": 2.0 / 3.0, "high": 1.0}
FORMAT_REPAIR_CONTEXT_PREFIX = "需要修复的原响应如下：\n<invalid_response>\n"
FORMAT_REPAIR_CONTEXT_SUFFIX = "\n</invalid_response>"


@dataclass(frozen=True, slots=True)
class RenderedInputs:
    clean: Image.Image
    numbered: Image.Image
    clean_sha256: str
    numbered_sha256: str
    render_config: dict[str, Any]
    label_boxes: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CaseInput:
    specimen_key: str
    dataset_id: str
    split: str
    capture_group_id: str
    impacted_surface_path: str
    source_sha256: str
    clean: Image.Image
    numbered: Image.Image
    clean_sha256: str
    numbered_sha256: str
    render_config: dict[str, Any]
    signature: str
    signature_data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SurfaceRegion:
    cells: tuple[int, ...]
    cue: str
    alternative: str
    confidence: str


@dataclass(frozen=True, slots=True)
class SurfacePercept:
    regions: tuple[SurfaceRegion, ...]
    no_reliable_cue: bool


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def image_sha256(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return _sha256_bytes(buffer.getvalue())


def _grid_only(clean: Image.Image) -> Image.Image:
    image = clean.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    line_width = max(1, round(max(width, height) / 512))
    for index in range(1, 8):
        x = round(index * width / 8)
        y = round(index * height / 8)
        draw.line((x, 0, x, height - 1), fill=(0, 255, 255, 110), width=line_width)
        draw.line((0, y, width - 1, y), fill=(0, 255, 255, 110), width=line_width)
    return image


def _render_readable(
    clean: Image.Image,
) -> tuple[Image.Image, tuple[dict[str, Any], ...], dict[str, Any]]:
    image = _grid_only(clean)
    width, height = image.size
    cell_side = min(
        min(
            round((index + 1) * width / 8) - round(index * width / 8)
            for index in range(8)
        ),
        min(
            round((index + 1) * height / 8) - round(index * height / 8)
            for index in range(8)
        ),
    )
    font_size = max(12, round(24 * cell_side / 128))
    inset = max(3, round(10 * cell_side / 128))
    probe = ImageDraw.Draw(image)
    boxes: list[dict[str, Any]] = []
    while font_size > 0:
        font = ImageFont.truetype(str(FONT), font_size)
        boxes = []
        for cell in range(64):
            x0 = round((cell % 8) * width / 8)
            y0 = round((cell // 8) * height / 8)
            x1 = round((cell % 8 + 1) * width / 8)
            y1 = round((cell // 8 + 1) * height / 8)
            bx0, by0, bx1, by1 = probe.textbbox((0, 0), str(cell), font=font)
            tx, ty = x0 + inset, y0 + inset
            box = [tx, ty, tx + bx1 - bx0, ty + by1 - by0]
            panel = [box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2]
            boxes.append(
                {
                    "cell_id": cell,
                    "cell_box": [x0, y0, x1, y1],
                    "text_box": box,
                    "panel_box": panel,
                    "draw_origin": [tx - bx0, ty - by0],
                }
            )
        if all(
            box["cell_box"][0] <= box["panel_box"][0]
            and box["cell_box"][1] <= box["panel_box"][1]
            and box["panel_box"][2] <= box["cell_box"][2]
            and box["panel_box"][3] <= box["cell_box"][3]
            for box in boxes
        ):
            break
        font_size -= 1
    if font_size == 0:
        raise ValueError("cannot fit uniformly inset labels")

    draw = ImageDraw.Draw(image, "RGBA")
    for box in boxes:
        x0, y0, x1, y1 = box["panel_box"]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(0, 0, 0, 170))
        draw.text(
            tuple(box["draw_origin"]),
            str(box["cell_id"]),
            font=font,
            fill=(255, 255, 255, 230),
        )
        cell_x, cell_y = box["cell_box"][:2]
        box["r0_label_box"] = list(
            probe.textbbox(
                (cell_x + 2, cell_y + 1), str(box["cell_id"]), stroke_width=1
            )
        )
    config = {
        "font_path": str(FONT),
        "font_name": list(font.getname()),
        "font_sha256": _sha256_bytes(FONT.read_bytes()),
        "font_size": font_size,
        "inset": inset,
        "panel_rgba": [0, 0, 0, 170],
        "text_rgba": [255, 255, 255, 230],
    }
    return image, tuple(boxes), config


def render_c_inputs(source: Image.Image, *, max_edge: int = 1024) -> RenderedInputs:
    if not isinstance(source, Image.Image) or source.width < 10 or source.height < 10:
        raise ValueError("surface render request is invalid")
    source.load()
    clean = source.convert("RGB").transpose(Image.Transpose.ROTATE_270)
    longest = max(clean.size)
    if longest > max_edge:
        scale = max_edge / longest
        size = tuple(max(10, math.floor(axis * scale + 0.5)) for axis in clean.size)
        clean = clean.resize(size, Image.Resampling.LANCZOS)
    numbered, boxes, config = _render_readable(clean)
    return RenderedInputs(
        clean=clean,
        numbered=numbered,
        clean_sha256=image_sha256(clean),
        numbered_sha256=image_sha256(numbered),
        render_config=config,
        label_boxes=boxes,
    )


def load_roster(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in ALLOWED_SPLITS]
    if len(rows) != 211 or sum(row["split"] == "TRAIN" for row in rows) != 161:
        raise ValueError(
            "authorized TRAIN/VALID roster does not contain 161/50 specimens"
        )
    if any(row["split"] not in ALLOWED_SPLITS for row in rows):
        raise ValueError("TEST access is forbidden")
    if len({row["specimen_key"] for row in rows}) != len(rows):
        raise ValueError("authorized roster contains duplicate specimens")
    return rows


def _parse_percept(text: str) -> SurfacePercept:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("surface percept schema is invalid")
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("surface percept schema is invalid")
        candidate = "\n".join(lines[1:-1]).strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("surface percept schema is invalid") from error
    if type(payload) is not dict or set(payload) != {"regions", "no_reliable_cue"}:
        raise ValueError("surface percept schema is invalid")
    if type(payload["regions"]) is not list or len(payload["regions"]) > 2:
        raise ValueError("surface percept schema is invalid")
    regions = []
    for raw in payload["regions"]:
        if (
            type(raw) is not dict
            or set(raw) != {"cells", "cue", "alternative", "confidence"}
            or type(raw["cells"]) is not list
            or not 1 <= len(raw["cells"]) <= 4
            or any(type(cell) is not int or not 0 <= cell < 64 for cell in raw["cells"])
            or len(set(raw["cells"])) != len(raw["cells"])
            or type(raw["cue"]) is not str
            or not raw["cue"].strip()
            or len(raw["cue"]) > 120
            or type(raw["alternative"]) is not str
            or not raw["alternative"].strip()
            or len(raw["alternative"]) > 200
            or raw["confidence"] not in CONFIDENCE
        ):
            raise ValueError("surface percept schema is invalid")
        regions.append(
            SurfaceRegion(
                cells=tuple(raw["cells"]),
                cue=raw["cue"],
                alternative=raw["alternative"],
                confidence=raw["confidence"],
            )
        )
    no_reliable = payload["no_reliable_cue"]
    if type(no_reliable) is not bool or (no_reliable and regions):
        raise ValueError("surface percept schema is invalid")
    return SurfacePercept(tuple(regions), no_reliable)


def parse_contract(text: str) -> dict[str, Any]:
    try:
        percept = _parse_percept(text)
    except (TypeError, ValueError) as error:
        return {
            "parser_valid": False,
            "contract_valid": False,
            "error": str(error),
            "percept": None,
        }
    cells = [cell for region in percept.regions for cell in region.cells]
    valid = len(cells) == len(set(cells)) and (
        bool(percept.regions) != percept.no_reliable_cue
    )
    return {
        "parser_valid": True,
        "contract_valid": valid,
        "error": None if valid else "DUPLICATE_CELLS_OR_EMPTY_NO_CUE_CONTRADICTION",
        "percept": json.loads(json.dumps(asdict(percept))),
    }


def repair_text(raw: str, repair_prompt: str) -> str:
    return (
        f"{repair_prompt}\n{FORMAT_REPAIR_CONTEXT_PREFIX}"
        f"{raw}{FORMAT_REPAIR_CONTEXT_SUFFIX}"
    )


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _finalize_valid(state: dict[str, Any], entry: dict[str, Any], kind: str) -> None:
    percept = entry["percept"]
    state["final_percept"] = percept
    state["available"] = True
    state["no_reliable_cue"] = percept["no_reliable_cue"]
    if percept["no_reliable_cue"]:
        state["status"] = "VALID_NO_RELIABLE_CUE"
    elif kind == "PRIMARY":
        state["status"] = "VALID_FIRST_PASS"
    elif kind == "FORMAT_REPAIR":
        state["status"] = "VALID_AFTER_REPAIR"
    else:
        state["status"] = "VALID_AFTER_INTERRUPTED_RETRY"


def execute_case(
    directory: str | Path,
    case: CaseInput,
    prompt: str,
    backend: Any,
    *,
    repair_prompt: str = "FORMAT_REPAIR",
) -> dict[str, Any]:
    """Execute at most two attempts and durably preserve each transition."""
    run_dir = Path(directory)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != case.signature:
            raise ValueError("frozen signature mismatch; no cache reuse")
        if state.get("status") in TERMINAL_STATUSES:
            return state
        attempts = state.get("attempts", [])
        if len(attempts) >= 2:
            return state
        if (
            state.get("status") == "STARTED"
            and attempts
            and attempts[-1]["status"] == "STARTED"
        ):
            attempts[-1].update(
                status="INTERRUPTED", error="process interrupted before result"
            )
            state["status"] = "INCOMPLETE"
            atomic_json(state_path, state)
    else:
        state = {
            "specimen_key": case.specimen_key,
            "signature": case.signature,
            "signature_data": case.signature_data,
            "status": "STARTED",
            "available": False,
            "no_reliable_cue": False,
            "attempts": [],
        }
        atomic_json(state_path, state)

    while len(state["attempts"]) < 2:
        number = len(state["attempts"]) + 1
        if number == 1:
            kind = "PRIMARY"
            actual_prompt = prompt
        elif state["attempts"][0]["status"] in {"FAILED", "INTERRUPTED"}:
            kind = "INTERRUPTED_RETRY"
            actual_prompt = prompt
        else:
            kind = "FORMAT_REPAIR"
            actual_prompt = repair_text(state["attempts"][0]["text"], repair_prompt)
        _write_text_atomic(run_dir / f"prompt_{number}.txt", actual_prompt)
        entry: dict[str, Any] = {
            "attempt_number": number,
            "kind": kind,
            "status": "STARTED",
            "started_unix": time.time(),
            "prompt_sha256": _sha256_bytes(actual_prompt.encode("utf-8")),
        }
        state["attempts"].append(entry)
        state["status"] = "STARTED"
        atomic_json(state_path, state)
        started = time.monotonic()
        try:
            result = backend.infer(
                (case.clean, case.numbered), actual_prompt, run_dir, number
            )
            if type(result) is not dict or type(result.get("text")) is not str:
                raise TypeError("backend result is invalid")
            entry.update(result)
            entry.update(parse_contract(result["text"]))
            entry["status"] = "COMPLETED"
            _write_text_atomic(run_dir / f"raw_{number}.txt", result["text"])
            atomic_json(
                run_dir / f"generated_token_ids_{number}.json",
                result.get("generated_token_ids", []),
            )
            _write_text_atomic(
                run_dir / f"chat_{number}.txt", str(result.get("chat_text", ""))
            )
            atomic_json(
                run_dir / f"input_{number}.json", result.get("input_metadata", {})
            )
        except Exception as error:  # noqa: BLE001 -- preserve every failed attempt
            entry.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            state["status"] = "INCOMPLETE"
        finally:
            entry["elapsed_seconds"] = time.monotonic() - started
            atomic_json(run_dir / f"attempt_{number}.json", entry)
            atomic_json(state_path, state)

        if entry["status"] == "FAILED":
            return state
        if entry["contract_valid"]:
            _finalize_valid(state, entry, kind)
            atomic_json(state_path, state)
            return state
        if number == 2:
            state.update(
                status="SCHEMA_INVALID_AFTER_ONE_REPAIR",
                available=False,
                no_reliable_cue=False,
                final_percept=None,
            )
            atomic_json(state_path, state)
            return state
    return state


def _vector_string(values: np.ndarray) -> str:
    return ";".join(str(float(value)) for value in values.astype(np.float32))


def build_feature_row(
    case: CaseInput, state: dict[str, Any], *, reused_from_pilot: bool
) -> dict[str, str]:
    indicator = np.zeros(64, dtype=np.float32)
    confidence = np.zeros(64, dtype=np.float32)
    if state.get("available") and state.get("final_percept"):
        for region in state["final_percept"]["regions"]:
            for cell in region["cells"]:
                indicator[cell] = np.float32(1.0)
                confidence[cell] = np.float32(CONFIDENCE[region["confidence"]])
    return {
        "specimen_key": case.specimen_key,
        "dataset_id": case.dataset_id,
        "split": case.split,
        "capture_group_id": case.capture_group_id,
        "vlm_available": str(bool(state.get("available"))),
        "no_reliable_cue": str(bool(state.get("no_reliable_cue"))),
        "region_indicator": _vector_string(indicator),
        "confidence": _vector_string(confidence),
        "status": str(state["status"]),
        "signature": case.signature,
        "cache_key": case.signature,
        "reused_from_pilot": str(reused_from_pilot),
    }


def validate_model_config(config: dict[str, Any]) -> None:
    expected = {
        "model_repository": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "dtype": "bfloat16",
        "attention": "sdpa",
        "processor": {
            "use_fast": False,
            "min_pixels": 200704,
            "max_pixels": 1003520,
        },
        "generation": {
            "max_new_tokens": 500,
            "do_sample": False,
            "temperature": None,
            "use_cache": True,
        },
        "batch_size": 1,
        "image_order": ["clean", "numbered"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen VLM model configuration changed")
    if _sha256_bytes(config.get("chat_template", "").encode("utf-8")) != config.get(
        "chat_template_sha256"
    ):
        raise ValueError("frozen VLM chat template identity changed")
    model_path = config.get("model_path")
    if (
        type(model_path) is not str
        or Path(model_path).name != expected["model_revision"]
        or not Path(model_path).is_dir()
    ):
        raise FileNotFoundError("frozen local Qwen model path is unavailable")


class QwenBackend:
    """Lazy one-GPU Qwen backend with exact recorded chat and token metadata."""

    def __init__(self, model_config: dict[str, Any], *, deadline: float) -> None:
        validate_model_config(model_config)
        self.model_config = model_config
        self.model_path = model_config["model_path"]
        self.deadline = deadline
        self.model = None
        self.processor = None
        self.forward_calls = 0

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible:
            raise RuntimeError("exactly one CUDA_VISIBLE_DEVICES entry is required")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
            min_pixels=200704,
            max_pixels=1003520,
        )
        if (
            _sha256_bytes(self.processor.chat_template.encode("utf-8"))
            != self.model_config["chat_template_sha256"]
        ):
            raise ValueError("loaded processor chat template changed")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda:0")
        self.model.eval().requires_grad_(False)
        self.model.register_forward_pre_hook(self._count_forward)

    def _count_forward(self, _module: Any, _args: Any) -> None:
        self.forward_calls += 1

    def infer(
        self,
        images: tuple[Image.Image, Image.Image],
        prompt: str,
        _directory: Path,
        _number: int,
    ) -> dict[str, Any]:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("backend is not loaded")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("cumulative VLM deadline reached")
        chat = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[chat], images=list(images), padding=True, return_tensors="pt"
        ).to("cuda:0")
        before = self.forward_calls
        kwargs = {
            "max_new_tokens": 500,
            "do_sample": False,
            "temperature": None,
            "use_cache": True,
        }

        def expired(_signum: int, _frame: Any) -> None:
            raise TimeoutError("generation exceeded the 120-second limit")

        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(
            signal.ITIMER_REAL, max(0.001, min(120.0, self.deadline - time.monotonic()))
        )
        try:
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **kwargs)
            torch.cuda.synchronize()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
        answer = self.processor.batch_decode(
            [ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        metadata = {
            "chat_sha256": _sha256_bytes(chat.encode("utf-8")),
            "image_sha256": [image_sha256(image) for image in images],
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "input_tokens": int(inputs["attention_mask"].sum()),
            "input_token_ids": inputs["input_ids"][0].tolist(),
            "generation_kwargs": kwargs,
        }
        return {
            "text": answer,
            "generated_token_ids": ids,
            "input_tokens": metadata["input_tokens"],
            "output_tokens": len(ids),
            "forward_calls": self.forward_calls - before,
            "chat_text": chat,
            "input_metadata": metadata,
        }


def _slug(specimen_key: str) -> str:
    return specimen_key.replace(":", "__")


def _signature(value: dict[str, Any]) -> str:
    return _sha256_bytes(canonical_json(value))


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _case_from_row(
    row: dict[str, str],
    source_root: Path,
    model_config: dict[str, Any],
    prompt_sha256: str,
) -> CaseInput:
    source_path = source_root / row["impacted_surface_path"]
    if sha256_file(source_path) != row["surface_sha256"]:
        raise ValueError(f"surface source hash changed: {row['specimen_key']}")
    with Image.open(source_path) as source:
        rendered = render_c_inputs(source)
    signature_data = {
        **model_config,
        "prompt_sha256": prompt_sha256,
        "clean_sha256": rendered.clean_sha256,
        "numbered_sha256": rendered.numbered_sha256,
        "render_id": "R1",
        "render_config": rendered.render_config,
    }
    return CaseInput(
        specimen_key=row["specimen_key"],
        dataset_id=row["dataset_id"],
        split=row["split"],
        capture_group_id=row["capture_group_id"],
        impacted_surface_path=row["impacted_surface_path"],
        source_sha256=row["surface_sha256"],
        clean=rendered.clean,
        numbered=rendered.numbered,
        clean_sha256=rendered.clean_sha256,
        numbered_sha256=rendered.numbered_sha256,
        render_config=rendered.render_config,
        signature=_signature(signature_data),
        signature_data=signature_data,
    )


def _save_input_record(vlm_root: Path, case: CaseInput) -> None:
    directory = vlm_root / "inputs" / _slug(case.specimen_key)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "specimen_key": case.specimen_key,
        "dataset_id": case.dataset_id,
        "split": case.split,
        "source_path": case.impacted_surface_path,
        "source_sha256": case.source_sha256,
        "clean_sha256": case.clean_sha256,
        "numbered_sha256": case.numbered_sha256,
        "render_config": case.render_config,
        "signature": case.signature,
        "full_inputs_reconstructed_by": "export-inputs",
    }
    atomic_json(directory / "input_identity.json", metadata)
    thumbnail = case.numbered.copy()
    thumbnail.thumbnail((320, 320), Image.Resampling.LANCZOS)
    thumbnail.save(
        directory / "numbered_thumbnail.png",
        format="PNG",
        optimize=False,
        compress_level=9,
    )


def _pilot_jobs(context: TaskContext) -> dict[str, dict[str, Any]]:
    lock = json.loads(
        (context.path("grounding_pilot") / "experiment_lock.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        job["specimen_key"]: job for job in lock["jobs"] if job["variant"] == "C_P0_R1"
    }


def _reuse_pilot_case(
    context: TaskContext,
    case: CaseInput,
    destination: Path,
    pilot_job: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if pilot_job is None or pilot_job["signature"] != case.signature:
        return None
    if pilot_job["signature_data"] != case.signature_data:
        raise ValueError(f"pilot signature collision for {case.specimen_key}")
    source = (
        context.path("grounding_pilot") / "runs" / _slug(case.specimen_key) / "C_P0_R1"
    )
    old_state = json.loads((source / "state.json").read_text(encoding="utf-8"))
    if old_state["signature"] != case.signature or not old_state["status"].startswith(
        "VALID"
    ):
        raise ValueError(f"pilot state is not reusable: {case.specimen_key}")
    if not destination.exists():
        shutil.copytree(source, destination)
    state = json.loads((destination / "state.json").read_text(encoding="utf-8"))
    if state.get("signature") != case.signature:
        raise ValueError(
            f"existing reused state signature changed: {case.specimen_key}"
        )
    percept = state.get("final_percept")
    state.update(
        signature_data=case.signature_data,
        available=True,
        no_reliable_cue=bool(percept and percept["no_reliable_cue"]),
        reused_from_pilot=True,
        reuse_source=str(source.relative_to(context.root)),
    )
    atomic_json(destination / "state.json", state)
    return state


def _load_resource_usage(context: TaskContext) -> tuple[Path, dict[str, Any]]:
    path = context.path("output") / "resource_usage.json"
    if not path.is_file():
        raise RuntimeError("prepare must complete before VLM execution")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write_vlm_contract(
    context: TaskContext,
    protocol: dict[str, Any],
    state_records: list[dict[str, Any]],
) -> dict[str, str]:
    vlm_root = context.path("vlm")
    render_configs = {
        canonical_json(record["case"].render_config) for record in state_records
    }
    if len(render_configs) != 1:
        raise ValueError("C input records do not share one R1 render configuration")
    input_rows = []
    for record in state_records:
        case = record["case"]
        state = record["state"]
        input_rows.append(
            {
                "specimen_key": case.specimen_key,
                "dataset_id": case.dataset_id,
                "split": case.split,
                "capture_group_id": case.capture_group_id,
                "source_path": case.impacted_surface_path,
                "source_sha256": case.source_sha256,
                "clean_sha256": case.clean_sha256,
                "numbered_sha256": case.numbered_sha256,
                "signature": case.signature,
                "status": state["status"],
                "attempts": len(state.get("attempts", [])),
                "reused_from_pilot": bool(record["reused_from_pilot"]),
            }
        )
    input_manifest = vlm_root / "input_manifest.csv"
    _atomic_csv(input_manifest, input_rows)
    config_lock = {
        "schema_version": 1,
        "task_id": context.task_id,
        "prior_version": context.scope["prior_version"],
        "prompt_sha256": protocol["prompt_sha256"],
        "repair_sha256": protocol["repair_sha256"],
        "model_config": protocol["model_config"],
        "render_id": "R1",
        "render_config": state_records[0]["case"].render_config,
        "image_order": ["clean", "numbered"],
        "cohort": {
            "rows": len(state_records),
            "train": sum(record["case"].split == "TRAIN" for record in state_records),
            "valid": sum(record["case"].split == "VALID" for record in state_records),
            "test": 0,
        },
        "maximum_generation_attempts_per_case": 2,
        "test_accessed": False,
    }
    config_path = vlm_root / "config_lock.json"
    atomic_json(config_path, config_lock)
    return {
        "input_manifest": input_manifest.name,
        "input_manifest_sha256": sha256_file(input_manifest),
        "config_lock": config_path.name,
        "config_lock_sha256": sha256_file(config_path),
    }


def _write_vlm_outputs(
    context: TaskContext,
    protocol: dict[str, Any],
    rows: list[dict[str, str]],
    state_records: list[dict[str, Any]],
) -> dict[str, Any]:
    vlm_root = context.path("vlm")
    contract = _write_vlm_contract(context, protocol, state_records)
    terminal = sum(
        record["state"]["status"] in TERMINAL_STATUSES for record in state_records
    )
    reused = sum(record["reused_from_pilot"] for record in state_records)
    new_records = [
        record for record in state_records if not record["reused_from_pilot"]
    ]
    attempts = sum(len(record["state"].get("attempts", [])) for record in new_records)
    primary_jobs = sum(
        any(
            attempt["kind"] == "PRIMARY"
            for attempt in record["state"].get("attempts", [])
        )
        for record in new_records
    )
    output_tokens = sum(
        int(attempt.get("output_tokens", 0))
        for record in new_records
        for attempt in record["state"].get("attempts", [])
    )
    forward_calls = sum(
        int(attempt.get("forward_calls", 0))
        for record in new_records
        for attempt in record["state"].get("attempts", [])
    )
    if primary_jobs > 211 or attempts > 422 or output_tokens > 211000:
        raise RuntimeError("VLM generation authorization exceeded")
    manifest = {
        "schema_version": 1,
        "task_id": context.task_id,
        "prior_version": context.scope["prior_version"],
        "execution_code_sha256": sha256_file(Path(__file__)),
        "status": "C_PRIOR_COMPLETE" if terminal == 211 else "INCOMPLETE",
        "rows": len(state_records),
        "terminal_rows": terminal,
        "train_rows": sum(row["split"] == "TRAIN" for row in rows),
        "valid_rows": sum(row["split"] == "VALID" for row in rows),
        "test_rows": 0,
        "reused_pilot_rows": reused,
        "new_unique_primary_jobs": primary_jobs,
        "new_generation_attempts": attempts,
        "new_output_tokens": output_tokens,
        "new_qwen_top_level_forward_calls": forward_calls,
        **contract,
        "feature_csv": "vlm_actor_features_fit.csv",
        "feature_csv_sha256": None,
        "states": [
            {
                "specimen_key": record["case"].specimen_key,
                "signature": record["case"].signature,
                "status": record["state"]["status"],
                "attempts": len(record["state"].get("attempts", [])),
                "reused_from_pilot": record["reused_from_pilot"],
            }
            for record in state_records
        ],
    }
    features_path = vlm_root / "vlm_actor_features_fit.csv"
    if len(rows) == 211:
        _atomic_csv(features_path, rows)
        manifest["feature_csv_sha256"] = sha256_file(features_path)
    atomic_json(vlm_root / "vlm_manifest_fit.json", manifest)
    return manifest


def _audit_first_new_case(
    context: TaskContext, state_records: list[dict[str, Any]], prompt_sha256: str
) -> dict[str, Any]:
    record = next(
        (
            row
            for row in state_records
            if not row["reused_from_pilot"] and row["state"].get("attempts")
        ),
        None,
    )
    if record is None:
        raise ValueError("no newly generated C case is available for wiring audit")
    case = record["case"]
    run_dir = context.path("vlm") / "runs" / _slug(case.specimen_key)
    attempt = record["state"]["attempts"][0]
    number = int(attempt["attempt_number"])
    input_path = run_dir / f"input_{number}.json"
    prompt_path = run_dir / f"prompt_{number}.txt"
    raw_path = run_dir / f"raw_{number}.txt"
    token_path = run_dir / f"generated_token_ids_{number}.json"
    if not all(
        path.is_file() for path in (input_path, prompt_path, raw_path, token_path)
    ):
        raise FileNotFoundError(
            "first new C case is missing a required inference record"
        )
    metadata = json.loads(input_path.read_text(encoding="utf-8"))
    expected_images = [case.clean_sha256, case.numbered_sha256]
    if (
        metadata.get("image_sha256") != expected_images
        or _sha256_bytes(prompt_path.read_bytes()) != prompt_sha256
    ):
        raise ValueError("first new C case did not receive the frozen P0/R1 inputs")
    audit = {
        "status": "PASS",
        "specimen_key": case.specimen_key,
        "split": case.split,
        "attempt_number": number,
        "attempt_kind": attempt["kind"],
        "prompt_sha256": prompt_sha256,
        "image_sha256": expected_images,
        "input_tokens": attempt.get("input_tokens"),
        "output_tokens": attempt.get("output_tokens"),
        "forward_calls": attempt.get("forward_calls"),
        "saved_records": [
            path.relative_to(context.root).as_posix()
            for path in (input_path, prompt_path, raw_path, token_path)
        ],
    }
    atomic_json(context.path("artifacts") / "FIRST_NEW_C_CASE_WIRING.json", audit)
    return audit


def vlm_stage(context: TaskContext, backend: Any | None = None) -> dict[str, Any]:
    protocol_path = context.path("output") / "protocol_lock.json"
    if not protocol_path.is_file():
        raise RuntimeError("prepare must complete before VLM execution")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["scope_sha256"] != context.scope_sha256:
        raise ValueError("protocol lock belongs to a different scope")
    prompt = (context.root / context.scope["vlm"]["prompt_source"]).read_text(
        encoding="utf-8"
    )
    if _sha256_bytes(prompt.encode("utf-8")) != protocol["prompt_sha256"]:
        raise ValueError("P0 prompt changed after preparation")
    repair_prompt = (
        context.path("grounding_pilot") / "prompts/FORMAT_REPAIR.txt"
    ).read_text(encoding="utf-8")
    roster = load_roster(context.path("data") / "candidate_queue.csv")
    feature_manifest = json.loads(
        (context.path("data") / "feature_bank_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_root = Path(feature_manifest["encoder_execution_root"])
    pilot_jobs = _pilot_jobs(context)
    vlm_root = context.path("vlm")
    (vlm_root / "runs").mkdir(parents=True, exist_ok=True)
    context.transition("vlm", "RUNNING", completed=0, total=211)

    created_backend = backend is None
    session_started = time.monotonic()
    resource_path, resource = _load_resource_usage(context)
    remaining_seconds = 5400.0 - float(resource.get("vlm_gpu_seconds", 0.0))
    if created_backend and remaining_seconds <= 0:
        raise RuntimeError("VLM GPU session budget is exhausted")
    model_config = dict(protocol["model_config"])
    validate_model_config(model_config)
    active_backend = backend
    state_records: list[dict[str, Any]] = []
    feature_rows: list[dict[str, str]] = []
    try:
        for index, row in enumerate(roster, start=1):
            case = _case_from_row(
                row,
                source_root,
                model_config,
                protocol["prompt_sha256"],
            )
            _save_input_record(vlm_root, case)
            run_dir = vlm_root / "runs" / _slug(case.specimen_key)
            state = None
            reused = False
            if not (run_dir / "state.json").is_file():
                state = _reuse_pilot_case(
                    context, case, run_dir, pilot_jobs.get(case.specimen_key)
                )
                reused = state is not None
            if state is None and (run_dir / "state.json").is_file():
                state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
                if state.get("signature") != case.signature:
                    raise ValueError(f"stored C signature changed: {case.specimen_key}")
                reused = bool(state.get("reused_from_pilot"))
            if state is None or state.get("status") not in TERMINAL_STATUSES:
                if active_backend is None:
                    active_backend = QwenBackend(
                        model_config,
                        deadline=session_started + remaining_seconds,
                    )
                    active_backend.load()
                state = execute_case(
                    run_dir,
                    case,
                    prompt,
                    active_backend,
                    repair_prompt=repair_prompt,
                )
            state_records.append(
                {"case": case, "state": state, "reused_from_pilot": reused}
            )
            feature_rows.append(
                build_feature_row(case, state, reused_from_pilot=reused)
            )
            if index == 1 or index % 10 == 0 or index == 211:
                context.transition(
                    "vlm",
                    "RUNNING",
                    completed=index,
                    total=211,
                    terminal=sum(
                        record["state"]["status"] in TERMINAL_STATUSES
                        for record in state_records
                    ),
                )
    finally:
        if created_backend and active_backend is not None:
            elapsed = time.monotonic() - session_started
            resource["vlm_gpu_seconds"] = (
                float(resource.get("vlm_gpu_seconds", 0.0)) + elapsed
            )
            resource["status"] = "VLM_SESSION_RECORDED"
            atomic_json(resource_path, resource)

    manifest = _write_vlm_outputs(context, protocol, feature_rows, state_records)
    if manifest["status"] == "C_PRIOR_COMPLETE":
        manifest["first_new_case_wiring"] = _audit_first_new_case(
            context, state_records, protocol["prompt_sha256"]
        )
        atomic_json(vlm_root / "vlm_manifest_fit.json", manifest)
    resource["new_vlm_primary_jobs"] = manifest["new_unique_primary_jobs"]
    resource["new_vlm_attempts"] = manifest["new_generation_attempts"]
    resource["new_vlm_output_tokens"] = manifest["new_output_tokens"]
    resource["status"] = manifest["status"]
    atomic_json(resource_path, resource)
    phase_status = "COMPLETE" if manifest["status"] == "C_PRIOR_COMPLETE" else "PARTIAL"
    context.transition(
        "vlm",
        phase_status,
        terminal_rows=manifest["terminal_rows"],
        reused_pilot_rows=manifest["reused_pilot_rows"],
        new_generation_attempts=manifest["new_generation_attempts"],
    )
    return manifest


def export_inputs(context: TaskContext, specimen_key: str) -> dict[str, Any]:
    roster = load_roster(context.path("data") / "candidate_queue.csv")
    try:
        row = next(row for row in roster if row["specimen_key"] == specimen_key)
    except StopIteration as error:
        raise ValueError(
            "specimen key is not in the authorized TRAIN/VALID roster"
        ) from error
    protocol = json.loads(
        (context.path("output") / "protocol_lock.json").read_text(encoding="utf-8")
    )
    feature_manifest = json.loads(
        (context.path("data") / "feature_bank_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    case = _case_from_row(
        row,
        Path(feature_manifest["encoder_execution_root"]),
        protocol["model_config"],
        protocol["prompt_sha256"],
    )
    destination = context.path("vlm") / "exported_inputs" / _slug(specimen_key)
    destination.mkdir(parents=True, exist_ok=True)
    case.clean.save(destination / "clean.png", optimize=False, compress_level=9)
    case.numbered.save(destination / "R1.png", optimize=False, compress_level=9)
    metadata = {
        "specimen_key": case.specimen_key,
        "clean_sha256": case.clean_sha256,
        "numbered_sha256": case.numbered_sha256,
        "render_config": case.render_config,
        "signature": case.signature,
        "model_calls": 0,
    }
    atomic_json(destination / "metadata.json", metadata)
    return metadata
