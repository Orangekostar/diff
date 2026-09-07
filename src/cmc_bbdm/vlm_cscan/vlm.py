"""Frozen VLM parsing, inference, and resumable response cache."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image

from .contracts import SurfacePlan, SurfaceRegion

INITIAL_SURFACE_PROMPT = """你是材料试样检测的表面观察模块。输入包含同一试样的清晰表面图和带编号的 8x8 网格图。
这些图片没有提供内部超声。只描述真实可见的线索；不要把反光、污渍、纹理差异自动判成损伤。
凹陷样、裂纹样都只是表面外观假设。允许没有可靠线索。
坐标使用网格 cell_id 0..63，不使用文件名和试样编号。
返回最多 3 个候选区域，总计最多 8 个唯一优先 cell。
同时返回可见线索、干扰解释、置信级别和一份粗扫优先顺序。
不要猜内部损伤范围、CAI、铺层、冲击能量或损伤深度。
只输出 JSON，schema 为 {"regions":[{"cells":[27],"visible_cue":"indentation_like","alternative":"illumination_possible","confidence":"low"}],"priority_cells":[27],"initial_skill":"SURVEY_ROI","no_reliable_surface_cue":false}。"""

FORMAT_REPAIR_PROMPT = """上一响应不是合法 schema。不要解释，只重新输出一个 JSON 对象；cell 只能是 0..63，最多 3 个 region、8 个唯一 priority cell。"""

REPLAN_FORMAT_REPAIR_PROMPT = """上一响应不是合法菜单选择。不要解释，只输出 JSON：{{"menu_id":"给定菜单中的一个 ID","reason_code":"简短可见证据理由"}}。合法 menu_id：{menu_ids}。"""


@dataclass(frozen=True, slots=True)
class VLMRawResponse:
    text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if (
            type(self.text) is not str
            or not self.text
            or not math.isfinite(float(self.latency_seconds))
            or self.latency_seconds < 0.0
            or type(self.input_tokens) is not int
            or self.input_tokens < 0
            or type(self.output_tokens) is not int
            or self.output_tokens < 0
        ):
            raise ValueError("VLM response is invalid")


@dataclass(frozen=True, slots=True)
class SurfacePlanRequest:
    model_repository: str
    model_revision: str
    prompt_sha256: str
    clean_image_sha256: str
    gridded_image_sha256: str
    preprocessing_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.model_repository
            or len(self.model_revision) not in {40, 64}
            or set(self.model_revision) - set("0123456789abcdef")
            or any(
                len(value) != 64 or set(value) - set("0123456789abcdef")
                for value in (
                    self.prompt_sha256,
                    self.clean_image_sha256,
                    self.gridded_image_sha256,
                    self.preprocessing_sha256,
                )
            )
        ):
            raise ValueError("surface-plan request identity is invalid")

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class SurfacePlanInference:
    plan: SurfacePlan
    cache_hit: bool
    original_latency_seconds: float
    original_input_tokens: int
    original_output_tokens: int
    original_call_count: int
    actual_call_count: int
    cache_key: str
    raw_text: str
    parse_status: str
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class ReplanChoice:
    menu_id: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class ReplanRequest:
    model_repository: str
    model_revision: str
    prompt_sha256: str
    clean_image_sha256: str
    evidence_image_sha256: str
    evidence_state_sha256: str
    menu_sha256: str
    task: str
    event_code: str

    def __post_init__(self) -> None:
        if (
            not self.model_repository
            or len(self.model_revision) not in {40, 64}
            or set(self.model_revision) - set("0123456789abcdef")
            or any(
                len(value) != 64 or set(value) - set("0123456789abcdef")
                for value in (
                    self.prompt_sha256,
                    self.clean_image_sha256,
                    self.evidence_image_sha256,
                    self.evidence_state_sha256,
                    self.menu_sha256,
                )
            )
            or self.task not in {"LOCATE", "CHARACTERIZE"}
            or not self.event_code
        ):
            raise ValueError("replan request identity is invalid")

    @property
    def cache_key(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplanInference:
    choice: ReplanChoice
    cache_hit: bool
    original_latency_seconds: float
    original_input_tokens: int
    original_output_tokens: int
    original_call_count: int
    actual_call_count: int
    cache_key: str
    raw_text: str
    parse_status: str
    fallback_reason: str | None


class SurfacePlanCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def resolve(
        self,
        request: SurfacePlanRequest,
        infer: Callable[[str], VLMRawResponse],
    ) -> SurfacePlanInference:
        if type(request) is not SurfacePlanRequest or not callable(infer):
            raise TypeError("surface-plan cache request is invalid")
        cached_result = self.get(request)
        if cached_result is not None:
            return cached_result
        calls: list[VLMRawResponse] = []
        plan: SurfacePlan | None = None
        last_error: ValueError | None = None
        for prompt in (INITIAL_SURFACE_PROMPT, FORMAT_REPAIR_PROMPT):
            response = infer(prompt)
            if type(response) is not VLMRawResponse:
                raise TypeError("VLM backend returned an invalid response")
            calls.append(response)
            try:
                plan = parse_surface_plan(response.text)
                break
            except ValueError as error:
                last_error = error
        if plan is None:
            plan = SurfacePlan((), (), "BROADEN_SEARCH", True)
            parse_status = "FALLBACK"
            fallback_reason = f"FORMAT_INVALID:{type(last_error).__name__}"
        else:
            parse_status = "PARSED"
            fallback_reason = None
        row = {
            "schema_version": 1,
            "record_type": "initial_surface_plan",
            "cache_key": request.cache_key,
            "request": asdict(request),
            "plan": _plan_payload(plan),
            "raw_text": calls[-1].text,
            "parse_status": parse_status,
            "fallback_reason": fallback_reason,
            "latency_seconds": sum(item.latency_seconds for item in calls),
            "input_tokens": sum(item.input_tokens for item in calls),
            "output_tokens": sum(item.output_tokens for item in calls),
            "call_count": len(calls),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        return SurfacePlanInference(
            plan=plan,
            cache_hit=False,
            original_latency_seconds=float(row["latency_seconds"]),
            original_input_tokens=int(row["input_tokens"]),
            original_output_tokens=int(row["output_tokens"]),
            original_call_count=int(row["call_count"]),
            actual_call_count=len(calls),
            cache_key=request.cache_key,
            raw_text=calls[-1].text,
            parse_status=parse_status,
            fallback_reason=fallback_reason,
        )

    def get(self, request: SurfacePlanRequest) -> SurfacePlanInference | None:
        if type(request) is not SurfacePlanRequest:
            raise TypeError("surface-plan cache request is invalid")
        cached = self._load().get(request.cache_key)
        if cached is None:
            return None
        plan = _plan_from_payload(cached["plan"])
        return SurfacePlanInference(
            plan=plan,
            cache_hit=True,
            original_latency_seconds=float(cached["latency_seconds"]),
            original_input_tokens=int(cached["input_tokens"]),
            original_output_tokens=int(cached["output_tokens"]),
            original_call_count=int(cached["call_count"]),
            actual_call_count=0,
            cache_key=request.cache_key,
            raw_text=str(cached["raw_text"]),
            parse_status=str(cached["parse_status"]),
            fallback_reason=cached.get("fallback_reason"),
        )

    def _load(self) -> dict[str, dict[str, object]]:
        if not self.path.exists():
            return {}
        rows: dict[str, dict[str, object]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    key = row["cache_key"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError("surface-plan cache is corrupt") from error
                if row.get("record_type", "initial_surface_plan") != "initial_surface_plan":
                    continue
                if type(key) is not str or key in rows:
                    raise ValueError("surface-plan cache contains duplicate identities")
                rows[key] = row
        return rows


class ReplanChoiceCache:
    """Append-only tool-choice cache colocated with initial surface plans."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def resolve(
        self,
        request: ReplanRequest,
        *,
        prompt: str,
        legal_menu_ids: tuple[str, ...],
        infer: Callable[[str], VLMRawResponse],
    ) -> ReplanInference:
        self._validate(request, prompt, legal_menu_ids, infer)
        cached = self.get(request, legal_menu_ids=legal_menu_ids)
        if cached is not None:
            return cached
        calls: list[VLMRawResponse] = []
        choice: ReplanChoice | None = None
        last_error: ValueError | None = None
        prompts = (
            prompt,
            REPLAN_FORMAT_REPAIR_PROMPT.format(menu_ids=",".join(legal_menu_ids)),
        )
        for current_prompt in prompts:
            response = infer(current_prompt)
            if type(response) is not VLMRawResponse:
                raise TypeError("VLM backend returned an invalid response")
            calls.append(response)
            try:
                choice = parse_replan_choice(
                    response.text,
                    legal_menu_ids=legal_menu_ids,
                )
                break
            except ValueError as error:
                last_error = error
        if choice is None:
            choice = ReplanChoice(legal_menu_ids[0], "FORMAT_FALLBACK_FIRST_LEGAL")
            parse_status = "FALLBACK"
            fallback_reason = f"FORMAT_INVALID:{type(last_error).__name__}"
        else:
            parse_status = "PARSED"
            fallback_reason = None
        row = {
            "schema_version": 1,
            "record_type": "replan_choice",
            "cache_key": request.cache_key,
            "request": asdict(request),
            "choice": asdict(choice),
            "raw_text": calls[-1].text,
            "parse_status": parse_status,
            "fallback_reason": fallback_reason,
            "latency_seconds": sum(item.latency_seconds for item in calls),
            "input_tokens": sum(item.input_tokens for item in calls),
            "output_tokens": sum(item.output_tokens for item in calls),
            "call_count": len(calls),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        return self._from_row(row, cache_hit=False)

    def get(
        self,
        request: ReplanRequest,
        *,
        legal_menu_ids: tuple[str, ...],
    ) -> ReplanInference | None:
        if type(request) is not ReplanRequest:
            raise TypeError("replan cache request is invalid")
        _validate_menu_ids(legal_menu_ids)
        row = self._load().get(request.cache_key)
        if row is None:
            return None
        result = self._from_row(row, cache_hit=True)
        if result.choice.menu_id not in legal_menu_ids:
            raise ValueError("cached VLM choice is outside the current legal menu")
        return result

    def _validate(
        self,
        request: ReplanRequest,
        prompt: str,
        legal_menu_ids: tuple[str, ...],
        infer: Callable[[str], VLMRawResponse],
    ) -> None:
        if (
            type(request) is not ReplanRequest
            or type(prompt) is not str
            or not prompt
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            != request.prompt_sha256
            or not callable(infer)
        ):
            raise ValueError("replan cache request is invalid")
        _validate_menu_ids(legal_menu_ids)

    def _load(self) -> dict[str, dict[str, object]]:
        if not self.path.exists():
            return {}
        rows: dict[str, dict[str, object]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    if row.get("record_type") != "replan_choice":
                        continue
                    key = row["cache_key"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError("replan cache is corrupt") from error
                if type(key) is not str or key in rows:
                    raise ValueError("replan cache contains duplicate identities")
                rows[key] = row
        return rows

    @staticmethod
    def _from_row(row: dict[str, object], *, cache_hit: bool) -> ReplanInference:
        try:
            raw_choice = row["choice"]
            if type(raw_choice) is not dict:
                raise TypeError
            choice = ReplanChoice(
                menu_id=raw_choice["menu_id"],
                reason_code=raw_choice["reason_code"],
            )
            return ReplanInference(
                choice=choice,
                cache_hit=cache_hit,
                original_latency_seconds=float(row["latency_seconds"]),
                original_input_tokens=int(row["input_tokens"]),
                original_output_tokens=int(row["output_tokens"]),
                original_call_count=int(row["call_count"]),
                actual_call_count=0 if cache_hit else int(row["call_count"]),
                cache_key=str(row["cache_key"]),
                raw_text=str(row["raw_text"]),
                parse_status=str(row["parse_status"]),
                fallback_reason=row.get("fallback_reason"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("replan cache row is invalid") from error


class QwenVLBackend:
    """One lazily loaded local Qwen2.5-VL process on one explicit GPU."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str,
        dtype: str,
        max_new_tokens: int,
        min_visual_tokens: int = 256,
        max_visual_tokens: int = 1280,
    ) -> None:
        path = Path(model_path).resolve(strict=True)
        if (
            not path.is_dir()
            or not (path / "config.json").is_file()
            or device != "cuda:0"
            or dtype != "bfloat16"
            or type(max_new_tokens) is not int
            or not 1 <= max_new_tokens <= 700
            or type(min_visual_tokens) is not int
            or type(max_visual_tokens) is not int
            or not 4 <= min_visual_tokens <= max_visual_tokens <= 16384
        ):
            raise ValueError("Qwen backend configuration is invalid")
        self.model_path = path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.min_visual_tokens = min_visual_tokens
        self.max_visual_tokens = max_visual_tokens
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("configured CUDA device is unavailable")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        pixels_per_token = 28 * 28
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
            min_pixels=self.min_visual_tokens * pixels_per_token,
            max_pixels=self.max_visual_tokens * pixels_per_token,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device)
        self._model.eval()

    def infer(self, images: tuple[Image.Image, ...], prompt: str) -> VLMRawResponse:
        if (
            type(images) is not tuple
            or not images
            or any(not isinstance(image, Image.Image) for image in images)
            or type(prompt) is not str
            or not prompt
        ):
            raise ValueError("Qwen inference request is invalid")
        self.load()
        model = self._model
        processor = self._processor
        if model is None or processor is None:
            raise RuntimeError("Qwen backend did not load")
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _image in images),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=list(images),
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        input_tokens = int(inputs["attention_mask"].sum().item())
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                use_cache=True,
            )
        torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return VLMRawResponse(
            text=decoded,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=int(trimmed.shape[1]),
        )


def parse_surface_plan(text: str) -> SurfacePlan:
    try:
        payload = _json_object(text)
        return _plan_from_payload(payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("surface plan JSON is invalid") from error


def parse_replan_choice(
    text: str, *, legal_menu_ids: tuple[str, ...]
) -> ReplanChoice:
    _validate_menu_ids(legal_menu_ids)
    try:
        payload = _json_object(text)
        if not {"menu_id", "reason_code"} <= set(payload) <= {
            "menu_id",
            "reason_code",
            "confidence",
        }:
            raise ValueError
        menu_id = payload["menu_id"]
        reason_code = payload["reason_code"]
        if menu_id not in legal_menu_ids or type(reason_code) is not str or not reason_code:
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("VLM choice is not in the legal menu") from error
    return ReplanChoice(menu_id=menu_id, reason_code=reason_code)


def _validate_menu_ids(legal_menu_ids: tuple[str, ...]) -> None:
    if (
        type(legal_menu_ids) is not tuple
        or not legal_menu_ids
        or len(set(legal_menu_ids)) != len(legal_menu_ids)
        or any(type(value) is not str or not value for value in legal_menu_ids)
    ):
        raise ValueError("legal menu is invalid")


def _json_object(text: str) -> dict[str, object]:
    if type(text) is not str:
        raise TypeError
    stripped = text.strip()
    start = stripped.find("{")
    stop = stripped.rfind("}")
    if start < 0 or stop < start:
        raise ValueError
    payload = json.loads(stripped[start : stop + 1])
    if type(payload) is not dict:
        raise TypeError
    return payload


def _plan_from_payload(payload: object) -> SurfacePlan:
    if type(payload) is not dict or set(payload) != {
        "regions",
        "priority_cells",
        "initial_skill",
        "no_reliable_surface_cue",
    }:
        raise ValueError
    raw_regions = payload["regions"]
    raw_priority = payload["priority_cells"]
    if type(raw_regions) is not list or type(raw_priority) is not list:
        raise TypeError
    regions = []
    for raw in raw_regions:
        if type(raw) is not dict or set(raw) != {
            "cells",
            "visible_cue",
            "alternative",
            "confidence",
        }:
            raise ValueError
        if type(raw["cells"]) is not list:
            raise TypeError
        regions.append(
            SurfaceRegion(
                cells=tuple(raw["cells"]),
                visible_cue=raw["visible_cue"],
                alternative=raw["alternative"],
                confidence=raw["confidence"],
            )
        )
    return SurfacePlan(
        regions=tuple(regions),
        priority_cells=tuple(raw_priority),
        initial_skill=payload["initial_skill"],
        no_reliable_surface_cue=payload["no_reliable_surface_cue"],
    )


def _plan_payload(plan: SurfacePlan) -> dict[str, object]:
    return {
        "regions": [
            {
                "cells": list(region.cells),
                "visible_cue": region.visible_cue,
                "alternative": region.alternative,
                "confidence": region.confidence,
            }
            for region in plan.regions
        ],
        "priority_cells": list(plan.priority_cells),
        "initial_skill": plan.initial_skill,
        "no_reliable_surface_cue": plan.no_reliable_surface_cue,
    }


__all__ = [
    "FORMAT_REPAIR_PROMPT",
    "INITIAL_SURFACE_PROMPT",
    "REPLAN_FORMAT_REPAIR_PROMPT",
    "QwenVLBackend",
    "ReplanChoice",
    "ReplanChoiceCache",
    "ReplanInference",
    "ReplanRequest",
    "SurfacePlanCache",
    "SurfacePlanInference",
    "SurfacePlanRequest",
    "VLMRawResponse",
    "parse_replan_choice",
    "parse_surface_plan",
]
