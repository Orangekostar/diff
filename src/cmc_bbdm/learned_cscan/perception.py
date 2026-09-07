"""Action-free frozen-VLM surface perception and response cache."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from cmc_bbdm.vlm_cscan.vlm import VLMRawResponse

SURFACE_PERCEPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["regions", "no_reliable_cue"],
    "properties": {
        "regions": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cells", "cue", "alternative", "confidence"],
                "properties": {
                    "cells": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {"type": "integer", "minimum": 0, "maximum": 63},
                    },
                    "cue": {"type": "string", "minLength": 1, "maxLength": 120},
                    "alternative": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "unknown"],
                    },
                },
            },
        },
        "no_reliable_cue": {"type": "boolean"},
    },
}

SURFACE_PERCEPT_PROMPT = """你是材料试样的表面观察模块。输入是同一张真实表面图和带有 8x8 编号网格的版本。
仅描述图像中可见的表面线索，不推断内部超声结果、损伤深度、材料铺层或任务完成情况。
每个候选区域给出 0 到 63 范围内的网格编号集合、可见线索类型、一个合理的非损伤替代解释和序数置信等级。
允许没有可靠线索；线索类型可以是 unknown 或 none。最多两个区域，每个区域最多四个网格，所有区域合计最多八个不重复网格。
不要推荐扫描、工具、策略或后续操作。只返回符合所附 JSON schema 的一个对象。
JSON schema:
""" + json.dumps(SURFACE_PERCEPT_SCHEMA, sort_keys=True, separators=(",", ":"))

FORMAT_REPAIR_PROMPT = """上一响应不符合所附 JSON schema。不要解释，也不要增加字段；只重新返回一个合法 JSON 对象。
所有 regions 的 cells 合计不得超过 8，cell 不得重复；如果原响应超过 8 个，删除最低置信区域末尾的多余 cell。
JSON schema:
""" + json.dumps(SURFACE_PERCEPT_SCHEMA, sort_keys=True, separators=(",", ":"))
FORMAT_REPAIR_CONTEXT_PREFIX = "需要修复的原响应如下：\n<invalid_response>\n"
FORMAT_REPAIR_CONTEXT_SUFFIX = "\n</invalid_response>"


@dataclass(frozen=True, slots=True)
class SurfaceRegion:
    cells: tuple[int, ...]
    cue: str
    alternative: str
    confidence: str

    def __post_init__(self) -> None:
        if (
            type(self.cells) is not tuple
            or not self.cells
            or len(self.cells) > 4
            or len(set(self.cells)) != len(self.cells)
            or any(type(cell) is not int or not 0 <= cell < 64 for cell in self.cells)
            or type(self.cue) is not str
            or not self.cue.strip()
            or len(self.cue) > 120
            or type(self.alternative) is not str
            or not self.alternative.strip()
            or len(self.alternative) > 200
            or self.confidence not in {"low", "medium", "high", "unknown"}
        ):
            raise ValueError("surface region does not match schema")


@dataclass(frozen=True, slots=True)
class SurfacePercept:
    regions: tuple[SurfaceRegion, ...]
    no_reliable_cue: bool

    def __post_init__(self) -> None:
        cells = tuple(cell for region in self.regions for cell in region.cells)
        if (
            type(self.regions) is not tuple
            or len(self.regions) > 2
            or any(type(region) is not SurfaceRegion for region in self.regions)
            or len(cells) > 8
            or type(self.no_reliable_cue) is not bool
            or (self.no_reliable_cue and bool(self.regions))
        ):
            raise ValueError("surface percept does not match schema")


@dataclass(frozen=True, slots=True)
class SurfacePerceptRequest:
    model_revision: str
    clean_image_sha256: str
    gridded_image_sha256: str
    render_version: str
    prompt_sha256: str
    schema_version: int

    def __post_init__(self) -> None:
        if (
            len(self.model_revision) not in {40, 64}
            or set(self.model_revision) - set("0123456789abcdef")
            or any(
                len(value) != 64 or set(value) - set("0123456789abcdef")
                for value in (
                    self.clean_image_sha256,
                    self.gridded_image_sha256,
                    self.prompt_sha256,
                )
            )
            or not self.render_version
            or type(self.schema_version) is not int
            or self.schema_version < 1
        ):
            raise ValueError("surface percept request is invalid")

    @property
    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class SurfacePerceptInference:
    percept: SurfacePercept
    cache_hit: bool
    cache_key: str
    raw_text: str
    actual_call_count: int
    original_call_count: int
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    repaired: bool


def parse_surface_percept(text: str) -> SurfacePercept:
    if type(text) is not str or not text.strip():
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
    raw_regions = payload["regions"]
    if type(raw_regions) is not list:
        raise ValueError("surface percept schema is invalid")
    regions: list[SurfaceRegion] = []
    for raw in raw_regions:
        if (
            type(raw) is not dict
            or set(raw) != {"cells", "cue", "alternative", "confidence"}
            or type(raw["cells"]) is not list
        ):
            raise ValueError("surface percept schema is invalid")
        try:
            regions.append(
                SurfaceRegion(
                    cells=tuple(raw["cells"]),
                    cue=raw["cue"],
                    alternative=raw["alternative"],
                    confidence=raw["confidence"],
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError("surface percept schema is invalid") from error
    try:
        return SurfacePercept(
            regions=tuple(regions), no_reliable_cue=payload["no_reliable_cue"]
        )
    except (TypeError, ValueError) as error:
        raise ValueError("surface percept schema is invalid") from error


def display_cell_id(physical_cell_id: int) -> int:
    if type(physical_cell_id) is not int or not 0 <= physical_cell_id < 64:
        raise ValueError("physical cell id is invalid")
    return (17 * physical_cell_id + 13) % 64


def physical_cell_id(display_id: int) -> int:
    if type(display_id) is not int or not 0 <= display_id < 64:
        raise ValueError("display cell id is invalid")
    return (49 * (display_id - 13)) % 64


def map_percept_to_physical(percept: SurfacePercept) -> SurfacePercept:
    if type(percept) is not SurfacePercept:
        raise TypeError("typed surface percept is required")
    return SurfacePercept(
        regions=tuple(
            SurfaceRegion(
                cells=tuple(physical_cell_id(cell) for cell in region.cells),
                cue=region.cue,
                alternative=region.alternative,
                confidence=region.confidence,
            )
            for region in percept.regions
        ),
        no_reliable_cue=percept.no_reliable_cue,
    )


class SurfacePerceptCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def resolve(
        self,
        request: SurfacePerceptRequest,
        infer: Callable[[str], VLMRawResponse],
    ) -> SurfacePerceptInference:
        if type(request) is not SurfacePerceptRequest or not callable(infer):
            raise TypeError("surface percept cache request is invalid")
        cached = self.get(request)
        if cached is not None:
            return cached
        first = infer(SURFACE_PERCEPT_PROMPT)
        if type(first) is not VLMRawResponse:
            raise TypeError("VLM backend returned an invalid response")
        calls = [first]
        try:
            percept = parse_surface_percept(first.text)
        except ValueError:
            repair_prompt = (
                f"{FORMAT_REPAIR_PROMPT}\n{FORMAT_REPAIR_CONTEXT_PREFIX}"
                f"{first.text}{FORMAT_REPAIR_CONTEXT_SUFFIX}"
            )
            repaired = infer(repair_prompt)
            if type(repaired) is not VLMRawResponse:
                raise TypeError("VLM backend returned an invalid response")
            calls.append(repaired)
            try:
                percept = parse_surface_percept(repaired.text)
            except ValueError as repair_error:
                raise ValueError(
                    "surface percept remained invalid after one repair"
                ) from repair_error
        row = {
            "schema_version": 2,
            "record_type": "surface_percept",
            "cache_key": request.cache_key,
            "request": asdict(request),
            "percept": _percept_payload(percept),
            "raw_text": calls[-1].text,
            "latency_seconds": sum(call.latency_seconds for call in calls),
            "input_tokens": sum(call.input_tokens for call in calls),
            "output_tokens": sum(call.output_tokens for call in calls),
            "call_count": len(calls),
            "repaired": len(calls) == 2,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        return self._from_row(row, cache_hit=False)

    def get(
        self, request: SurfacePerceptRequest
    ) -> SurfacePerceptInference | None:
        if type(request) is not SurfacePerceptRequest:
            raise TypeError("surface percept cache request is invalid")
        row = self._load().get(request.cache_key)
        return None if row is None else self._from_row(row, cache_hit=True)

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
                    raise ValueError("surface percept cache is corrupt") from error
                if (
                    row.get("record_type") != "surface_percept"
                    or type(key) is not str
                    or key in rows
                ):
                    raise ValueError("surface percept cache contains invalid identities")
                rows[key] = row
        return rows

    @staticmethod
    def _from_row(
        row: dict[str, object], *, cache_hit: bool
    ) -> SurfacePerceptInference:
        try:
            percept = _percept_from_payload(row["percept"])
            latency = float(row["latency_seconds"])
            input_tokens = int(row["input_tokens"])
            output_tokens = int(row["output_tokens"])
            call_count = int(row["call_count"])
            if (
                not math.isfinite(latency)
                or latency < 0.0
                or min(input_tokens, output_tokens, call_count) < 0
            ):
                raise ValueError
            return SurfacePerceptInference(
                percept=percept,
                cache_hit=cache_hit,
                cache_key=str(row["cache_key"]),
                raw_text=str(row["raw_text"]),
                actual_call_count=0 if cache_hit else call_count,
                original_call_count=call_count,
                latency_seconds=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                repaired=bool(row["repaired"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("surface percept cache row is invalid") from error


def _percept_payload(percept: SurfacePercept) -> dict[str, object]:
    return {
        "regions": [asdict(region) for region in percept.regions],
        "no_reliable_cue": percept.no_reliable_cue,
    }


def _percept_from_payload(payload: object) -> SurfacePercept:
    return parse_surface_percept(json.dumps(payload, separators=(",", ":")))


__all__ = [
    "FORMAT_REPAIR_PROMPT",
    "SURFACE_PERCEPT_PROMPT",
    "SURFACE_PERCEPT_SCHEMA",
    "SurfacePercept",
    "SurfacePerceptCache",
    "SurfacePerceptInference",
    "SurfacePerceptRequest",
    "SurfaceRegion",
    "display_cell_id",
    "map_percept_to_physical",
    "parse_surface_percept",
    "physical_cell_id",
]
