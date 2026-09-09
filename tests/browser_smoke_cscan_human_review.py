#!/usr/bin/env python3
"""Targeted Chromium smoke for the local C-scan review delivery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Page, sync_playwright

import cmc_bbdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = str(PROJECT_ROOT / "src/cmc_bbdm")
if LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(LOCAL_PACKAGE)

from cmc_bbdm.vlm_cscan.references import reference_from_payload


def _load_packet(page: Page, path: Path, expected_mode: str) -> None:
    page.set_input_files("#packet-file", str(path))
    page.wait_for_function(
        "expected => document.querySelector('#mode-name').textContent.includes(expected)",
        arg=expected_mode,
    )


def _assert_layout(page: Page) -> None:
    boxes = [
        page.locator(selector).bounding_box()
        for selector in (".item-panel", ".viewer-panel", ".review-panel")
    ]
    if any(box is None for box in boxes):
        raise AssertionError("a primary work panel is not visible")
    left, center, right = boxes
    assert left["x"] + left["width"] <= center["x"] + 1
    assert center["x"] + center["width"] <= right["x"] + 1
    assert right["x"] + right["width"] <= 1440


def _assert_canvas_framing(page: Page) -> None:
    viewer = page.locator(".viewer-panel").bounding_box()
    canvas = page.locator("#review-svg").bounding_box()
    assert viewer is not None and canvas is not None
    if canvas["height"] < viewer["height"] * 0.55:
        raise AssertionError("review canvas is vertically collapsed")


def _assert_nonblank(path: Path) -> None:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    if pixels.std() < 8 or np.unique(pixels.reshape(-1, 3), axis=0).shape[0] < 100:
        raise AssertionError(f"browser screenshot is blank: {path}")


def _reference_flow(
    browser: object,
    *,
    html: Path,
    packet_path: Path,
    output_root: Path,
    errors: list[str],
) -> dict[str, object]:
    context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 900})
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "console",
        lambda message: errors.append(message.text) if message.type == "error" else None,
    )
    page.goto(html.as_uri())
    _load_packet(page, packet_path, "参考区域标注")
    _assert_layout(page)
    _assert_canvas_framing(page)
    page.click("#zoom-in")
    page.click('[data-tool="pan"]')
    svg = page.locator("#review-svg")
    box = svg.bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] * 0.52, box["y"] + box["height"] * 0.52)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.56, box["y"] + box["height"] * 0.56)
    page.mouse.up()
    page.click('[data-tool="certain"]')
    svg.click(position={"x": box["width"] * 0.35, "y": box["height"] * 0.36})
    svg.click(position={"x": box["width"] * 0.67, "y": box["height"] * 0.38})
    svg.dblclick(position={"x": box["width"] * 0.55, "y": box["height"] * 0.70})
    page.wait_for_selector("polygon.region.certain")
    page.click('[data-tool="select"]')
    page.locator("polygon.region.certain").click()
    handle = page.locator("circle.vertex-handle").first
    handle_box = handle.bounding_box()
    assert handle_box is not None
    page.mouse.move(handle_box["x"] + handle_box["width"] / 2, handle_box["y"] + handle_box["height"] / 2)
    page.mouse.down()
    page.mouse.move(handle_box["x"] + 12, handle_box["y"] + 8)
    page.mouse.up()
    page.fill("#reviewer-alias", "TEST_ONLY_BROWSER")
    page.select_option("#reference-type", "EXPERT_REVIEWED")
    page.fill("#reference-notes", "浏览器闭环，含逗号\n第二行")
    page.click("#confirm-reference")
    assert "已确认" in page.locator("#item-state").inner_text()
    with page.expect_download() as download_info:
        page.click("#export-session")
    session_path = output_root / "reference_session_TEST_ONLY.json"
    download_info.value.save_as(session_path)
    session = json.loads(session_path.read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    item = packet["items"][0]
    item_state = session["items"][item["item_id"]]
    payload = {
        "specimen_key": item["specimen_key"],
        "source_image_sha256": item["source_image_sha256"],
        "reference_type": session["reviewer"]["reference_type"],
        "review_state": "reviewed",
        "reviewer_alias": session["reviewer"]["reviewer_alias"],
        "frame": "registered_cscan",
        "regions": item_state["regions"],
        "uncertain_regions": item_state["uncertain_regions"],
    }
    reference = reference_from_payload(
        payload, native_shape=(item["native_height"], item["native_width"])
    )
    assert reference.formal_eligible
    assert np.any(reference.certain_mask)

    restored = context.new_page()
    restored.on("pageerror", lambda error: errors.append(str(error)))
    restored.goto(html.as_uri())
    _load_packet(restored, packet_path, "参考区域标注")
    restored.set_input_files("#session-file", str(session_path))
    restored.wait_for_function(
        "() => document.querySelector('#item-state').textContent.includes('已确认')"
    )
    assert restored.input_value("#reference-notes") == "浏览器闭环，含逗号\n第二行"
    assert restored.locator("polygon.region.certain").count() == 1
    screenshot = output_root / "reference_workflow.png"
    restored.screenshot(path=str(screenshot), full_page=True)
    _assert_nonblank(screenshot)
    context.close()
    return {
        "session_path": str(session_path),
        "screenshot_path": str(screenshot),
        "polygon_count": 1,
        "restored": True,
        "coordinate_parser_match": True,
    }


def _blind_flow(
    browser: object,
    *,
    html: Path,
    packet_path: Path,
    output_root: Path,
    errors: list[str],
) -> dict[str, object]:
    context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 900})
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "console",
        lambda message: errors.append(message.text) if message.type == "error" else None,
    )
    external_requests: list[str] = []
    page.on(
        "request",
        lambda request: external_requests.append(request.url)
        if request.url.startswith(("http://", "https://"))
        else None,
    )
    page.goto(html.as_uri())
    _load_packet(page, packet_path, "首次 STOP 报告盲评")
    _assert_layout(page)
    _assert_canvas_framing(page)
    page.fill("#blind-reviewer", "TEST_ONLY_BROWSER")
    page.click('[data-decision="UNABLE_TO_JUDGE"]')
    page.fill("#problem-type", "边界,模糊")
    page.fill("#review-basis", "仅依据首次停止时已测证据")
    page.fill("#blind-notes", "浏览器盲评闭环\n不导入研究结果")
    page.click("#confirm-blind")
    assert "已确认" in page.locator("#item-state").inner_text()
    with page.expect_download() as download_info:
        page.click("#export-session")
    session_path = output_root / "blind_session_TEST_ONLY.json"
    download_info.value.save_as(session_path)
    session = json.loads(session_path.read_text(encoding="utf-8"))
    confirmed = [row for row in session["items"].values() if row["state"] == "CONFIRMED"]
    assert len(confirmed) == 1
    assert confirmed[0]["decision"] == "UNABLE_TO_JUDGE"

    restored = context.new_page()
    restored.on("pageerror", lambda error: errors.append(str(error)))
    restored.goto(html.as_uri())
    _load_packet(restored, packet_path, "首次 STOP 报告盲评")
    restored.set_input_files("#session-file", str(session_path))
    restored.wait_for_function(
        "() => document.querySelector('[data-decision=UNABLE_TO_JUDGE]').classList.contains('active')"
    )
    screenshot = output_root / "blind_workflow.png"
    restored.screenshot(path=str(screenshot), full_page=True)
    _assert_nonblank(screenshot)
    context.close()
    return {
        "session_path": str(session_path),
        "screenshot_path": str(screenshot),
        "confirmed_test_only_decision_count": 1,
        "restored": True,
        "external_network_request_count": len(external_requests),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--reference-packet", type=Path, required=True)
    parser.add_argument("--blind-packet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    html = arguments.html.resolve(strict=True)
    reference_packet = arguments.reference_packet.resolve(strict=True)
    blind_packet = arguments.blind_packet.resolve(strict=True)
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        version = browser.version
        reference = _reference_flow(
            browser,
            html=html,
            packet_path=reference_packet,
            output_root=output,
            errors=errors,
        )
        blind = _blind_flow(
            browser,
            html=html,
            packet_path=blind_packet,
            output_root=output,
            errors=errors,
        )
        browser.close()
    if errors:
        raise AssertionError("browser errors: " + " | ".join(errors))
    summary = {
        "schema_version": 1,
        "browser_smoke_status": "PASSED",
        "browser": "Chromium",
        "browser_version": version,
        "viewport": {"width": 1440, "height": 900},
        "reference": reference,
        "blind": blind,
        "page_error_count": 0,
        "console_error_count": 0,
        "real_reference_return_count": 0,
        "real_blind_review_count": 0,
        "test_only_outputs_imported_into_research": False,
    }
    summary_path = output / "browser_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
