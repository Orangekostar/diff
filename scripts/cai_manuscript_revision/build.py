"""Build editable English and bilingual manuscript documents.

The module intentionally only handles source parsing and document conversion.
It does not import the research package or alter any source material.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree


_BLOCK_RE = re.compile(
    r"(?m)^[ \t]*<!--[ \t]*block:[ \t]*(.*?)[ \t]*-->[ \t]*(?:\r?\n|$)"
)
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_PDF_IMAGE_RE = re.compile(
    r"(?P<open>!\[[^\]]*\]\(\s*)"
    r"(?P<path>figures/F[^)\s]*?\.png)"
    r"(?P<close>[^)]*\))"
)
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_W = "{" + _W_NS + "}"


class BuildError(RuntimeError):
    """Raised when an external document-build command fails."""


@dataclass(frozen=True)
class _Block:
    block_id: str
    content: str


@dataclass(frozen=True)
class _DocumentSources:
    manuscript_en: tuple[_Block, ...]
    manuscript_zh: tuple[_Block, ...]
    supplementary_en: tuple[_Block, ...]
    supplementary_zh: tuple[_Block, ...]
    shared_ids: Mapping[str, frozenset[str]]


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required input is missing: {path}") from exc


def _parse_blocks(path: Path) -> tuple[_Block, ...]:
    text = _read_utf8(path)
    matches = list(_BLOCK_RE.finditer(text))
    if not matches:
        raise ValueError(f"{path}: no block markers found")
    prefix = text[: matches[0].start()]
    if prefix.strip():
        raise ValueError(f"{path}: unmarked content before block ID {matches[0].group(1)!r}")

    blocks: list[_Block] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        block_id = match.group(1).strip()
        if not block_id:
            raise ValueError(f"{path}: empty block ID")
        if block_id in seen:
            raise ValueError(f"{path}: duplicate block ID {block_id!r}")
        seen.add(block_id)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end() : end].strip("\r\n")
        blocks.append(_Block(block_id, content))
    return tuple(blocks)


def _load_alignment(path: Path) -> dict[str, frozenset[str]]:
    try:
        raw = json.loads(_read_utf8(path))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level value must be an object")

    result: dict[str, frozenset[str]] = {}
    for kind in ("manuscript", "supplementary"):
        section = raw.get(kind)
        if not isinstance(section, dict):
            raise ValueError(f"{path}: missing object for {kind!r}")
        shared = section.get("shared_ids")
        if not isinstance(shared, list) or not all(isinstance(item, str) for item in shared):
            raise ValueError(f"{path}: {kind}.shared_ids must be a list of strings")
        normalized_shared = [item.strip() for item in shared]
        if any(not item for item in normalized_shared):
            raise ValueError(f"{path}: empty shared ID in {kind}")
        if len(normalized_shared) != len(set(normalized_shared)):
            duplicate = next(item for item in normalized_shared if normalized_shared.count(item) > 1)
            raise ValueError(f"{path}: duplicate shared ID {duplicate!r} in {kind}")
        result[kind] = frozenset(normalized_shared)
    return result


def _validate_pair(
    en_path: Path,
    zh_path: Path,
    en_blocks: Sequence[_Block],
    zh_blocks: Sequence[_Block],
    alignment_path: Path,
    shared_ids: frozenset[str],
) -> None:
    en_ids = {block.block_id for block in en_blocks}
    zh_ids = {block.block_id for block in zh_blocks}
    for block_id in sorted(en_ids - zh_ids):
        raise ValueError(f"{zh_path}: missing block ID {block_id!r} corresponding to {en_path}")
    for block_id in sorted(zh_ids - en_ids):
        raise ValueError(f"{en_path}: missing block ID {block_id!r} corresponding to {zh_path}")
    for block_id in sorted(shared_ids - en_ids):
        raise ValueError(f"{alignment_path}: shared ID {block_id!r} is missing from {en_path}")
    for block_id in sorted(shared_ids - zh_ids):
        raise ValueError(f"{alignment_path}: shared ID {block_id!r} is missing from {zh_path}")


def _load_sources(paper_root: Path) -> _DocumentSources:
    source_root = paper_root / "source"
    alignment_path = source_root / "alignment.json"
    paths = {
        "manuscript_en": source_root / "manuscript_en.md",
        "manuscript_zh": source_root / "manuscript_zh.md",
        "supplementary_en": source_root / "supplementary_en.md",
        "supplementary_zh": source_root / "supplementary_zh.md",
    }
    for path in (*paths.values(), alignment_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    alignment = _load_alignment(alignment_path)
    parsed = {name: _parse_blocks(path) for name, path in paths.items()}
    _validate_pair(
        paths["manuscript_en"],
        paths["manuscript_zh"],
        parsed["manuscript_en"],
        parsed["manuscript_zh"],
        alignment_path,
        alignment["manuscript"],
    )
    _validate_pair(
        paths["supplementary_en"],
        paths["supplementary_zh"],
        parsed["supplementary_en"],
        parsed["supplementary_zh"],
        alignment_path,
        alignment["supplementary"],
    )
    return _DocumentSources(
        manuscript_en=parsed["manuscript_en"],
        manuscript_zh=parsed["manuscript_zh"],
        supplementary_en=parsed["supplementary_en"],
        supplementary_zh=parsed["supplementary_zh"],
        shared_ids=alignment,
    )


def _serialize_blocks(blocks: Iterable[str]) -> str:
    parts = [content.strip("\r\n") for content in blocks]
    return "\n\n".join(parts).rstrip("\r\n") + "\n"


def _heading(text: str) -> tuple[str, str] | None:
    nonempty = [line for line in text.strip().splitlines() if line.strip()]
    if len(nonempty) != 1:
        return None
    match = _HEADING_RE.fullmatch(nonempty[0].strip())
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def _merge_blocks(en: _Block, zh: _Block) -> str:
    if en.content == zh.content:
        return en.content
    en_heading = _heading(en.content)
    zh_heading = _heading(zh.content)
    if en_heading and zh_heading:
        level, en_title = en_heading
        _, zh_title = zh_heading
        return f"{level} {en_title} / {zh_title}"
    return _serialize_blocks((en.content, zh.content)).rstrip("\n")


def _english_markdown(blocks: Sequence[_Block]) -> str:
    return _serialize_blocks(block.content for block in blocks)


def _bilingual_markdown(
    en_blocks: Sequence[_Block], zh_blocks: Sequence[_Block], shared_ids: frozenset[str]
) -> str:
    zh_by_id = {block.block_id: block for block in zh_blocks}
    parts: list[str] = []
    for en in en_blocks:
        if en.block_id in shared_ids:
            parts.append(en.content)
        else:
            parts.append(_merge_blocks(en, zh_by_id[en.block_id]))
    return _serialize_blocks(parts)


def _prepare_markdown_sources(
    sources: _DocumentSources, build_root: Path
) -> dict[str, str]:
    manuscript_bilingual = _bilingual_markdown(
        sources.manuscript_en, sources.manuscript_zh, sources.shared_ids["manuscript"]
    )
    supplementary_bilingual = _bilingual_markdown(
        sources.supplementary_en,
        sources.supplementary_zh,
        sources.shared_ids["supplementary"],
    )
    outputs = {
        "manuscript_bilingual": manuscript_bilingual,
        "supplementary_bilingual": supplementary_bilingual,
    }
    for stem, content in outputs.items():
        (build_root / f"{stem}.md").write_text(content, encoding="utf-8", newline="\n")
    return {
        "manuscript_en": _english_markdown(sources.manuscript_en),
        "manuscript_bilingual": manuscript_bilingual,
        "supplementary_en": _english_markdown(sources.supplementary_en),
        "supplementary_bilingual": supplementary_bilingual,
    }


def _parse_languages(languages: str) -> tuple[str, ...]:
    if not isinstance(languages, str):
        raise ValueError("languages must be a comma-separated string containing only en and bilingual")
    values = [value.strip() for value in languages.split(",") if value.strip()]
    if not values or any(value not in {"en", "bilingual"} for value in values):
        raise ValueError("languages must contain only en and bilingual")
    if len(values) != len(set(values)):
        duplicate = next(value for value in values if values.count(value) > 1)
        raise ValueError(f"duplicate language value: {duplicate}")
    return tuple(value for value in ("en", "bilingual") if value in values)


def _find_pandoc() -> tuple[str, str | None]:
    pypandoc_version: str | None = None
    try:
        import pypandoc  # type: ignore

        pypandoc_version = getattr(pypandoc, "__version__", None)
        try:
            candidate = pypandoc.get_pandoc_path()
        except Exception:
            candidate = None
        if candidate:
            candidate_path = Path(str(candidate))
            if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
                return str(candidate_path), pypandoc_version
    except ImportError:
        pass

    candidate = shutil.which("pandoc")
    if candidate:
        return candidate, pypandoc_version
    raise FileNotFoundError("pandoc was not found via pypandoc.get_pandoc_path() or PATH")


def _find_xelatex() -> str:
    candidate = shutil.which("xelatex")
    if not candidate:
        raise FileNotFoundError("xelatex was not found on PATH")
    return candidate


def _build_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    return env


def _command_text(command: Sequence[str]) -> str:
    return shlex.join([str(item) for item in command])


class _CommandRunner:
    def __init__(self, build_root: Path, cwd: Path, env: Mapping[str, str]) -> None:
        self.build_root = build_root
        self.cwd = cwd
        self.env = dict(env)
        self.records: list[dict[str, Any]] = []

    def run(self, purpose: str, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        index = len(self.records) + 1
        safe_purpose = re.sub(r"[^A-Za-z0-9_.-]+", "_", purpose).strip("_") or "command"
        log_path = self.build_root / f"{index:02d}_{safe_purpose}.log"
        command_list = [str(item) for item in command]
        record: dict[str, Any] = {
            "purpose": purpose,
            "command": command_list,
            "cwd": str(self.cwd),
            "log": str(log_path),
            "exit_code": None,
        }
        try:
            completed = subprocess.run(
                command_list,
                cwd=str(self.cwd),
                env=self.env,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            record["exit_code"] = exc.returncode
            record["stdout"] = exc.stdout or ""
            record["stderr"] = exc.stderr or ""
            self.records.append(record)
            _write_command_log(log_path, purpose, command_list, exc.returncode, exc.stdout, exc.stderr)
            raise BuildError(
                f"Command failed with exit code {exc.returncode}: {_command_text(command_list)}; "
                f"see {log_path}"
            ) from exc
        except OSError as exc:
            record["error"] = str(exc)
            self.records.append(record)
            _write_command_log(log_path, purpose, command_list, None, "", str(exc))
            raise BuildError(
                f"Command could not start: {_command_text(command_list)}; see {log_path}: {exc}"
            ) from exc
        record["exit_code"] = completed.returncode
        record["stdout"] = completed.stdout or ""
        record["stderr"] = completed.stderr or ""
        self.records.append(record)
        _write_command_log(
            log_path,
            purpose,
            command_list,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
        return completed


def _write_command_log(
    path: Path,
    purpose: str,
    command: Sequence[str],
    exit_code: int | None,
    stdout: str | None,
    stderr: str | None,
) -> None:
    content = [
        f"purpose: {purpose}",
        f"command: {_command_text(command)}",
        f"exit_code: {exit_code}",
        "",
        "[stdout]",
        stdout or "",
        "",
        "[stderr]",
        stderr or "",
        "",
    ]
    path.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8", newline="\n")


def _patch_docx_fonts(reference_docx: Path) -> None:
    ElementTree.register_namespace("w", _W_NS)
    temporary = reference_docx.with_name(reference_docx.name + ".patched")
    with zipfile.ZipFile(reference_docx, "r") as source_zip, zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED
    ) as target_zip:
        for info in source_zip.infolist():
            data = source_zip.read(info.filename)
            if info.filename == "word/styles.xml":
                data = _patch_styles_xml(data)
            target_zip.writestr(info, data)
    temporary.replace(reference_docx)


def _patch_styles_xml(data: bytes) -> bytes:
    root = ElementTree.fromstring(data)
    font_attrs = {
        _W + "ascii": "Liberation Serif",
        _W + "hAnsi": "Liberation Serif",
        _W + "eastAsia": "WenQuanYi Zen Hei",
        _W + "cs": "WenQuanYi Zen Hei",
    }
    theme_attrs = {
        _W + "asciiTheme",
        _W + "hAnsiTheme",
        _W + "eastAsiaTheme",
        _W + "csTheme",
    }
    for fonts in root.iter(_W + "rFonts"):
        for attr in theme_attrs:
            fonts.attrib.pop(attr, None)
        fonts.attrib.update(font_attrs)

    doc_defaults = root.find(_W + "docDefaults")
    if doc_defaults is None:
        doc_defaults = ElementTree.SubElement(root, _W + "docDefaults")
    rpr_default = doc_defaults.find(_W + "rPrDefault")
    if rpr_default is None:
        rpr_default = ElementTree.SubElement(doc_defaults, _W + "rPrDefault")
    rpr = rpr_default.find(_W + "rPr")
    if rpr is None:
        rpr = ElementTree.SubElement(rpr_default, _W + "rPr")
    fonts = rpr.find(_W + "rFonts")
    if fonts is None:
        fonts = ElementTree.SubElement(rpr, _W + "rFonts")
    for attr in theme_attrs:
        fonts.attrib.pop(attr, None)
    fonts.attrib.update(font_attrs)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _make_pdf_header(path: Path) -> None:
    header = r"""\usepackage{fontspec}
\usepackage{xeCJK}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{longtable}
\usepackage{etoolbox}
\newcommand{\Needspace}[1]{\par\begingroup
\dimen0=#1\relax
\dimen2=\pagegoal\advance\dimen2 by -\pagetotal
\ifdim\dimen2<\dimen0\newpage\fi
\endgroup}
\geometry{a4paper,margin=20mm}
\setmainfont{Liberation Serif}
\setsansfont{Liberation Sans}
\setCJKmainfont{WenQuanYi Zen Hei}
\setCJKsansfont{WenQuanYi Zen Hei}
\setstretch{1.12}
\AtBeginEnvironment{longtable}{\small}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt minus 0pt
"""
    path.write_text(header, encoding="utf-8", newline="\n")


def _replace_pdf_images(markdown: str, paper_root: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        relative_png = match.group("path")
        relative_pdf = relative_png[:-4] + ".pdf"
        if (paper_root / relative_pdf).is_file():
            return f"{match.group('open')}{relative_pdf}{match.group('close')}"
        return match.group(0)

    text = _PDF_IMAGE_RE.sub(replace, markdown)
    if "# Supplementary material" not in text:
        # Main tables are short: reserve room for their complete caption and rows.
        def reserve_table(match):
            lines = {"1":20,"2":24,"3":21,"4":20,"5":35}[match.group(2)]
            return "```{=latex}\n" + r"\Needspace{" + str(lines) + r"\baselineskip}" + "\n```\n\n" + match.group(1)
        text = re.sub(r"(?m)^(\*\*Table ([1-5])\.)", reserve_table, text)
    else:
        # A supplement table may span pages; keep both caption languages and
        # its opening rows together, then allow normal repeated-header breaks.
        text = re.sub(r"(?m)^(\*\*Table S\d+[a-z]?\.)", lambda m:
                      "```{=latex}\n" + r"\Needspace{12\baselineskip}" +
                      "\n```\n\n" + m.group(1), text)
    # Keep each real figure with its manually supplied bilingual caption and
    # suppress Pandoc's duplicate implicit caption/LaTeX float.
    pattern = r"(!\[[^\n]*\n\n\*\*Figure[^\n]+(?:\n\n\*\*图[^\n]+)?)"
    return re.sub(pattern, lambda m: "```{=latex}\n\\begin{minipage}{\\linewidth}\n```\n\n" + m.group(1) + "\n\n```{=latex}\n\\end{minipage}\n```", text)


def _make_resource_path(paper_root: Path) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return os.pathsep.join(str(path) for path in (paper_root, paper_root / "source", repo_root))


def _make_reference_docx(
    runner: _CommandRunner, pandoc: str, temporary_root: Path, resource_path: str
) -> Path:
    source = temporary_root / "reference.md"
    reference_docx = temporary_root / "reference.docx"
    source.write_text("# Reference\n\nReference text.\n", encoding="utf-8", newline="\n")
    runner.run(
        "reference_docx",
        [
            pandoc,
            "--from=markdown-implicit_figures",
            "--to=docx",
            str(source),
            f"--output={reference_docx}",
            f"--resource-path={resource_path}",
        ],
    )
    _patch_docx_fonts(reference_docx)
    return reference_docx


def _pandoc_docx_command(
    pandoc: str, source: Path, output: Path, reference_docx: Path, resource_path: str
) -> list[str]:
    return [
        pandoc,
        "--from=markdown-implicit_figures",
        "--to=docx",
        str(source),
        f"--output={output}",
        f"--reference-doc={reference_docx}",
        f"--resource-path={resource_path}",
    ]


def _pandoc_pdf_command(
    pandoc: str,
    xelatex: str,
    source: Path,
    output: Path,
    header: Path,
    resource_path: str,
) -> list[str]:
    return [
        pandoc,
        "--from=markdown-implicit_figures",
        "--to=pdf",
        str(source),
        f"--output={output}",
        f"--pdf-engine={xelatex}",
        f"--include-in-header={header}",
        "--variable=papersize:a4",
        "--variable=geometry:margin=20mm",
        "--variable=fontsize:11pt",
        "--variable=mainfont:Liberation Serif",
        "--variable=sansfont:Liberation Sans",
        "--variable=CJKmainfont:WenQuanYi Zen Hei",
        "--variable=linestretch:1.12",
        f"--resource-path={resource_path}",
    ]


def _pdf_page_count(path: Path) -> tuple[int | None, str | None]:
    try:
        import fitz  # type: ignore
    except ImportError:
        return None, "fitz is not installed"
    try:
        with fitz.open(path) as document:
            return document.page_count, None
    except Exception as exc:  # pragma: no cover - depends on external PDF parser
        return None, str(exc)


def _docx_omath_count(path: Path) -> int:
    count = 0
    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue
            try:
                root = ElementTree.fromstring(archive.read(name))
            except ElementTree.ParseError:
                continue
            count += sum(1 for element in root.iter() if element.tag == "{" + _M_NS + "}oMath")
    return count


def _version_line(output: str) -> str | None:
    for line in output.splitlines():
        if line.strip():
            return line.strip()
    return None


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _output_record(path: Path, kind: str) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "kind": kind, "bytes": path.stat().st_size}
    if kind == "pdf":
        pages, error = _pdf_page_count(path)
        record["pdf_pages"] = pages
        if error:
            record["pdf_page_count_error"] = error
    elif kind == "docx":
        record["omath_count"] = _docx_omath_count(path)
    return record


def build_documents(
    paper_root: Path, artifact_root: Path, languages: str = "en,bilingual"
) -> dict[str, Any]:
    """Build selected English/bilingual manuscript and supplementary outputs."""
    paper_root = Path(paper_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    selected_languages = _parse_languages(languages)
    if not paper_root.is_dir():
        raise FileNotFoundError(f"paper_root is not a directory: {paper_root}")

    sources = _load_sources(paper_root)
    pandoc, pypandoc_version = _find_pandoc()
    xelatex = _find_xelatex()
    build_root = paper_root / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    runner = _CommandRunner(build_root, paper_root, _build_environment())
    resource_path = _make_resource_path(paper_root)
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "pypandoc": pypandoc_version,
        "pandoc_path": pandoc,
        "xelatex_path": xelatex,
    }
    markdown_paths: dict[str, str] = {}
    output_paths: dict[str, Path] = {}

    try:
        pandoc_version = runner.run("pandoc_version", [pandoc, "--version"])
        xelatex_version = runner.run("xelatex_version", [xelatex, "--version"])
        versions["pandoc"] = _version_line(pandoc_version.stdout)
        versions["xelatex"] = _version_line(xelatex_version.stdout)

        markdown_text = _prepare_markdown_sources(sources, build_root)
        markdown_paths = {
            "manuscript_bilingual": str(build_root / "manuscript_bilingual.md"),
            "supplementary_bilingual": str(build_root / "supplementary_bilingual.md"),
        }
        with tempfile.TemporaryDirectory(prefix="cai-doc-build-", dir=str(build_root)) as temp_name:
            temporary_root = Path(temp_name)
            reference_docx = _make_reference_docx(runner, pandoc, temporary_root, resource_path)
            header = temporary_root / "pdf_header.tex"
            _make_pdf_header(header)

            for document_name in ("manuscript", "supplementary"):
                for language in selected_languages:
                    stem = f"{document_name}_{language}"
                    source_text = markdown_text[stem]
                    source = temporary_root / f"{stem}.md"
                    source.write_text(source_text, encoding="utf-8", newline="\n")
                    docx_path = build_root / f"{stem}.docx"
                    pdf_source = temporary_root / f"{stem}_pdf.md"
                    pdf_source.write_text(
                        _replace_pdf_images(source_text, paper_root),
                        encoding="utf-8",
                        newline="\n",
                    )
                    pdf_path = build_root / f"{stem}.pdf"
                    output_paths[f"{stem}.docx"] = docx_path
                    output_paths[f"{stem}.pdf"] = pdf_path
                    runner.run(
                        f"{stem}_docx",
                        _pandoc_docx_command(
                            pandoc, source, docx_path, reference_docx, resource_path
                        ),
                    )
                    runner.run(
                        f"{stem}_pdf",
                        _pandoc_pdf_command(
                            pandoc, xelatex, pdf_source, pdf_path, header, resource_path
                        ),
                    )

        outputs = {
            filename: _output_record(path, path.suffix.lstrip("."))
            for filename, path in output_paths.items()
        }
        report: dict[str, Any] = {
            "status": "success",
            "paper_root": str(paper_root),
            "artifact_root": str(artifact_root),
            "build_root": str(build_root),
            "languages": list(selected_languages),
            "resource_path": resource_path,
            "versions": versions,
            "commands": runner.records,
            "markdown": markdown_paths,
            "outputs": outputs,
            "source_blocks": {
                "manuscript_en": len(sources.manuscript_en),
                "manuscript_zh": len(sources.manuscript_zh),
                "supplementary_en": len(sources.supplementary_en),
                "supplementary_zh": len(sources.supplementary_zh),
                "manuscript_shared": sorted(sources.shared_ids["manuscript"]),
                "supplementary_shared": sorted(sources.shared_ids["supplementary"]),
            },
        }
        _write_report(artifact_root / "BUILD_REPORT.json", report)
        return report
    except Exception as exc:
        failure_report: dict[str, Any] = {
            "status": "failed",
            "paper_root": str(paper_root),
            "artifact_root": str(artifact_root),
            "build_root": str(build_root),
            "languages": list(selected_languages),
            "resource_path": resource_path,
            "versions": versions,
            "commands": runner.records,
            "markdown": markdown_paths,
            "outputs": {},
            "error": str(exc),
        }
        for filename, path in output_paths.items():
            if path.is_file():
                try:
                    failure_report["outputs"][filename] = _output_record(
                        path, path.suffix.lstrip(".")
                    )
                except Exception as output_exc:  # pragma: no cover - partial failure path
                    failure_report["outputs"][filename] = {"path": str(path), "error": str(output_exc)}
        try:
            _write_report(artifact_root / "BUILD_REPORT.json", failure_report)
        except OSError:
            pass
        raise


__all__ = ["BuildError", "build_documents"]
