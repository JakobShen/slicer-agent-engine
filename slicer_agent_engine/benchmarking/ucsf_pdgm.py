from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


_SEGMENTATION_NAME_HINTS = (
    "tumor_segmentation",
    "tumour_segmentation",
    "segmentation",
    "tumor_mask",
    "tumour_mask",
    "mask",
    "label",
    "seg",
)


@dataclass(frozen=True)
class UCSFPDGMCase:
    case_id: str
    csv_id: str
    case_dir: Path
    nifti_files: List[Path]
    diagnosis: Optional[str]
    label: Optional[str]  # "A"/"B"/"C" or None if unknown
    segmentation_file: Optional[Path] = None


def folder_to_csv_id(folder_name: str) -> Optional[str]:
    """Map UCSF-PDGM folder names to metadata CSV IDs.

    Examples:
      - UCSF-PDGM-0004_nifti -> UCSF-PDGM-004
      - UCSF-PDGM-004_nifti  -> UCSF-PDGM-004
      - UCSF-PDGM-0004       -> UCSF-PDGM-004
    """

    m = re.search(r"UCSF-PDGM-(\d+)", folder_name)
    if not m:
        return None
    n = int(m.group(1))
    return f"UCSF-PDGM-{n:03d}"


def diagnosis_to_label(diagnosis: Optional[str]) -> Optional[str]:
    """Convert a pathology diagnosis string into the 3-way benchmark label.

    A: Glioblastoma
    B: Oligodendroglioma / Astrocytoma
    C: No tumor
    """

    if diagnosis is None:
        return None
    d = str(diagnosis).strip().lower()
    if not d or d == "nan":
        return None
    if "glioblastoma" in d:
        return "A"
    if ("oligodendroglioma" in d) or ("astrocytoma" in d):
        return "B"
    if ("no tumor" in d) or ("no tumour" in d) or ("normal" in d) or ("control" in d):
        return "C"
    return None


def diagnosis_subtype(diagnosis: Optional[str]) -> Optional[str]:
    """Return a canonical subtype key used for balanced UCSF-PDGM sampling."""

    if diagnosis is None:
        return None
    d = str(diagnosis).strip().lower()
    if not d or d == "nan":
        return None
    if "glioblastoma" in d:
        return "Glioblastoma"
    if "oligodendroglioma" in d:
        return "Oligodendroglioma"
    if "astrocytoma" in d:
        return "Astrocytoma"
    return None


def read_metadata(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load the UCSF-PDGM metadata CSV into a dict keyed by ID."""

    csv_path = Path(csv_path).expanduser().resolve()
    meta: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("ID")
            if not rid:
                continue
            meta[str(rid)] = row
    return meta


def _is_nifti_path(path: Path) -> bool:
    name = str(path.name).lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _looks_like_segmentation_name(name: str) -> bool:
    name_l = str(name or "").lower()
    return any(token in name_l for token in _SEGMENTATION_NAME_HINTS)


def find_segmentation_file(case_dir: Path) -> Optional[Path]:
    """Return the most likely NIfTI segmentation file inside a UCSF-PDGM case folder."""

    case_dir = Path(case_dir)
    if not case_dir.exists() or not case_dir.is_dir():
        return None

    candidates: List[Path] = []
    for path in sorted(case_dir.iterdir()):
        if not path.is_file() or not _is_nifti_path(path):
            continue
        if _looks_like_segmentation_name(path.name):
            candidates.append(path)
    if not candidates:
        return None

    def _score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        score = 0
        if "tumor_segmentation" in name or "tumour_segmentation" in name:
            score += 100
        if "segmentation" in name:
            score += 50
        if "tumor" in name or "tumour" in name:
            score += 20
        if "mask" in name:
            score += 10
        if "label" in name:
            score += 5
        return (-score, name)

    candidates.sort(key=_score)
    return candidates[0]


def select_core_modalities(
    case_dir: Path,
    *,
    modalities: Sequence[str] = ("T1c", "FLAIR", "T2", "T1"),
    prefer_bias: bool = True,
    include_segmentations: bool = False,
    include_extra: bool = False,
) -> List[Path]:
    """Pick a deterministic subset of NIfTI files for tumor classification.

    - Prefers bias-corrected versions when present (e.g. *_T1c_bias.nii.gz)
    - Excludes segmentation masks by default to avoid label leakage
    - If include_extra=True, returns *all* non-seg NIfTI files (sorted)
    """

    case_dir = Path(case_dir)
    all_files = sorted([p for p in case_dir.iterdir() if p.is_file() and _is_nifti_path(p)])
    if not include_segmentations:
        all_files = [p for p in all_files if not _looks_like_segmentation_name(p.name)]

    if include_extra:
        return all_files

    def pick(mod: str) -> Optional[Path]:
        mod_l = mod.lower()
        cands = [p for p in all_files if mod_l in p.name.lower()]
        if mod_l == "t1":
            cands = [p for p in cands if "t1c" not in p.name.lower()]
        if not cands:
            return None
        if prefer_bias:
            bias = [p for p in cands if "bias" in p.name.lower()]
            if bias:
                return sorted(bias)[0]
        return sorted(cands)[0]

    chosen: List[Path] = []
    for m in modalities:
        p = pick(m)
        if p is not None:
            chosen.append(p)

    return chosen if chosen else all_files


def iter_cases(
    data_root: Path,
    metadata_csv: Path,
    *,
    limit: Optional[int] = None,
    include_extra: bool = False,
    include_segmentations: bool = False,
    prefer_bias: bool = True,
    balance_diagnosis_subtypes: bool = False,
) -> Iterable[UCSFPDGMCase]:
    """Yield UCSF-PDGM cases from ``data_root``.

    Expects per-case folders like ``UCSF-PDGM-0004_nifti``.
    """

    data_root = Path(data_root).expanduser().resolve()
    meta = read_metadata(metadata_csv)

    case_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("UCSF-PDGM-") and "nifti" in p.name.lower()])

    cases: List[UCSFPDGMCase] = []
    for case_dir in case_dirs:
        csv_id = folder_to_csv_id(case_dir.name)
        if not csv_id:
            continue
        row = meta.get(csv_id, {})
        diagnosis = row.get("Final pathologic diagnosis (WHO 2021)") or row.get("Final pathologic diagnosis")
        label = diagnosis_to_label(diagnosis)
        nifti_files = select_core_modalities(
            case_dir,
            prefer_bias=prefer_bias,
            include_segmentations=include_segmentations,
            include_extra=include_extra,
        )
        cases.append(
            UCSFPDGMCase(
                case_id=case_dir.name,
                csv_id=csv_id,
                case_dir=case_dir,
                nifti_files=nifti_files,
                diagnosis=diagnosis,
                label=label,
                segmentation_file=find_segmentation_file(case_dir),
            )
        )

    if balance_diagnosis_subtypes:
        buckets: Dict[str, List[UCSFPDGMCase]] = {
            "Glioblastoma": [],
            "Oligodendroglioma": [],
            "Astrocytoma": [],
            "__other__": [],
        }
        for case in cases:
            subtype = diagnosis_subtype(case.diagnosis)
            if subtype in buckets:
                buckets[subtype].append(case)
            else:
                buckets["__other__"].append(case)

        balanced: List[UCSFPDGMCase] = []
        ordered_keys = ["Glioblastoma", "Oligodendroglioma", "Astrocytoma"]
        while any(buckets[key] for key in ordered_keys):
            for key in ordered_keys:
                if buckets[key]:
                    balanced.append(buckets[key].pop(0))
        balanced.extend(buckets["__other__"])
        cases = balanced

    if limit and limit > 0:
        cases = cases[: int(limit)]

    for case in cases:
        yield case
