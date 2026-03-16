"""Slicer-side fixed entry bridge for `/slicer/exec`.

This file is imported inside a running 3D Slicer Python environment.

Design constraints
-----------------
- Keep the public surface **small** and **stable** (`dispatch(payload)` only).
- All args/results must be JSON-serializable (basic types).
- Prefer deterministic operations; avoid UI dependencies where possible.

Security note
-------------
Anything callable through `/slicer/exec` runs with the same permissions as the Slicer process.
Do not expose exec to untrusted networks.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import slicer


BRIDGE_VERSION = "0.5.0"

_SLICE_VIEW_MAP = {
    "axial": "Red",
    "red": "Red",
    "r": "Red",
    "sagittal": "Yellow",
    "yellow": "Yellow",
    "y": "Yellow",
    "coronal": "Green",
    "green": "Green",
    "g": "Green",
}
_PUBLIC_VIEW_MAP = {
    "axial": "axial",
    "red": "axial",
    "r": "axial",
    "sagittal": "sagittal",
    "yellow": "sagittal",
    "y": "sagittal",
    "coronal": "coronal",
    "green": "coronal",
    "g": "coronal",
}
_SLICE_PUBLIC_VIEWS = (
    ("red", "axial"),
    ("yellow", "sagittal"),
    ("green", "coronal"),
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _set_active_volume(volume_node) -> None:
    app_logic = slicer.app.applicationLogic()
    sel = app_logic.GetSelectionNode()
    sel.SetReferenceActiveVolumeID(volume_node.GetID())
    app_logic.PropagateVolumeSelection(0)


def _get_node_by_id_or_name(node_id: Optional[str] = None, node_name: Optional[str] = None):
    if node_id:
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node:
            return node
    if node_name:
        return slicer.util.getNode(node_name)
    raise ValueError("Must provide node_id or node_name")


def _get_active_volume_node():
    app_logic = slicer.app.applicationLogic()
    sel = app_logic.GetSelectionNode()
    vid = sel.GetActiveVolumeID()
    if vid:
        n = slicer.mrmlScene.GetNodeByID(vid)
        if n:
            return n
    vols = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    return vols[0] if vols else None


def _require_layout_manager():
    lm = slicer.app.layoutManager()
    if lm is None:
        raise RuntimeError(
            "Slicer layoutManager is not available. "
            "This typically means Slicer was started without a main window. "
            "Slice rendering tools require a main window (or Xvfb on headless Linux)."
        )
    return lm


def _norm_view(view: str) -> str:
    v = (view or "").strip().lower()
    if v in _SLICE_VIEW_MAP:
        return _SLICE_VIEW_MAP[v]
    raise ValueError(f"Unsupported view: {view!r} (expected axial/sagittal/coronal)")


def _public_view(view: str) -> str:
    v = (view or "").strip().lower()
    if v in _PUBLIC_VIEW_MAP:
        return _PUBLIC_VIEW_MAP[v]
    raise ValueError(f"Unsupported view: {view!r} (expected axial/sagittal/coronal)")


def _slice_widget(view: str):
    lm = _require_layout_manager()
    w = lm.sliceWidget(_norm_view(view))
    if w is None:
        raise RuntimeError(f"Slice widget not found for view={view!r}")
    return w


def _slice_node(view: str):
    return _slice_widget(view).mrmlSliceNode()


def _slice_logic(view: str):
    return _slice_widget(view).sliceLogic()


def _slice_composite_node(view: str):
    return _slice_widget(view).mrmlSliceCompositeNode()


def _node_by_id(node_id: Optional[str]):
    if not node_id:
        return None
    try:
        return slicer.mrmlScene.GetNodeByID(str(node_id))
    except Exception:
        return None


def _slice_volume_nodes(view: str):
    comp = _slice_composite_node(view)
    bg_id = str(comp.GetBackgroundVolumeID()) if comp and hasattr(comp, "GetBackgroundVolumeID") else None
    fg_id = str(comp.GetForegroundVolumeID()) if comp and hasattr(comp, "GetForegroundVolumeID") else None
    label_id = str(comp.GetLabelVolumeID()) if comp and hasattr(comp, "GetLabelVolumeID") else None
    return _node_by_id(bg_id), _node_by_id(fg_id), _node_by_id(label_id)


def _best_slice_volume_node(view: str):
    bg_node, fg_node, label_node = _slice_volume_nodes(view)
    for node in (bg_node, fg_node, label_node, _get_active_volume_node()):
        if node is not None and hasattr(node, "IsA") and (node.IsA("vtkMRMLScalarVolumeNode") or node.IsA("vtkMRMLLabelMapVolumeNode")):
            return node
    return None


def _volume_ras_bounds(volume_node) -> List[float]:
    bounds = [0.0] * 6
    if volume_node is None:
        raise RuntimeError("No volume node for bounds computation")
    if hasattr(volume_node, "GetRASBounds"):
        volume_node.GetRASBounds(bounds)
        return [float(x) for x in bounds]
    if hasattr(volume_node, "GetBounds"):
        volume_node.GetBounds(bounds)
        return [float(x) for x in bounds]
    raise RuntimeError("Volume node does not expose RAS bounds")


def _slice_normal_ras(sn) -> List[float]:
    try:
        m = sn.GetSliceToRAS()
        n = [float(m.GetElement(0, 2)), float(m.GetElement(1, 2)), float(m.GetElement(2, 2))]
        mag = math.sqrt(sum(v * v for v in n))
        if mag > 0:
            return [v / mag for v in n]
    except Exception:
        pass
    orientation = str(sn.GetOrientationString() if hasattr(sn, "GetOrientationString") else "").lower()
    if "sag" in orientation:
        return [1.0, 0.0, 0.0]
    if "cor" in orientation:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _project_bounds_onto_normal(bounds: Sequence[float], normal: Sequence[float]) -> Tuple[float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
    nx, ny, nz = [float(v) for v in normal]
    projections: List[float] = []
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            for z in (zmin, zmax):
                projections.append(x * nx + y * ny + z * nz)
    return float(min(projections)), float(max(projections))


def _fallback_slice_offset_range(*, view: str) -> Dict[str, Any]:
    sn = _slice_node(view)
    volume_node = _best_slice_volume_node(view)
    if volume_node is None:
        raise RuntimeError("No visible volume available to estimate slice offset range.")
    bounds = _volume_ras_bounds(volume_node)
    lo, hi = _project_bounds_onto_normal(bounds, _slice_normal_ras(sn))
    if hi < lo:
        lo, hi = hi, lo
    return {
        "view": _public_view(view),
        "min_offset": float(lo),
        "max_offset": float(hi),
        "derived_from": "volume_ras_bounds",
        "source_volume_id": volume_node.GetID(),
        "source_volume_name": volume_node.GetName(),
    }


def _reset_interaction_state() -> Dict[str, Any]:
    details: Dict[str, Any] = {"ok": True}
    try:
        app_logic = slicer.app.applicationLogic()
        interaction_node = app_logic.GetInteractionNode() if app_logic is not None else None
        if interaction_node is not None:
            if hasattr(interaction_node, "SetCurrentInteractionMode") and hasattr(slicer.vtkMRMLInteractionNode, "ViewTransform"):
                interaction_node.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
            if hasattr(interaction_node, "SetPlaceModePersistence"):
                interaction_node.SetPlaceModePersistence(0)
            details["interaction_mode"] = int(interaction_node.GetCurrentInteractionMode()) if hasattr(interaction_node, "GetCurrentInteractionMode") else None
    except Exception as e:
        details["interaction_warning"] = str(e)
    slicer.app.processEvents()
    return details


def _capture_slice_view_png(*, view: str, output_path: str, include_controller: bool = False) -> Dict[str, Any]:
    output_path = os.path.abspath(os.path.expanduser(str(output_path)))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sw = _slice_widget(view)
    widget = sw if include_controller else sw.sliceView()
    pixmap = widget.grab()
    ok = bool(pixmap.save(output_path))
    if not ok:
        raise RuntimeError(f"Failed to save slice view capture for {view!r} to {output_path}")
    slicer.app.processEvents()
    return {"ok": True, "view": _public_view(view), "output_path": output_path, "include_controller": bool(include_controller)}


def _jump_slices_to_location(x: float, y: float, z: float, centered: bool = True) -> None:
    if hasattr(slicer.util, "jumpSlicesToLocation"):
        try:
            slicer.util.jumpSlicesToLocation(float(x), float(y), float(z), bool(centered))
            slicer.app.processEvents()
            return
        except Exception:
            pass
    markups_logic_factory = getattr(getattr(slicer.modules, "markups", None), "logic", None)
    if callable(markups_logic_factory):
        logic = markups_logic_factory()
        if logic is not None and hasattr(logic, "JumpSlicesToLocation"):
            logic.JumpSlicesToLocation(float(x), float(y), float(z), bool(centered))
            slicer.app.processEvents()
            return
    raise RuntimeError("No supported JumpSlicesToLocation API is available in this Slicer build.")


def _safe_slab_type_as_string(sn) -> Optional[str]:
    """Best-effort slab reconstruction type string across Slicer/VTK wrapper variants.

    Some Slicer builds expose `GetSlabReconstructionTypeAsString(int)` (utility/static method),
    while others expose `GetSlabReconstructionTypeAsString()` (no-arg instance method).

    We avoid raising here because slab state is often informational.
    """

    if not hasattr(sn, "GetSlabReconstructionTypeAsString"):
        return None

    fn = getattr(sn, "GetSlabReconstructionTypeAsString")

    # Variant B (most common in recent wrappers): requires the type argument.
    try:
        t = int(sn.GetSlabReconstructionType()) if hasattr(sn, "GetSlabReconstructionType") else None
        if t is not None:
            return str(fn(t))
    except Exception:
        pass

    # Variant A: no-arg
    try:
        return str(fn())
    except Exception:
        return None


def _slab_type_code_for(sn, target: str) -> Optional[int]:
    """Return the integer slab reconstruction type code for a desired target.

    Slicer/VTK wrapper variants differ across versions:
      - Some expose SetSlabReconstructionTypeToMax/Min/Mean
      - Some only expose SetSlabReconstructionType(int)
      - The integer-to-mode mapping is not guaranteed stable

    We therefore discover the mapping dynamically using
    GetSlabReconstructionTypeAsString(int) when available.
    """

    if not hasattr(sn, "GetSlabReconstructionTypeAsString"):
        return None

    fn = getattr(sn, "GetSlabReconstructionTypeAsString")

    # Only works for the variant that takes the type argument.
    try:
        _ = fn(0)
        takes_arg = True
    except TypeError:
        takes_arg = False
    except Exception:
        takes_arg = True
    if not takes_arg:
        return None

    t = (target or "").strip().lower()
    if t in {"mip", "max", "maximum"}:
        keywords = {"max", "maximum", "mip"}
    elif t in {"min", "minimum"}:
        keywords = {"min", "minimum"}
    elif t in {"mean", "avg", "average"}:
        keywords = {"mean", "avg", "average"}
    else:
        return None

    # Scan a small range of likely enum values.
    for code in range(0, 8):
        try:
            s = str(fn(int(code))).lower()
        except Exception:
            continue
        if any(k in s for k in keywords):
            return int(code)
    return None


def _volume_display_node(volume_node):
    if volume_node is None:
        return None
    dn = volume_node.GetDisplayNode()
    if dn is None:
        # Create default display nodes if missing
        try:
            slicer.modules.volumes.logic().CreateDefaultVolumeDisplayNodes(volume_node)
        except Exception:
            pass
        dn = volume_node.GetDisplayNode()
    return dn


def _is_nifti_path(p: str) -> bool:
    pl = (p or "").lower()
    return pl.endswith(".nii") or pl.endswith(".nii.gz")


def _load_volume_file(path: str, *, name: Optional[str] = None):
    """Load a scalar volume from disk (NIfTI, NRRD, etc.).

    We primarily use this for NIfTI (*.nii, *.nii.gz).
    The exact ``slicer.util.loadVolume`` signature can vary across versions,
    so we implement a small compatibility shim.
    """

    props = {}
    if name:
        props["name"] = str(name)

    # Newer Slicer: loadVolume(path, properties=..., returnNode=True) -> (success, node)
    try:
        success, node = slicer.util.loadVolume(str(path), properties=props or None, returnNode=True)
        return bool(success), node
    except TypeError:
        pass
    except Exception:
        # Fall through to older call patterns
        pass

    # Older Slicer: loadVolume(path, properties=...) -> node (or raises)
    try:
        node = slicer.util.loadVolume(str(path), properties=props or None)
        return bool(node is not None), node
    except Exception:
        return False, None


def _select_active_by_preference(loaded: List[Dict[str, Any]], *, prefer: Optional[Sequence[str]] = None):
    """Pick an active volume from loaded list using filename/name heuristics."""

    if not loaded:
        return None
    if not prefer:
        return loaded[0]

    prefer_l = [str(x).lower() for x in prefer]
    # Score each loaded item by earliest preference substring match.
    best = None
    best_score = 10**9
    for item in loaded:
        key = f"{item.get('name','')} {item.get('file','')}".lower()
        score = 10**9
        for i, sub in enumerate(prefer_l):
            if sub and sub in key:
                score = i
                break
        if score < best_score:
            best_score = score
            best = item
    return best or loaded[0]


# -----------------------------------------------------------------------------
# L0: Viewer actuation (deterministic)
# -----------------------------------------------------------------------------


def _get_viewer_state() -> Dict[str, Any]:
    vol = _get_active_volume_node()
    display = _volume_display_node(vol)

    state: Dict[str, Any] = {
        "active_volume_id": vol.GetID() if vol else None,
        "active_volume_name": vol.GetName() if vol else None,
        "volume_window": float(display.GetWindow()) if display else None,
        "volume_level": float(display.GetLevel()) if display else None,
        "volume_auto_window_level": bool(display.GetAutoWindowLevel()) if display else None,
        "volume_interpolate": bool(display.GetInterpolate()) if display and hasattr(display, "GetInterpolate") else None,
        "views": {},
    }

    for slicer_view, public_view in _SLICE_PUBLIC_VIEWS:
        try:
            sn = _slice_node(slicer_view)
            comp = _slice_composite_node(slicer_view)
            bg_id = str(comp.GetBackgroundVolumeID()) if comp else None
            fg_id = str(comp.GetForegroundVolumeID()) if comp else None
            bg_node = slicer.mrmlScene.GetNodeByID(bg_id) if bg_id else None
            fg_node = slicer.mrmlScene.GetNodeByID(fg_id) if fg_id else None
            view_state: Dict[str, Any] = {
                "orientation": sn.GetOrientationString() if hasattr(sn, "GetOrientationString") else str(sn.GetOrientation()),
                "slice_offset": float(sn.GetSliceOffset()),
                "field_of_view": list(sn.GetFieldOfView()) if hasattr(sn, "GetFieldOfView") else None,
                "xyz_origin": list(sn.GetXYZOrigin()) if hasattr(sn, "GetXYZOrigin") else None,
                "background_volume_id": bg_id,
                "background_volume_name": bg_node.GetName() if bg_node is not None else None,
                "foreground_volume_id": fg_id,
                "foreground_volume_name": fg_node.GetName() if fg_node is not None else None,
                "foreground_opacity": float(comp.GetForegroundOpacity()) if comp and hasattr(comp, "GetForegroundOpacity") else None,
                "linked_control": int(comp.GetLinkedControl()) if comp and hasattr(comp, "GetLinkedControl") else None,
            }

            try:
                offset_range = _get_slice_offset_range(view=public_view)
                view_state["min_offset"] = offset_range.get("min_offset")
                view_state["max_offset"] = offset_range.get("max_offset")
            except Exception:
                view_state["min_offset"] = None
                view_state["max_offset"] = None

            # Thick slab (if available in this Slicer build)
            if hasattr(sn, "GetSlabReconstructionEnabled"):
                view_state["slab_enabled"] = bool(sn.GetSlabReconstructionEnabled())
            if hasattr(sn, "GetSlabReconstructionThickness"):
                view_state["slab_thickness_mm"] = float(sn.GetSlabReconstructionThickness())
            slab_type_str = _safe_slab_type_as_string(sn)
            if slab_type_str is not None:
                view_state["slab_type"] = slab_type_str
            elif hasattr(sn, "GetSlabReconstructionType"):
                view_state["slab_type"] = int(sn.GetSlabReconstructionType())

            state["views"][public_view] = view_state
        except Exception as e:
            state["views"][public_view] = {"error": str(e)}

    return state


def _set_window_level(
    *,
    window: Optional[float] = None,
    level: Optional[float] = None,
    auto: Optional[bool] = None,
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    vol = None
    if volume_id or volume_name:
        vol = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name)
    else:
        vol = _get_active_volume_node()

    if vol is None:
        raise RuntimeError("No active volume to set window/level on.")

    dn = _volume_display_node(vol)
    if dn is None:
        raise RuntimeError("Volume display node is not available for window/level setting.")

    if auto is not None:
        dn.SetAutoWindowLevel(bool(auto))

    if window is not None and level is not None:
        dn.SetAutoWindowLevel(False)
        dn.SetWindowLevel(float(window), float(level))

    slicer.app.processEvents()

    return {
        "volume_id": vol.GetID(),
        "volume_name": vol.GetName(),
        "auto_window_level": bool(dn.GetAutoWindowLevel()),
        "window": float(dn.GetWindow()),
        "level": float(dn.GetLevel()),
    }


def _set_interpolation(
    *,
    interpolate: bool,
    volume_id: Optional[str] = None,
    volume_name: Optional[str] = None,
) -> Dict[str, Any]:
    vol = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if vol is None:
        raise RuntimeError("No active volume to set interpolation on.")

    dn = _volume_display_node(vol)
    if dn is None or not hasattr(dn, "SetInterpolate"):
        raise RuntimeError("Volume display node does not support interpolation toggling.")

    dn.SetInterpolate(1 if interpolate else 0)
    slicer.app.processEvents()
    return {"volume_id": vol.GetID(), "volume_name": vol.GetName(), "interpolate": bool(dn.GetInterpolate())}


def _set_slice_orientation(*, view: str, orientation: str) -> Dict[str, Any]:
    sn = _slice_node(view)
    o = (orientation or "").strip().lower()

    if o in {"axial", "transverse"}:
        if hasattr(sn, "SetOrientationToAxial"):
            sn.SetOrientationToAxial()
        else:
            sn.SetOrientation("Axial")
    elif o in {"sagittal", "sag"}:
        if hasattr(sn, "SetOrientationToSagittal"):
            sn.SetOrientationToSagittal()
        else:
            sn.SetOrientation("Sagittal")
    elif o in {"coronal", "cor"}:
        if hasattr(sn, "SetOrientationToCoronal"):
            sn.SetOrientationToCoronal()
        else:
            sn.SetOrientation("Coronal")
    else:
        # Allow custom orientation strings (Slicer supports additional named orientations)
        sn.SetOrientation(str(orientation))

    slicer.app.processEvents()
    return {"view": _public_view(view), "orientation": sn.GetOrientationString() if hasattr(sn, "GetOrientationString") else str(sn.GetOrientation())}


def _get_slice_offset_range(*, view: str) -> Dict[str, Any]:
    logic = _slice_logic(view)
    lo = None
    hi = None
    if hasattr(logic, "GetLowestVolumeSliceOffset") and hasattr(logic, "GetHighestVolumeSliceOffset"):
        try:
            lo = float(logic.GetLowestVolumeSliceOffset())
            hi = float(logic.GetHighestVolumeSliceOffset())
        except Exception:
            lo = None
            hi = None
    elif hasattr(logic, "GetSliceOffsetRange"):
        try:
            r = logic.GetSliceOffsetRange()
            lo = float(r[0])
            hi = float(r[1])
        except Exception:
            lo = None
            hi = None

    if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return _fallback_slice_offset_range(view=view)

    if hi < lo:
        lo, hi = hi, lo
    return {"view": _public_view(view), "min_offset": float(lo), "max_offset": float(hi), "derived_from": "slice_logic"}


def _set_slice_offset(*, view: str, offset: float) -> Dict[str, Any]:
    sn = _slice_node(view)
    sn.SetSliceOffset(float(offset))
    slicer.app.processEvents()
    return {"view": _public_view(view), "slice_offset": float(sn.GetSliceOffset())}


def _set_slice_scroll_to(*, view: str, scroll_to: float) -> Dict[str, Any]:
    r = _get_slice_offset_range(view=view)
    lo = float(r["min_offset"])
    hi = float(r["max_offset"])
    t = float(scroll_to)
    t = max(0.0, min(1.0, t))
    off = lo + t * (hi - lo)
    return _set_slice_offset(view=view, offset=off) | {"scroll_to": t, "min_offset": lo, "max_offset": hi}


def _fit_slice(*, view: str) -> Dict[str, Any]:
    logic = _slice_logic(view)
    logic.FitSliceToAll()
    slicer.app.processEvents()
    sn = _slice_node(view)
    return {"view": _public_view(view), "field_of_view": list(sn.GetFieldOfView()), "slice_offset": float(sn.GetSliceOffset())}


def _zoom_slice_relative(*, view: str, factor: float) -> Dict[str, Any]:
    if factor <= 0:
        raise ValueError("factor must be > 0")
    sn = _slice_node(view)
    if not hasattr(sn, "GetFieldOfView") or not hasattr(sn, "SetFieldOfView"):
        raise RuntimeError("SliceNode does not support field-of-view control in this Slicer build.")
    fov = sn.GetFieldOfView()
    sn.SetFieldOfView(float(fov[0]) / float(factor), float(fov[1]) / float(factor), float(fov[2]))
    slicer.app.processEvents()
    return {"view": _public_view(view), "field_of_view": list(sn.GetFieldOfView()), "factor": float(factor)}




def _set_field_of_view(*, view: str, field_of_view: Sequence[float]) -> Dict[str, Any]:
    sn = _slice_node(view)
    if not hasattr(sn, "SetFieldOfView"):
        raise RuntimeError("SliceNode does not support SetFieldOfView in this Slicer build.")
    if len(field_of_view) != 3:
        raise ValueError("field_of_view must be length-3 [x,y,z]")
    fx, fy, fz = [float(v) for v in field_of_view]
    sn.SetFieldOfView(fx, fy, fz)
    slicer.app.processEvents()
    return {"view": _public_view(view), "field_of_view": list(sn.GetFieldOfView())}


def _set_xyz_origin(*, view: str, xyz_origin: Sequence[float]) -> Dict[str, Any]:
    sn = _slice_node(view)
    if not hasattr(sn, "SetXYZOrigin"):
        raise RuntimeError("SliceNode does not support SetXYZOrigin in this Slicer build.")
    if len(xyz_origin) != 3:
        raise ValueError("xyz_origin must be length-3 [x,y,z]")
    x, y, z = [float(v) for v in xyz_origin]
    sn.SetXYZOrigin(x, y, z)
    slicer.app.processEvents()
    return {"view": _public_view(view), "xyz_origin": list(sn.GetXYZOrigin())}
def _jump_to_ras(*, ras: Sequence[float], centered: bool = True) -> Dict[str, Any]:
    if len(ras) != 3:
        raise ValueError("ras must be length-3 [x,y,z] in RAS (mm)")
    x, y, z = [float(v) for v in ras]
    _jump_slices_to_location(x, y, z, centered)
    slicer.app.processEvents()
    return {"ras": [x, y, z], "centered": bool(centered)}


def _volume_center_ras(volume_node) -> List[float]:
    bounds = _volume_ras_bounds(volume_node)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return [float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0), float((zmin + zmax) / 2.0)]


def _recover_standard_views(*, volume_id: Optional[str] = None, volume_name: Optional[str] = None, centered: bool = True, fit: bool = True) -> Dict[str, Any]:
    volume_node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if volume_node is None:
        raise RuntimeError("No volume available to recover standard views.")
    if hasattr(volume_node, "IsA") and (volume_node.IsA("vtkMRMLScalarVolumeNode") or volume_node.IsA("vtkMRMLLabelMapVolumeNode")):
        _set_active_volume(volume_node)

    orientation_results: Dict[str, Any] = {}
    for view_name, orientation in (("axial", "axial"), ("sagittal", "sagittal"), ("coronal", "coronal")):
        orientation_results[view_name] = _set_slice_orientation(view=view_name, orientation=orientation)

    center_ras = _volume_center_ras(volume_node)
    _jump_slices_to_location(center_ras[0], center_ras[1], center_ras[2], centered)

    fit_results: Dict[str, Any] = {}
    if fit:
        for view_name in ("axial", "sagittal", "coronal"):
            fit_results[view_name] = _fit_slice(view=view_name)

    return {
        "volume_id": volume_node.GetID(),
        "volume_name": volume_node.GetName(),
        "center_ras": center_ras,
        "centered": bool(centered),
        "fit": bool(fit),
        "orientations": orientation_results,
        "fit_results": fit_results,
        "viewer_state": _get_viewer_state(),
    }


def _set_linked_slices(*, enabled: bool) -> Dict[str, Any]:
    results: Dict[str, Any] = {"enabled": bool(enabled), "views": {}}
    for slicer_view, public_view in _SLICE_PUBLIC_VIEWS:
        try:
            comp = _slice_composite_node(slicer_view)
            if comp is None:
                continue
            if hasattr(comp, "SetLinkedControl"):
                comp.SetLinkedControl(1 if enabled else 0)
            if hasattr(comp, "SetHotLinkedControl"):
                comp.SetHotLinkedControl(1 if enabled else 0)
            results["views"][public_view] = {
                "linked_control": int(comp.GetLinkedControl()) if hasattr(comp, "GetLinkedControl") else None
            }
        except Exception as e:
            results["views"][public_view] = {"error": str(e)}
    slicer.app.processEvents()
    return results


def _set_layout(*, layout: str) -> Dict[str, Any]:
    """Set a standard Slicer layout.

    Note: not all Slicer builds expose the same layout constants. We resolve
    constants via ``getattr`` with safe fallbacks so that missing optional
    constants do not break the tool.
    """

    _require_layout_manager()
    ln = slicer.app.layoutManager().layoutLogic().GetLayoutNode()

    def _layout_const(primary: str, fallback: str) -> int:
        # vtkMRMLLayoutNode layout constants are integers.
        if hasattr(slicer.vtkMRMLLayoutNode, primary):
            return int(getattr(slicer.vtkMRMLLayoutNode, primary))
        if hasattr(slicer.vtkMRMLLayoutNode, fallback):
            return int(getattr(slicer.vtkMRMLLayoutNode, fallback))
        raise AttributeError(
            f"Neither layout constant {primary!r} nor fallback {fallback!r} exists on vtkMRMLLayoutNode"
        )

    name = (layout or "").strip().lower()
    # Small, deterministic subset of layouts.
    layout_map = {
        "four_up": _layout_const("SlicerLayoutFourUpView", "SlicerLayoutConventionalView"),
        "fourup": _layout_const("SlicerLayoutFourUpView", "SlicerLayoutConventionalView"),
        "conventional": _layout_const("SlicerLayoutConventionalView", "SlicerLayoutFourUpView"),
        # Some builds no longer expose a dedicated 3-slice-only layout.
        # Fall back to FourUp (3 slices + 3D) which is always safe.
        "three_up": _layout_const("SlicerLayoutThreeUpView", "SlicerLayoutFourUpView"),
        "one_up_axial": _layout_const("SlicerLayoutOneUpRedSliceView", "SlicerLayoutFourUpView"),
        "one_up_sagittal": _layout_const("SlicerLayoutOneUpYellowSliceView", "SlicerLayoutFourUpView"),
        "one_up_coronal": _layout_const("SlicerLayoutOneUpGreenSliceView", "SlicerLayoutFourUpView"),
        "one_up_red": _layout_const("SlicerLayoutOneUpRedSliceView", "SlicerLayoutFourUpView"),
        "one_up_yellow": _layout_const("SlicerLayoutOneUpYellowSliceView", "SlicerLayoutFourUpView"),
        "one_up_green": _layout_const("SlicerLayoutOneUpGreenSliceView", "SlicerLayoutFourUpView"),
    }
    if name not in layout_map:
        raise ValueError(f"Unsupported layout: {layout!r}. Supported: {sorted(layout_map.keys())}")
    ln.SetViewArrangement(int(layout_map[name]))
    slicer.app.processEvents()
    return {"layout": name, "layout_id": int(layout_map[name])}


# -----------------------------------------------------------------------------
# L1: Clinical deterministic quantification
# -----------------------------------------------------------------------------


def _set_thick_slab(*, view: str, enabled: bool, thickness_mm: float = 0.0, mode: str = "mip") -> Dict[str, Any]:
    sn = _slice_node(view)
    if not hasattr(sn, "SetSlabReconstructionEnabled"):
        raise RuntimeError("This Slicer build does not support thick slab reconstruction on SliceNode.")

    sn.SetSlabReconstructionEnabled(1 if enabled else 0)

    if hasattr(sn, "SetSlabReconstructionThickness"):
        sn.SetSlabReconstructionThickness(float(thickness_mm))

    # Best-effort mapping across Slicer builds.
    mode_l = (mode or "").strip().lower()

    if mode_l in {"mip", "max", "maximum"}:
        target = "max"
        keywords = {"max", "maximum", "mip"}
        preferred_setters = (
            "SetSlabReconstructionTypeToMax",
            "SetSlabReconstructionTypeToMaximum",
            "SetSlabReconstructionTypeToMIP",
        )
    elif mode_l in {"min", "minimum"}:
        target = "min"
        keywords = {"min", "minimum"}
        preferred_setters = (
            "SetSlabReconstructionTypeToMin",
            "SetSlabReconstructionTypeToMinimum",
        )
    elif mode_l in {"mean", "avg", "average"}:
        target = "mean"
        keywords = {"mean", "avg", "average"}
        preferred_setters = (
            "SetSlabReconstructionTypeToMean",
            "SetSlabReconstructionTypeToAverage",
        )
    else:
        raise ValueError("mode must be one of: mip/max, min, mean")

    # 1) Try dedicated convenience setters if available.
    applied = False
    for fn in preferred_setters:
        if hasattr(sn, fn):
            try:
                getattr(sn, fn)()
                applied = True
                break
            except Exception:
                pass

    # 2) Resolve enum mapping dynamically via GetSlabReconstructionTypeAsString(int)
    #    (mapping differs across VTK wrapper builds).
    if not applied and hasattr(sn, "SetSlabReconstructionType"):
        code = _slab_type_code_for(sn, target)
        if code is not None:
            try:
                sn.SetSlabReconstructionType(int(code))
                applied = True
            except Exception:
                applied = False

    # 3) Last resort: brute-force a small enum range and keep the first match.
    if not applied and hasattr(sn, "SetSlabReconstructionType"):
        for code in range(0, 8):
            try:
                sn.SetSlabReconstructionType(int(code))
                s = (_safe_slab_type_as_string(sn) or "").lower()
                if any(k in s for k in keywords):
                    applied = True
                    break
            except Exception:
                continue

    if not applied:
        raise RuntimeError(f"Unable to set slab reconstruction type for mode={mode_l!r} on this Slicer build")

    slicer.app.processEvents()

    out: Dict[str, Any] = {
        "view": _public_view(view),
        "enabled": bool(sn.GetSlabReconstructionEnabled()) if hasattr(sn, "GetSlabReconstructionEnabled") else bool(enabled),
        "thickness_mm": float(sn.GetSlabReconstructionThickness()) if hasattr(sn, "GetSlabReconstructionThickness") else float(thickness_mm),
    }
    slab_mode_str = _safe_slab_type_as_string(sn)
    if slab_mode_str is not None:
        out["mode"] = slab_mode_str
    elif hasattr(sn, "GetSlabReconstructionType"):
        out["mode"] = int(sn.GetSlabReconstructionType())
    else:
        out["mode"] = mode_l
    return out


def _set_fusion(*, background_volume_id: str, foreground_volume_id: str, opacity: float = 0.5) -> Dict[str, Any]:
    bg = slicer.mrmlScene.GetNodeByID(background_volume_id)
    fg = slicer.mrmlScene.GetNodeByID(foreground_volume_id)
    if bg is None:
        raise ValueError(f"Background volume not found: {background_volume_id}")
    if fg is None:
        raise ValueError(f"Foreground volume not found: {foreground_volume_id}")
    slicer.util.setSliceViewerLayers(background=bg, foreground=fg, foregroundOpacity=float(opacity))
    slicer.app.processEvents()
    return {"background_volume_id": bg.GetID(), "foreground_volume_id": fg.GetID(), "opacity": float(opacity)}


def _clear_fusion() -> Dict[str, Any]:
    results: Dict[str, Any] = {"views": {}}
    for slicer_view, public_view in _SLICE_PUBLIC_VIEWS:
        comp = _slice_composite_node(slicer_view)
        if comp is None:
            results["views"][public_view] = {"error": "slice composite node missing"}
            continue
        try:
            if hasattr(comp, "SetForegroundVolumeID"):
                try:
                    comp.SetForegroundVolumeID(None)
                except Exception:
                    comp.SetForegroundVolumeID("")
            if hasattr(comp, "SetForegroundOpacity"):
                comp.SetForegroundOpacity(0.0)
            results["views"][public_view] = {
                "background_volume_id": str(comp.GetBackgroundVolumeID()) if hasattr(comp, "GetBackgroundVolumeID") else None,
                "foreground_volume_id": str(comp.GetForegroundVolumeID()) if hasattr(comp, "GetForegroundVolumeID") else None,
                "foreground_opacity": float(comp.GetForegroundOpacity()) if hasattr(comp, "GetForegroundOpacity") else None,
            }
        except Exception as e:
            results["views"][public_view] = {"error": str(e)}
    slicer.app.processEvents()
    return results


def _compute_subtraction(*, volume_a_id: str, volume_b_id: str, output_name: str = "Subtraction") -> Dict[str, Any]:
    va = slicer.mrmlScene.GetNodeByID(volume_a_id)
    vb = slicer.mrmlScene.GetNodeByID(volume_b_id)
    if va is None or vb is None:
        raise ValueError("volume_a_id and volume_b_id must exist in the scene.")
    if not va.IsA("vtkMRMLScalarVolumeNode") or not vb.IsA("vtkMRMLScalarVolumeNode"):
        raise ValueError("Subtraction currently supports scalar volumes only.")

    import numpy as np

    a = slicer.util.arrayFromVolume(va).astype(np.float32, copy=False)
    b = slicer.util.arrayFromVolume(vb).astype(np.float32, copy=False)
    if a.shape != b.shape:
        raise ValueError(f"Volume shapes differ: {a.shape} vs {b.shape} (must be resampled first).")

    diff = a - b

    # Clone geometry from A for determinism.
    out = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, va, str(output_name))
    slicer.util.updateVolumeFromArray(out, diff)

    dn = _volume_display_node(out)
    if dn is not None:
        dn.SetAutoWindowLevel(True)

    _set_active_volume(out)
    slicer.app.processEvents()
    return {"output_volume_id": out.GetID(), "output_volume_name": out.GetName()}


def _ras_to_ijk_float(volume_node, ras: Sequence[float]) -> Tuple[float, float, float]:
    if len(ras) != 3:
        raise ValueError("ras must be length-3 [x,y,z]")
    import vtk

    m = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m)
    x, y, z = [float(v) for v in ras]
    out = [0.0, 0.0, 0.0, 0.0]
    m.MultiplyPoint([x, y, z, 1.0], out)
    return float(out[0]), float(out[1]), float(out[2])


def _roi_stats_ijk(
    volume_id: Optional[str],
    volume_name: Optional[str],
    ijk_min: List[int],
    ijk_max: List[int],
) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name)
    if not node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
        raise ValueError(f"Node is not a scalar volume: {node.GetClassName()}")

    if len(ijk_min) != 3 or len(ijk_max) != 3:
        raise ValueError("ijk_min and ijk_max must be length-3 arrays [i,j,k]")

    i0, j0, k0 = [int(x) for x in ijk_min]
    i1, j1, k1 = [int(x) for x in ijk_max]

    dims = node.GetImageData().GetDimensions()  # (i, j, k)
    i0 = max(0, min(i0, dims[0]))
    i1 = max(0, min(i1, dims[0]))
    j0 = max(0, min(j0, dims[1]))
    j1 = max(0, min(j1, dims[1]))
    k0 = max(0, min(k0, dims[2]))
    k1 = max(0, min(k1, dims[2]))

    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        raise ValueError(f"Empty ROI after clamping. ijk_min={ijk_min} ijk_max={ijk_max} dims={dims}")

    import numpy as np

    arr = slicer.util.arrayFromVolume(node)  # shape: (k, j, i)
    roi = arr[k0:k1, j0:j1, i0:i1]
    roi_f = roi.astype(np.float64, copy=False)

    return {
        "volume_id": node.GetID(),
        "volume_name": node.GetName(),
        "dims_ijk": list(dims),
        "ijk_min": [i0, j0, k0],
        "ijk_max": [i1, j1, k1],
        "n_voxels": int(roi_f.size),
        "min": float(np.min(roi_f)),
        "max": float(np.max(roi_f)),
        "mean": float(np.mean(roi_f)),
        "std": float(np.std(roi_f)),
    }


def _roi_stats_ras_box(
    *,
    volume_id: Optional[str],
    volume_name: Optional[str],
    ras_min: Sequence[float],
    ras_max: Sequence[float],
) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if node is None:
        raise RuntimeError("No active volume for ROI stats.")
    if not node.IsA("vtkMRMLScalarVolumeNode"):
        raise ValueError("roi_stats_ras_box currently supports scalar volumes only.")

    if len(ras_min) != 3 or len(ras_max) != 3:
        raise ValueError("ras_min and ras_max must be length-3 [x,y,z]")

    xmin, ymin, zmin = [float(v) for v in ras_min]
    xmax, ymax, zmax = [float(v) for v in ras_max]
    # 8 corners in RAS
    corners = [
        (xmin, ymin, zmin),
        (xmin, ymin, zmax),
        (xmin, ymax, zmin),
        (xmin, ymax, zmax),
        (xmax, ymin, zmin),
        (xmax, ymin, zmax),
        (xmax, ymax, zmin),
        (xmax, ymax, zmax),
    ]
    ijk = [_ras_to_ijk_float(node, c) for c in corners]
    is_ = [p[0] for p in ijk]
    js_ = [p[1] for p in ijk]
    ks_ = [p[2] for p in ijk]

    i0 = int(math.floor(min(is_)))
    j0 = int(math.floor(min(js_)))
    k0 = int(math.floor(min(ks_)))
    i1 = int(math.floor(max(is_))) + 1
    j1 = int(math.floor(max(js_))) + 1
    k1 = int(math.floor(max(ks_))) + 1

    return _roi_stats_ijk(volume_id=node.GetID(), volume_name=None, ijk_min=[i0, j0, k0], ijk_max=[i1, j1, k1]) | {
        "ras_min": [xmin, ymin, zmin],
        "ras_max": [xmax, ymax, zmax],
        "note": "ROI is computed as the axis-aligned IJK bounding box of the provided RAS box corners.",
    }


def _sample_intensity_ras(
    *,
    volume_id: Optional[str],
    volume_name: Optional[str],
    ras: Sequence[float],
    method: str = "nearest",
) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name) if (volume_id or volume_name) else _get_active_volume_node()
    if node is None:
        raise RuntimeError("No active volume for sampling.")
    if not node.IsA("vtkMRMLScalarVolumeNode"):
        raise ValueError("sample_intensity_ras currently supports scalar volumes only.")

    import numpy as np

    i_f, j_f, k_f = _ras_to_ijk_float(node, ras)
    arr = slicer.util.arrayFromVolume(node)  # (k,j,i)
    kz, jy, ix = arr.shape  # k, j, i

    m = (method or "").strip().lower()

    def _in_bounds(i: int, j: int, k: int) -> bool:
        return (0 <= i < ix) and (0 <= j < jy) and (0 <= k < kz)

    if m == "nearest":
        i = int(round(i_f))
        j = int(round(j_f))
        k = int(round(k_f))
        if not _in_bounds(i, j, k):
            return {"in_bounds": False, "value": None, "ijk": [i, j, k], "ras": [float(x) for x in ras]}
        return {"in_bounds": True, "value": float(arr[k, j, i]), "ijk": [i, j, k], "ras": [float(x) for x in ras]}

    if m in {"trilinear", "linear"}:
        i0 = int(math.floor(i_f))
        j0 = int(math.floor(j_f))
        k0 = int(math.floor(k_f))
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        if not (_in_bounds(i0, j0, k0) and _in_bounds(i1, j1, k1)):
            return {"in_bounds": False, "value": None, "ijk_float": [i_f, j_f, k_f], "ras": [float(x) for x in ras]}

        di = i_f - i0
        dj = j_f - j0
        dk = k_f - k0

        # 8-corner trilinear interpolation
        v000 = float(arr[k0, j0, i0])
        v100 = float(arr[k0, j0, i1])
        v010 = float(arr[k0, j1, i0])
        v110 = float(arr[k0, j1, i1])
        v001 = float(arr[k1, j0, i0])
        v101 = float(arr[k1, j0, i1])
        v011 = float(arr[k1, j1, i0])
        v111 = float(arr[k1, j1, i1])

        v00 = v000 * (1 - di) + v100 * di
        v01 = v001 * (1 - di) + v101 * di
        v10 = v010 * (1 - di) + v110 * di
        v11 = v011 * (1 - di) + v111 * di

        v0 = v00 * (1 - dj) + v10 * dj
        v1 = v01 * (1 - dj) + v11 * dj

        v = v0 * (1 - dk) + v1 * dk

        return {"in_bounds": True, "value": float(v), "ijk_float": [i_f, j_f, k_f], "ras": [float(x) for x in ras]}

    raise ValueError("method must be one of: nearest, trilinear")


def _measure_distance_ras(*, p1: Sequence[float], p2: Sequence[float]) -> Dict[str, Any]:
    if len(p1) != 3 or len(p2) != 3:
        raise ValueError("p1 and p2 must be length-3 [x,y,z] in RAS (mm)")
    x1, y1, z1 = [float(v) for v in p1]
    x2, y2, z2 = [float(v) for v in p2]
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return {"p1": [x1, y1, z1], "p2": [x2, y2, z2], "distance_mm": float(d)}


def _measure_angle_ras(*, p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> Dict[str, Any]:
    if len(p1) != 3 or len(p2) != 3 or len(p3) != 3:
        raise ValueError("p1,p2,p3 must be length-3 [x,y,z] in RAS (mm)")
    import numpy as np

    a = np.array([float(v) for v in p1], dtype=np.float64)
    b = np.array([float(v) for v in p2], dtype=np.float64)
    c = np.array([float(v) for v in p3], dtype=np.float64)

    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("Degenerate angle: two points coincide with vertex.")
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))
    return {"p1": a.tolist(), "p2": b.tolist(), "p3": c.tolist(), "angle_deg": float(ang)}


def _measure_area_polygon_ras(*, points: Sequence[Sequence[float]]) -> Dict[str, Any]:
    if len(points) < 3:
        raise ValueError("Need at least 3 points for polygon area.")
    import numpy as np

    pts = np.array([[float(v) for v in p] for p in points], dtype=np.float64)
    p0 = pts[0]
    # Define plane from first 3 points
    v1 = pts[1] - p0
    v2 = pts[2] - p0
    n = np.cross(v1, v2)
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        raise ValueError("Degenerate polygon: first three points are collinear.")
    n = n / nn
    xaxis = v1 / float(np.linalg.norm(v1))
    yaxis = np.cross(n, xaxis)

    # Project to 2D
    xy = np.stack([np.dot(pts - p0, xaxis), np.dot(pts - p0, yaxis)], axis=1)
    x = xy[:, 0]
    y = xy[:, 1]
    # Shoelace formula
    area = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return {"n_points": int(len(points)), "area_mm2": float(area)}


# -----------------------------------------------------------------------------
# Existing tools (scene/DICOM/export)
# -----------------------------------------------------------------------------


def _clear_scene() -> Dict[str, Any]:
    reset = _reset_interaction_state()
    slicer.mrmlScene.Clear()
    slicer.app.processEvents()
    return {"cleared": True, "reset_interaction": reset}


def _load_dicom(
    dicom_dir: str,
    clear_scene_first: bool = True,
    active_prefer: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    dicom_dir = os.path.expanduser(dicom_dir)
    dicom_dir = os.path.abspath(dicom_dir)
    if not os.path.isdir(dicom_dir):
        raise ValueError(f"dicom_dir is not a directory: {dicom_dir}")

    return _load_dicom_series(
        dicom_dirs=[dicom_dir],
        clear_scene_first=clear_scene_first,
        active_prefer=active_prefer,
    )


def _load_dicom_series(
    *,
    dicom_dirs: Sequence[str],
    clear_scene_first: bool = True,
    active_prefer: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    series_dirs: List[str] = []
    for d in dicom_dirs or []:
        dp = os.path.abspath(os.path.expanduser(str(d)))
        if os.path.isdir(dp):
            series_dirs.append(dp)

    if not series_dirs:
        raise ValueError("No valid dicom_dirs provided")

    if clear_scene_first:
        _reset_interaction_state()
        slicer.mrmlScene.Clear()

    from DICOMLib import DICOMUtils

    loaded_node_ids: List[str] = []
    patient_uids: List[str] = []
    imported_dirs: List[str] = []
    skipped_dirs: List[Dict[str, Any]] = []
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        for d in series_dirs:
            try:
                DICOMUtils.importDicom(d, db)
                imported_dirs.append(d)
            except Exception as e:
                skipped_dirs.append({"dir": d, "reason": f"import_failed: {e}"})

        patient_uids = list(db.patients())
        for patient_uid in patient_uids:
            try:
                loaded_node_ids.extend(DICOMUtils.loadPatientByUID(patient_uid))
            except Exception as e:
                skipped_dirs.append({"dir": str(patient_uid), "reason": f"load_patient_failed: {e}"})

    volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    active_volume_id = None
    active_volume_name = None
    if volumes:
        loaded_items: List[Dict[str, Any]] = []
        loaded_set = set(loaded_node_ids)
        for v in volumes:
            if v.GetID() in loaded_set:
                loaded_items.append({"id": v.GetID(), "name": v.GetName(), "file": v.GetName()})

        pick = _select_active_by_preference(loaded_items if loaded_items else [{"id": v.GetID(), "name": v.GetName(), "file": v.GetName()} for v in volumes], prefer=active_prefer)
        if pick is not None:
            volume_node = slicer.mrmlScene.GetNodeByID(str(pick.get("id")))
            if volume_node is not None:
                _set_active_volume(volume_node)
                active_volume_id = volume_node.GetID()
                active_volume_name = volume_node.GetName()

    slicer.app.processEvents()

    return {
        "loaded_node_ids": loaded_node_ids,
        "patient_uids": patient_uids,
        "active_volume_id": active_volume_id,
        "active_volume_name": active_volume_name,
        "n_scalar_volumes": len(volumes),
        "imported_dirs": imported_dirs,
        "skipped_dirs": skipped_dirs,
    }


def _load_nifti(
    *,
    nifti_dir: Optional[str] = None,
    nifti_files: Optional[Sequence[str]] = None,
    clear_scene_first: bool = True,
    active_prefer: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load NIfTI volumes from a directory or an explicit file list.

    This tool is intentionally conservative:
      - Only loads paths ending with .nii or .nii.gz
      - Ignores missing paths (records them as skipped)

    Selection of the active volume can be guided by `active_prefer`, a list of
    substrings (earlier = higher priority) matched against filename/name.
    """

    files: List[str] = []
    if nifti_files:
        files = [str(x) for x in nifti_files]
    elif nifti_dir:
        d = os.path.abspath(os.path.expanduser(str(nifti_dir)))
        if not os.path.isdir(d):
            raise ValueError(f"nifti_dir is not a directory: {d}")
        # Deterministic ordering.
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if os.path.isfile(p) and _is_nifti_path(p):
                files.append(p)
    else:
        raise ValueError("Must provide nifti_dir or nifti_files")

    if clear_scene_first:
        _reset_interaction_state()
        slicer.mrmlScene.Clear()

    loaded: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for f in files:
        fp = os.path.abspath(os.path.expanduser(str(f)))
        if not os.path.isfile(fp):
            skipped.append({"file": fp, "reason": "not_found"})
            continue
        if not _is_nifti_path(fp):
            skipped.append({"file": fp, "reason": "not_nifti"})
            continue

        name = os.path.splitext(os.path.basename(fp))[0]
        if name.endswith(".nii"):
            name = name[:-4]

        success, node = _load_volume_file(fp, name=name)
        if not success or node is None:
            skipped.append({"file": fp, "reason": "load_failed"})
            continue

        loaded.append({"file": fp, "id": node.GetID(), "name": node.GetName()})

    active_volume_id = None
    active_volume_name = None
    pick = _select_active_by_preference(loaded, prefer=active_prefer)
    if pick is not None:
        node = slicer.mrmlScene.GetNodeByID(str(pick.get("id")))
        if node is not None:
            _set_active_volume(node)
            active_volume_id = node.GetID()
            active_volume_name = node.GetName()

    slicer.app.processEvents()

    return {
        "loaded": loaded,
        "skipped": skipped,
        "active_volume_id": active_volume_id,
        "active_volume_name": active_volume_name,
        "n_loaded": len(loaded),
    }


def _select_volume(volume_id: Optional[str] = None, volume_name: Optional[str] = None) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name)
    if not node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
        raise ValueError(f"Node is not a volume: id={node.GetID()} class={node.GetClassName()} name={node.GetName()}")
    _set_active_volume(node)
    slicer.app.processEvents()
    return {"active_volume_id": node.GetID(), "active_volume_name": node.GetName()}


# -----------------------------------------------------------------------------
# Segmentation helpers / curated Segment Editor wrappers
# -----------------------------------------------------------------------------


def _list_segmentation_nodes() -> List[Any]:
    try:
        return list(slicer.util.getNodesByClass("vtkMRMLSegmentationNode"))
    except Exception:
        return []


def _get_segmentation_node(*, segmentation_id: Optional[str] = None, segmentation_name: Optional[str] = None):
    if segmentation_id or segmentation_name:
        node = _get_node_by_id_or_name(node_id=segmentation_id, node_name=segmentation_name)
        if not node.IsA("vtkMRMLSegmentationNode"):
            raise ValueError(f"Node is not a segmentation: id={node.GetID()} class={node.GetClassName()} name={node.GetName()}")
        return node

    nodes = _list_segmentation_nodes()
    if len(nodes) == 1:
        return nodes[0]
    if not nodes:
        raise ValueError("No segmentation nodes are loaded.")
    raise ValueError("Multiple segmentations are loaded; specify segmentation_id or segmentation_name.")


def _iter_segment_ids(segmentation_node) -> List[str]:
    segmentation = segmentation_node.GetSegmentation()
    ids: List[str] = []
    for i in range(int(segmentation.GetNumberOfSegments())):
        try:
            segment_id = segmentation.GetNthSegmentID(i)
        except Exception:
            segment_id = None
        if segment_id:
            ids.append(str(segment_id))
    return ids


def _find_segment_id(segmentation_node, *, segment_id: Optional[str] = None, segment_name: Optional[str] = None) -> Optional[str]:
    segmentation = segmentation_node.GetSegmentation()
    if segment_id:
        try:
            if segmentation.GetSegment(str(segment_id)) is not None:
                return str(segment_id)
        except Exception:
            pass
    if segment_name:
        try:
            sid = segmentation.GetSegmentIdBySegmentName(str(segment_name))
            if sid:
                return str(sid)
        except Exception:
            pass
        want = str(segment_name).strip().lower()
        for sid in _iter_segment_ids(segmentation_node):
            segment = segmentation.GetSegment(sid)
            if segment is None:
                continue
            if want == str(segment.GetName()).strip().lower():
                return str(sid)
        for sid in _iter_segment_ids(segmentation_node):
            segment = segmentation.GetSegment(sid)
            if segment is None:
                continue
            if want in str(segment.GetName()).strip().lower():
                return str(sid)
    return None


def _resolve_source_volume_node(*, volume_id: Optional[str] = None, volume_name: Optional[str] = None):
    node = None
    if volume_id or volume_name:
        node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name)
    else:
        node = _get_active_volume_node()
    if node is None:
        raise ValueError("No source volume is available.")
    if not node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
        raise ValueError(f"Source node is not a volume: {node.GetClassName()}")
    return node


def _ensure_segmentation_node(
    *,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    source_volume_node=None,
    create_if_missing: bool = False,
):
    try:
        node = _get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)
        created = False
    except Exception:
        if not create_if_missing:
            raise
        name = str(segmentation_name or "Segmentation")
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
        node.CreateDefaultDisplayNodes()
        created = True

    if source_volume_node is not None and hasattr(node, "SetReferenceImageGeometryParameterFromVolumeNode"):
        node.SetReferenceImageGeometryParameterFromVolumeNode(source_volume_node)

    display_node = node.GetDisplayNode()
    if display_node is not None:
        if hasattr(display_node, "SetVisibility2D"):
            display_node.SetVisibility2D(True)
        if hasattr(display_node, "SetVisibility3D"):
            display_node.SetVisibility3D(True)
        if hasattr(display_node, "SetOpacity"):
            display_node.SetOpacity(1.0)

    return node, created


def _ensure_segment(segmentation_node, *, segment_id: Optional[str] = None, segment_name: Optional[str] = None, create_if_missing: bool = False):
    segmentation = segmentation_node.GetSegmentation()
    sid = _find_segment_id(segmentation_node, segment_id=segment_id, segment_name=segment_name)
    if sid:
        return sid, False
    if not create_if_missing:
        raise ValueError(f"Segment not found in segmentation {segmentation_node.GetName()}: id={segment_id!r} name={segment_name!r}")

    name = str(segment_name or "Segment")
    new_id = None
    if hasattr(segmentation, "AddEmptySegment"):
        try:
            new_id = segmentation.AddEmptySegment("", name)
        except Exception:
            try:
                new_id = segmentation.AddEmptySegment(name)
            except Exception:
                new_id = None
    if not new_id:
        try:
            import vtkSegmentationCorePython as vtkSegmentationCore  # type: ignore

            segment = vtkSegmentationCore.vtkSegment()
            segment.SetName(name)
            segmentation.AddSegment(segment)
            new_id = _find_segment_id(segmentation_node, segment_name=name)
        except Exception as e:
            raise RuntimeError(f"Unable to create new segment {name!r}: {e}")
    return str(new_id), True


def _segment_summary(segmentation_node, segment_id: str) -> Dict[str, Any]:
    segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
    if segment is None:
        raise ValueError(f"Segment not found: {segment_id}")
    display_node = segmentation_node.GetDisplayNode()
    visible = None
    try:
        if display_node is not None and hasattr(display_node, "GetSegmentVisibility"):
            visible = bool(display_node.GetSegmentVisibility(segment_id))
    except Exception:
        visible = None
    return {
        "segment_id": str(segment_id),
        "segment_name": str(segment.GetName()),
        "color": list(segment.GetColor()) if hasattr(segment, "GetColor") else None,
        "visible": visible,
    }


def _segmentation_summary(segmentation_node) -> Dict[str, Any]:
    seg_ids = _iter_segment_ids(segmentation_node)
    return {
        "id": segmentation_node.GetID(),
        "name": segmentation_node.GetName(),
        "n_segments": len(seg_ids),
        "segments": [_segment_summary(segmentation_node, sid) for sid in seg_ids[:50]],
    }


def _list_segmentations() -> Dict[str, Any]:
    nodes = _list_segmentation_nodes()
    segs = [_segmentation_summary(node) for node in nodes]
    active_seg = segs[0] if len(segs) == 1 else None
    return {
        "segmentations": segs,
        "active_segmentation": active_seg,
        "n_segmentations": len(segs),
    }


def _list_segments(*, segmentation_id: Optional[str] = None, segmentation_name: Optional[str] = None) -> Dict[str, Any]:
    segmentation_node = _get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)
    seg_ids = _iter_segment_ids(segmentation_node)
    return {
        "segmentation_id": segmentation_node.GetID(),
        "segmentation_name": segmentation_node.GetName(),
        "segments": [_segment_summary(segmentation_node, sid) for sid in seg_ids],
        "n_segments": len(seg_ids),
    }


def _create_segment_editor(segmentation_node, source_volume_node=None):
    try:
        import SegmentEditorEffects  # noqa: F401
    except Exception:
        pass
    widget = slicer.qMRMLSegmentEditorWidget()
    widget.setMRMLScene(slicer.mrmlScene)
    editor_node = slicer.vtkMRMLSegmentEditorNode()
    slicer.mrmlScene.AddNode(editor_node)
    if hasattr(editor_node, "SetOverwriteMode") and hasattr(slicer.vtkMRMLSegmentEditorNode, "OverwriteNone"):
        editor_node.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
    widget.setMRMLSegmentEditorNode(editor_node)
    widget.setSegmentationNode(segmentation_node)
    if source_volume_node is not None:
        if hasattr(widget, "setSourceVolumeNode"):
            widget.setSourceVolumeNode(source_volume_node)
        elif hasattr(widget, "setMasterVolumeNode"):
            widget.setMasterVolumeNode(source_volume_node)
    return widget, editor_node


def _cleanup_segment_editor(widget, editor_node) -> None:
    try:
        if hasattr(widget, "setActiveEffectByName"):
            widget.setActiveEffectByName("")
    except Exception:
        pass
    try:
        if hasattr(widget, "setMRMLSegmentEditorNode"):
            widget.setMRMLSegmentEditorNode(None)
    except Exception:
        pass
    try:
        if hasattr(widget, "setSegmentationNode"):
            widget.setSegmentationNode(None)
    except Exception:
        pass
    try:
        if hasattr(widget, "setSourceVolumeNode"):
            widget.setSourceVolumeNode(None)
        elif hasattr(widget, "setMasterVolumeNode"):
            widget.setMasterVolumeNode(None)
    except Exception:
        pass
    try:
        if editor_node is not None:
            slicer.mrmlScene.RemoveNode(editor_node)
    except Exception:
        pass


def _set_selected_segment(widget, editor_node, segment_id: str) -> None:
    if hasattr(editor_node, "SetSelectedSegmentID"):
        editor_node.SetSelectedSegmentID(str(segment_id))
    if hasattr(widget, "setCurrentSegmentID"):
        try:
            widget.setCurrentSegmentID(str(segment_id))
        except Exception:
            pass
    slicer.app.processEvents()


def _apply_effect(widget, effect_name: str, params: Dict[str, Any]) -> None:
    widget.setActiveEffectByName(effect_name)
    effect = widget.activeEffect()
    if effect is None:
        raise RuntimeError(f"Segment Editor effect not available: {effect_name}")
    for key, value in params.items():
        if value is None:
            continue
        effect.setParameter(str(key), str(value))

    effect_impl = effect.self() if hasattr(effect, "self") else effect
    for fn_name in ("onApply", "apply"):
        fn = getattr(effect_impl, fn_name, None)
        if callable(fn):
            fn()
            slicer.app.processEvents()
            return
    raise RuntimeError(f"Could not apply Segment Editor effect {effect_name!r}: no onApply/apply method")


def _segment_basic_stats(segmentation_node, segment_id: str, source_volume_node=None) -> Dict[str, Any]:
    segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
    if segment is None:
        raise ValueError(f"Segment not found: {segment_id}")

    stats: Dict[str, Any] = {
        "segmentation_id": segmentation_node.GetID(),
        "segmentation_name": segmentation_node.GetName(),
        "segment_id": str(segment_id),
        "segment_name": segment.GetName(),
    }

    try:
        center_ras = segmentation_node.GetSegmentCenterRAS(segment_id)
        stats["center_ras"] = [float(x) for x in center_ras]
    except Exception:
        stats["center_ras"] = None

    if source_volume_node is None:
        return stats

    try:
        import numpy as np

        segment_array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, segment_id, source_volume_node)
        voxel_count = int(np.count_nonzero(segment_array))
        stats["voxel_count"] = voxel_count
        spacing = [float(x) for x in source_volume_node.GetSpacing()]
        voxel_volume_mm3 = abs(spacing[0] * spacing[1] * spacing[2])
        stats["volume_mm3"] = float(voxel_count * voxel_volume_mm3)
        stats["volume_cm3"] = float(stats["volume_mm3"] / 1000.0)

        if voxel_count > 0:
            volume_array = slicer.util.arrayFromVolume(source_volume_node)
            voxels = volume_array[segment_array > 0]
            stats["min"] = float(voxels.min())
            stats["max"] = float(voxels.max())
            stats["mean"] = float(voxels.mean())
            stats["median"] = float(np.median(voxels))
            stats["std"] = float(voxels.std())
    except Exception as e:
        stats["statistics_warning"] = str(e)

    stats["source_volume_id"] = source_volume_node.GetID()
    stats["source_volume_name"] = source_volume_node.GetName()
    return stats


def _segment_from_threshold(
    *,
    minimum_threshold: float,
    maximum_threshold: float,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=True,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=True,
    )

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(
            widget,
            "Threshold",
            {
                "MinimumThreshold": float(minimum_threshold),
                "MaximumThreshold": float(maximum_threshold),
            },
        )
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)

    slicer.app.processEvents()
    return {
        **_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node),
        "minimum_threshold": float(minimum_threshold),
        "maximum_threshold": float(maximum_threshold),
    }


def _slice_ras_from_normalized(*, view: str, x_norm: float, y_norm: float) -> List[float]:
    sn = _slice_node(view)
    dims = None
    if hasattr(sn, "GetDimensions"):
        try:
            dims = sn.GetDimensions()
        except Exception:
            dims = None
    if not dims or len(dims) < 2 or dims[0] <= 1 or dims[1] <= 1:
        try:
            slice_view = _slice_widget(view).sliceView()
            width_attr = getattr(slice_view, "width", None)
            height_attr = getattr(slice_view, "height", None)
            width = int(width_attr() if callable(width_attr) else width_attr)
            height = int(height_attr() if callable(height_attr) else height_attr)
            dims = (width, height, 1)
        except Exception:
            dims = (512, 512, 1)
    width = max(float(dims[0]), 1.0)
    height = max(float(dims[1]), 1.0)
    x = max(0.0, min(1.0, float(x_norm))) * width
    # Observed PNGs and normal image coordinates use top-left origin; VTK slice XY uses bottom-left.
    y = (1.0 - max(0.0, min(1.0, float(y_norm)))) * height
    m = sn.GetXYToRAS()
    out = [0.0, 0.0, 0.0, 0.0]
    m.MultiplyPoint([x, y, 0.0, 1.0], out)
    return [float(out[0]), float(out[1]), float(out[2])]




def _image_coords_to_ras(
    *,
    view: str,
    point_norm: Optional[Sequence[float]] = None,
    bbox_norm: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    public_view = _public_view(view)
    if (point_norm is None) == (bbox_norm is None):
        raise ValueError("Provide exactly one of point_norm or bbox_norm.")

    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    if point_norm is not None:
        if len(point_norm) != 2:
            raise ValueError("point_norm must be [x_norm, y_norm]")
        x_norm = _clamp01(float(point_norm[0]))
        y_norm = _clamp01(float(point_norm[1]))
        point_ras = _slice_ras_from_normalized(view=public_view, x_norm=x_norm, y_norm=y_norm)
        return {
            "ok": True,
            "view": public_view,
            "point_norm": [x_norm, y_norm],
            "point_ras": point_ras,
            "center_ras": point_ras,
        }

    if bbox_norm is None or len(bbox_norm) != 4:
        raise ValueError("bbox_norm must be [x_min_norm, y_min_norm, x_max_norm, y_max_norm]")
    x1, y1, x2, y2 = [_clamp01(float(v)) for v in bbox_norm]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    corners = {
        "top_left": _slice_ras_from_normalized(view=public_view, x_norm=x1, y_norm=y1),
        "top_right": _slice_ras_from_normalized(view=public_view, x_norm=x2, y_norm=y1),
        "bottom_right": _slice_ras_from_normalized(view=public_view, x_norm=x2, y_norm=y2),
        "bottom_left": _slice_ras_from_normalized(view=public_view, x_norm=x1, y_norm=y2),
    }
    center_ras = _slice_ras_from_normalized(view=public_view, x_norm=cx, y_norm=cy)
    return {
        "ok": True,
        "view": public_view,
        "bbox_norm": [x1, y1, x2, y2],
        "bbox_corners_ras": corners,
        "center_norm": [cx, cy],
        "center_ras": center_ras,
        "point_ras": center_ras,
    }



def _resolve_segment_seed_ras(
    *,
    seed_view: Optional[str] = None,
    seed_x_norm: Optional[float] = None,
    seed_y_norm: Optional[float] = None,
    ras_seed: Optional[Sequence[float]] = None,
    seed_segment_id: Optional[str] = None,
    seed_segment_name: Optional[str] = None,
    segmentation_node=None,
) -> Tuple[List[float], str]:
    if ras_seed is not None:
        if len(ras_seed) != 3:
            raise ValueError("ras_seed must be length-3 [x,y,z] in RAS mm")
        return [float(v) for v in ras_seed], "ras_seed"

    if seed_segment_id or seed_segment_name:
        if segmentation_node is None:
            raise ValueError("seed_segment_id/seed_segment_name requires a segmentation node")
        sid = _find_segment_id(segmentation_node, segment_id=seed_segment_id, segment_name=seed_segment_name)
        if not sid:
            raise ValueError(f"Seed segment not found: id={seed_segment_id!r} name={seed_segment_name!r}")
        center_ras = segmentation_node.GetSegmentCenterRAS(sid)
        return [float(v) for v in center_ras], "segment_center"

    if seed_view is not None and seed_x_norm is not None and seed_y_norm is not None:
        return _slice_ras_from_normalized(view=str(seed_view), x_norm=float(seed_x_norm), y_norm=float(seed_y_norm)), "view_normalized"

    raise ValueError(
        "Provide one seed source: ras_seed=[x,y,z], or seed_view + seed_x_norm + seed_y_norm, or seed_segment_id/seed_segment_name."
    )


def _segment_local_threshold(
    *,
    minimum_threshold: float,
    maximum_threshold: float,
    minimum_diameter_mm: float = 3.0,
    feature_size_mm: float = 3.0,
    segmentation_algorithm: str = "GrowCut",
    seed_view: Optional[str] = None,
    seed_x_norm: Optional[float] = None,
    seed_y_norm: Optional[float] = None,
    ras_seed: Optional[Sequence[float]] = None,
    seed_segment_id: Optional[str] = None,
    seed_segment_name: Optional[str] = None,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    import vtk

    allowed_algorithms = {"Masking", "GrowCut", "WaterShed"}
    algorithm = str(segmentation_algorithm or "GrowCut").strip()
    if algorithm not in allowed_algorithms:
        raise ValueError(f"Unsupported segmentation_algorithm: {segmentation_algorithm!r}. Allowed: {sorted(allowed_algorithms)}")

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=True,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=True,
    )

    seed_ras, seed_source = _resolve_segment_seed_ras(
        seed_view=seed_view,
        seed_x_norm=seed_x_norm,
        seed_y_norm=seed_y_norm,
        ras_seed=ras_seed,
        seed_segment_id=seed_segment_id,
        seed_segment_name=seed_segment_name,
        segmentation_node=segmentation_node,
    )
    seed_ijk = _ras_to_ijk_float(source_volume_node, seed_ras)
    ijk_points = vtk.vtkPoints()
    ijk_points.InsertNextPoint(float(seed_ijk[0]), float(seed_ijk[1]), float(seed_ijk[2]))

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        widget.setActiveEffectByName("Local Threshold")
        effect = widget.activeEffect()
        if effect is None:
            raise RuntimeError(
                "Segment Editor effect not available: Local Threshold. Install SegmentEditorExtraEffects and restart Slicer."
            )
        effect_name_attr = getattr(effect, "name", None)
        effect_name = effect_name_attr() if callable(effect_name_attr) else effect_name_attr
        if effect_name and "local threshold" not in str(effect_name).lower():
            raise RuntimeError(
                f"Active Segment Editor effect is {effect_name!r}, not 'Local Threshold'. Install SegmentEditorExtraEffects and restart Slicer."
            )
        effect.setParameter("MinimumThreshold", float(minimum_threshold))
        effect.setParameter("MaximumThreshold", float(maximum_threshold))
        effect.setParameter("MinimumDiameterMm", float(minimum_diameter_mm))
        effect.setParameter("FeatureSizeMm", float(feature_size_mm))
        effect.setParameter("SegmentationAlgorithm", algorithm)
        effect_impl = effect.self() if hasattr(effect, "self") else effect
        apply_fn = getattr(effect_impl, "apply", None)
        if not callable(apply_fn):
            raise RuntimeError("Local Threshold effect does not expose apply(ijkPoints) in this Slicer build.")
        apply_fn(ijk_points)
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)

    slicer.app.processEvents()
    return {
        **_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node),
        "minimum_threshold": float(minimum_threshold),
        "maximum_threshold": float(maximum_threshold),
        "minimum_diameter_mm": float(minimum_diameter_mm),
        "feature_size_mm": float(feature_size_mm),
        "segmentation_algorithm": algorithm,
        "seed_ras": [float(v) for v in seed_ras],
        "seed_ijk": [float(v) for v in seed_ijk],
        "seed_source": seed_source,
        "seed_view": str(seed_view) if seed_view is not None else None,
        "seed_x_norm": float(seed_x_norm) if seed_x_norm is not None else None,
        "seed_y_norm": float(seed_y_norm) if seed_y_norm is not None else None,
    }


def _segment_margin(
    *,
    margin_size_mm: float,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=False,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=False,
    )
    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(widget, "Margin", {"MarginSizeMm": float(margin_size_mm)})
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    return {**_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node), "margin_size_mm": float(margin_size_mm)}


def _segment_smoothing(
    *,
    smoothing_method: str = "MEDIAN",
    kernel_size_mm: float = 3.0,
    gaussian_std_mm: Optional[float] = None,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    allowed = {"MEDIAN", "GAUSSIAN", "MORPHOLOGICAL_CLOSING"}
    method = str(smoothing_method or "MEDIAN").strip().upper()
    if method not in allowed:
        raise ValueError(f"Unsupported smoothing_method: {smoothing_method!r}. Allowed: {sorted(allowed)}")

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=False,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=False,
    )
    params: Dict[str, Any] = {"SmoothingMethod": method, "KernelSizeMm": float(kernel_size_mm)}
    if method == "GAUSSIAN":
        params["GaussianStandardDeviationMm"] = float(gaussian_std_mm if gaussian_std_mm is not None else kernel_size_mm)

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(widget, "Smoothing", params)
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    return {**_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node), "smoothing_method": method, "kernel_size_mm": float(kernel_size_mm)}


def _segment_islands(
    *,
    operation: str = "KEEP_LARGEST_ISLAND",
    minimum_size: int = 1000,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    allowed = {"KEEP_LARGEST_ISLAND", "REMOVE_SMALL_ISLANDS"}
    op = str(operation or "KEEP_LARGEST_ISLAND").strip().upper()
    if op not in allowed:
        raise ValueError(f"Unsupported islands operation: {operation!r}. Allowed: {sorted(allowed)}")

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=False,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=False,
    )

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(widget, "Islands", {"Operation": op, "MinimumSize": int(minimum_size)})
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    return {**_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node), "operation": op, "minimum_size": int(minimum_size)}


def _segment_logical(
    *,
    operation: str,
    modifier_segment_id: Optional[str] = None,
    modifier_segment_name: Optional[str] = None,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    allowed = {"COPY", "UNION", "INTERSECT", "SUBTRACT", "INVERT", "CLEAR", "FILL"}
    op = str(operation or "").strip().upper()
    if op not in allowed:
        raise ValueError(f"Unsupported logical operation: {operation!r}. Allowed: {sorted(allowed)}")

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=False,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=(op in {"COPY", "FILL", "CLEAR", "INVERT"}),
    )
    params: Dict[str, Any] = {"Operation": op, "BypassMasking": 1}
    if op in {"COPY", "UNION", "INTERSECT", "SUBTRACT"}:
        modifier_id = _find_segment_id(segmentation_node, segment_id=modifier_segment_id, segment_name=modifier_segment_name)
        if not modifier_id:
            raise ValueError(f"Modifier segment not found: id={modifier_segment_id!r} name={modifier_segment_name!r}")
        params["ModifierSegmentID"] = modifier_id

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(widget, "Logical operators", params)
    finally:
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    return {**_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node), "operation": op, "modifier_segment_id": params.get("ModifierSegmentID")}


def _segment_edit_sphere(
    *,
    ras_center: Sequence[float],
    radius_mm: float,
    action: str = "add",
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    if len(ras_center) != 3:
        raise ValueError("ras_center must be length-3 [x,y,z] in RAS mm")
    op = str(action or "add").strip().lower()
    if op not in {"add", "erase"}:
        raise ValueError("action must be 'add' or 'erase'")

    import vtk

    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node, _ = _ensure_segmentation_node(
        segmentation_id=segmentation_id,
        segmentation_name=segmentation_name,
        source_volume_node=source_volume_node,
        create_if_missing=True,
    )
    target_segment_id, _ = _ensure_segment(
        segmentation_node,
        segment_id=segment_id,
        segment_name=segment_name,
        create_if_missing=True,
    )

    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(float(ras_center[0]), float(ras_center[1]), float(ras_center[2]))
    sphere.SetRadius(float(radius_mm))
    sphere.SetThetaResolution(24)
    sphere.SetPhiResolution(24)
    sphere.Update()

    temp_name = f"__tmp_sphere_{os.getpid()}"
    temp_segment_id = segmentation_node.AddSegmentFromClosedSurfaceRepresentation(sphere.GetOutput(), temp_name, [1.0, 0.0, 0.0])
    if not temp_segment_id:
        raise RuntimeError("Failed to create temporary sphere segment")

    widget = None
    editor_node = None
    try:
        widget, editor_node = _create_segment_editor(segmentation_node, source_volume_node)
        _set_selected_segment(widget, editor_node, target_segment_id)
        _apply_effect(
            widget,
            "Logical operators",
            {
                "Operation": "UNION" if op == "add" else "SUBTRACT",
                "ModifierSegmentID": str(temp_segment_id),
                "BypassMasking": 1,
            },
        )
    finally:
        try:
            segmentation_node.GetSegmentation().RemoveSegment(str(temp_segment_id))
        except Exception:
            pass
        if widget is not None:
            _cleanup_segment_editor(widget, editor_node)
    slicer.app.processEvents()
    return {
        **_segment_basic_stats(segmentation_node, target_segment_id, source_volume_node),
        "action": op,
        "ras_center": [float(x) for x in ras_center],
        "radius_mm": float(radius_mm),
    }


def _segment_statistics(
    *,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    segmentation_node = _get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)
    target_id = _find_segment_id(segmentation_node, segment_id=segment_id, segment_name=segment_name)
    segment_ids = [target_id] if target_id else _iter_segment_ids(segmentation_node)
    stats_list = [_segment_basic_stats(segmentation_node, sid, source_volume_node) for sid in segment_ids]
    result: Dict[str, Any] = {
        "segmentation_id": segmentation_node.GetID(),
        "segmentation_name": segmentation_node.GetName(),
        "source_volume_id": source_volume_node.GetID(),
        "source_volume_name": source_volume_node.GetName(),
        "segments": stats_list,
        "n_segments": len(stats_list),
    }
    if target_id and stats_list:
        result.update(stats_list[0])
    return result


def _center_on_segment(
    *,
    segmentation_id: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    segment_id: Optional[str] = None,
    segment_name: Optional[str] = None,
) -> Dict[str, Any]:
    segmentation_node = _get_segmentation_node(segmentation_id=segmentation_id, segmentation_name=segmentation_name)
    target_segment_id, _ = _ensure_segment(segmentation_node, segment_id=segment_id, segment_name=segment_name, create_if_missing=False)
    center_ras = segmentation_node.GetSegmentCenterRAS(target_segment_id)
    _jump_slices_to_location(float(center_ras[0]), float(center_ras[1]), float(center_ras[2]), True)
    slicer.app.processEvents()
    segment = segmentation_node.GetSegmentation().GetSegment(target_segment_id)
    return {
        "segmentation_id": segmentation_node.GetID(),
        "segmentation_name": segmentation_node.GetName(),
        "segment_id": str(target_segment_id),
        "segment_name": segment.GetName() if segment is not None else None,
        "center_ras": [float(x) for x in center_ras],
    }


def _resample_volume_isotropic(
    *,
    output_spacing_mm: float,
    source_volume_id: Optional[str] = None,
    source_volume_name: Optional[str] = None,
    output_name: Optional[str] = None,
    interpolation_type: str = "linear",
    make_active: bool = True,
) -> Dict[str, Any]:
    source_volume_node = _resolve_source_volume_node(volume_id=source_volume_id, volume_name=source_volume_name)
    spacing = float(output_spacing_mm)
    if spacing <= 0:
        raise ValueError("output_spacing_mm must be > 0")

    interpolation_map = {
        "linear": "linear",
        "nearest": "nn",
        "nearestneighbor": "nn",
        "nn": "nn",
        "bspline": "bspline",
        "cubic": "bspline",
        "windowedsinc": "ws",
        "ws": "ws",
    }
    interp_key = str(interpolation_type or "linear").strip().lower()
    cli_interp = interpolation_map.get(interp_key)
    if cli_interp is None:
        raise ValueError(f"Unsupported interpolation_type: {interpolation_type!r}")

    out_name = str(output_name or f"{source_volume_node.GetName()}_iso_{spacing:g}mm")
    out_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", out_name)
    params = {
        "InputVolume": source_volume_node.GetID(),
        "OutputVolume": out_node.GetID(),
        "outputPixelSpacing": f"{spacing},{spacing},{spacing}",
        "interpolationType": cli_interp,
    }
    cli_module = getattr(slicer.modules, "resamplescalarvolume", None)
    if cli_module is None:
        raise RuntimeError("Resample Scalar Volume module is unavailable in this Slicer build.")
    cli_node = slicer.cli.runSync(cli_module, None, params)
    status_string = cli_node.GetStatusString() if cli_node is not None and hasattr(cli_node, "GetStatusString") else None
    if cli_node is not None and hasattr(cli_node, "GetStatus") and hasattr(cli_node, "ErrorsMask"):
        if int(cli_node.GetStatus()) & int(cli_node.ErrorsMask):
            err_text = cli_node.GetErrorText() if hasattr(cli_node, "GetErrorText") else status_string
            try:
                slicer.mrmlScene.RemoveNode(out_node)
            except Exception:
                pass
            raise RuntimeError(f"Resample Scalar Volume failed: {err_text}")
    if make_active:
        _set_active_volume(out_node)
    out_spacing = [float(x) for x in out_node.GetSpacing()] if hasattr(out_node, "GetSpacing") else None
    slicer.app.processEvents()
    return {
        "source_volume_id": source_volume_node.GetID(),
        "source_volume_name": source_volume_node.GetName(),
        "output_volume_id": out_node.GetID(),
        "output_volume_name": out_node.GetName(),
        "output_spacing": out_spacing,
        "interpolation_type": interp_key,
        "status": status_string,
        "made_active": bool(make_active),
    }


def _export_segmentation_dicom(segmentation_id: str, reference_volume_id: str, output_folder: str) -> Dict[str, Any]:
    output_folder = os.path.expanduser(output_folder)
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    segmentation_node = slicer.mrmlScene.GetNodeByID(segmentation_id)
    if segmentation_node is None:
        raise ValueError(f"Segmentation node not found: {segmentation_id}")
    if not segmentation_node.IsA("vtkMRMLSegmentationNode"):
        raise ValueError(f"Node is not a segmentation: {segmentation_node.GetClassName()}")

    reference_volume_node = slicer.mrmlScene.GetNodeByID(reference_volume_id)
    if reference_volume_node is None:
        raise ValueError(f"Reference volume node not found: {reference_volume_id}")

    sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    reference_item = sh_node.GetItemByDataNode(reference_volume_node)
    if reference_item == 0:
        raise RuntimeError("Reference volume is not in subject hierarchy (unexpected).")

    study_item = sh_node.GetItemParent(reference_item)
    if study_item == 0:
        study_item = sh_node.GetSceneItemID()

    segmentation_item = sh_node.GetItemByDataNode(segmentation_node)
    if segmentation_item == 0:
        raise RuntimeError("Segmentation is not in subject hierarchy (unexpected).")

    sh_node.SetItemParent(segmentation_item, study_item)

    plugin_module = None
    try:
        import DICOMSegmentationPlugin as plugin_module  # type: ignore
    except Exception:
        plugin_module = None

    if plugin_module is None:
        dicom_plugins = getattr(slicer.modules, "dicomPlugins", None)
        if isinstance(dicom_plugins, dict):
            plugin_module = dicom_plugins.get("DICOMSegmentationPlugin")
        elif hasattr(dicom_plugins, "get"):
            try:
                plugin_module = dicom_plugins.get("DICOMSegmentationPlugin")
            except Exception:
                plugin_module = None
        elif dicom_plugins is not None and hasattr(dicom_plugins, "DICOMSegmentationPlugin"):
            plugin_module = getattr(dicom_plugins, "DICOMSegmentationPlugin")

    if plugin_module is None or not hasattr(plugin_module, "DICOMSegmentationPluginClass"):
        raise RuntimeError(
            "DICOMSegmentationPlugin is unavailable. Install Quantitative Reporting and restart Slicer before exporting DICOM SEG."
        )

    exporter = plugin_module.DICOMSegmentationPluginClass()
    exportables = exporter.examineForExport(segmentation_item)
    if not exportables:
        raise RuntimeError("No exportables found for segmentation (is it empty?)")

    for exp in exportables:
        exp.directory = output_folder

    exporter.export(exportables)

    written = []
    for fn in os.listdir(output_folder):
        if fn.lower().endswith(".dcm"):
            written.append(os.path.join(output_folder, fn))
    written.sort()

    return {"output_folder": output_folder, "files": written, "n_files": len(written)}


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------



def _ping_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bridge_version": BRIDGE_VERSION,
        "slicer_version": slicer.app.applicationVersion,
        "slicer_revision": slicer.app.revision,
    }


def _build_tool_handlers() -> Dict[str, Any]:
    return {
        # Scene / data
        "ping": _ping_handler,
        "clear_scene": lambda args: _clear_scene(),
        "load_dicom": lambda args: _load_dicom(
            dicom_dir=str(args["dicom_dir"]),
            clear_scene_first=bool(args.get("clear_scene_first", True)),
            active_prefer=args.get("active_prefer"),
        ),
        "load_dicom_series": lambda args: _load_dicom_series(
            dicom_dirs=args.get("dicom_dirs") or [],
            clear_scene_first=bool(args.get("clear_scene_first", True)),
            active_prefer=args.get("active_prefer"),
        ),
        "load_nifti": lambda args: _load_nifti(
            nifti_dir=args.get("nifti_dir"),
            nifti_files=args.get("nifti_files"),
            clear_scene_first=bool(args.get("clear_scene_first", True)),
            active_prefer=args.get("active_prefer"),
        ),
        "select_volume": lambda args: _select_volume(
            volume_id=args.get("volume_id"),
            volume_name=args.get("volume_name"),
        ),
        # Viewer actuation
        "get_viewer_state": lambda args: _get_viewer_state(),
        "set_window_level": lambda args: _set_window_level(
            window=args.get("window"),
            level=args.get("level"),
            auto=args.get("auto"),
            volume_id=args.get("volume_id"),
            volume_name=args.get("volume_name"),
        ),
        "set_interpolation": lambda args: _set_interpolation(
            interpolate=bool(args["interpolate"]),
            volume_id=args.get("volume_id"),
            volume_name=args.get("volume_name"),
        ),
        "set_slice_orientation": lambda args: _set_slice_orientation(
            view=str(args["view"]),
            orientation=str(args["orientation"]),
        ),
        "get_slice_offset_range": lambda args: _get_slice_offset_range(view=str(args["view"])),
        "set_slice_offset": lambda args: _set_slice_offset(
            view=str(args["view"]),
            offset=float(args["offset"]),
        ),
        "set_slice_scroll_to": lambda args: _set_slice_scroll_to(
            view=str(args["view"]),
            scroll_to=float(args["scroll_to"]),
        ),
        "fit_slice": lambda args: _fit_slice(view=str(args["view"])),
        "zoom_slice_relative": lambda args: _zoom_slice_relative(
            view=str(args["view"]),
            factor=float(args["factor"]),
        ),
        "set_field_of_view": lambda args: _set_field_of_view(
            view=str(args["view"]),
            field_of_view=args["field_of_view"],
        ),
        "set_xyz_origin": lambda args: _set_xyz_origin(
            view=str(args["view"]),
            xyz_origin=args["xyz_origin"],
        ),
        "jump_to_ras": lambda args: _jump_to_ras(
            ras=args["ras"],
            centered=bool(args.get("centered", True)),
        ),
        "capture_slice_view_png": lambda args: _capture_slice_view_png(
            view=str(args.get("view", "axial")),
            output_path=str(args["output_path"]),
            include_controller=bool(args.get("include_controller", False)),
        ),
        "image_coords_to_ras": lambda args: _image_coords_to_ras(
            view=str(args["view"]),
            point_norm=args.get("point_norm"),
            bbox_norm=args.get("bbox_norm"),
        ),
        "recover_standard_views": lambda args: _recover_standard_views(
            volume_id=args.get("volume_id"),
            volume_name=args.get("volume_name"),
            centered=bool(args.get("centered", True)),
            fit=bool(args.get("fit", True)),
        ),
        "set_linked_slices": lambda args: _set_linked_slices(enabled=bool(args["enabled"])),
        "set_layout": lambda args: _set_layout(layout=str(args["layout"])),
        # Deterministic quantification / derived volumes
        "roi_stats_ijk": lambda args: _roi_stats_ijk(
            volume_id=args.get("volume_id"),
            volume_name=args.get("volume_name"),
            ijk_min=args.get("ijk_min"),
            ijk_max=args.get("ijk_max"),
        ),
        "roi_stats_ras_box": lambda args: _roi_stats_ras_box(
            volume_id=args.get("volume_id"),
            volume_name=args.get("volume_name"),
            ras_min=args.get("ras_min"),
            ras_max=args.get("ras_max"),
        ),
        "sample_intensity_ras": lambda args: _sample_intensity_ras(
            volume_id=args.get("volume_id"),
            volume_name=args.get("volume_name"),
            ras=args.get("ras"),
            method=str(args.get("method", "nearest")),
        ),
        "measure_distance_ras": lambda args: _measure_distance_ras(
            p1=args.get("p1"),
            p2=args.get("p2"),
        ),
        "measure_angle_ras": lambda args: _measure_angle_ras(
            p1=args.get("p1"),
            p2=args.get("p2"),
            p3=args.get("p3"),
        ),
        "measure_area_polygon_ras": lambda args: _measure_area_polygon_ras(points=args.get("points")),
        "set_thick_slab": lambda args: _set_thick_slab(
            view=str(args["view"]),
            enabled=bool(args.get("enabled", True)),
            thickness_mm=float(args.get("thickness_mm", 0.0)),
            mode=str(args.get("mode", "mip")),
        ),
        "set_fusion": lambda args: _set_fusion(
            background_volume_id=str(args["background_volume_id"]),
            foreground_volume_id=str(args["foreground_volume_id"]),
            opacity=float(args.get("opacity", 0.5)),
        ),
        "clear_fusion": lambda args: _clear_fusion(),
        "compute_subtraction": lambda args: _compute_subtraction(
            volume_a_id=str(args["volume_a_id"]),
            volume_b_id=str(args["volume_b_id"]),
            output_name=str(args.get("output_name", "Subtraction")),
        ),
        # Curated segmentation pack
        "list_segmentations": lambda args: _list_segmentations(),
        "list_segments": lambda args: _list_segments(
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
        ),
        "segment_from_threshold": lambda args: _segment_from_threshold(
            minimum_threshold=float(args["minimum_threshold"]),
            maximum_threshold=float(args["maximum_threshold"]),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        ),
        "segment_local_threshold": lambda args: _segment_local_threshold(
            minimum_threshold=float(args["minimum_threshold"]),
            maximum_threshold=float(args["maximum_threshold"]),
            minimum_diameter_mm=float(args.get("minimum_diameter_mm", 3.0)),
            feature_size_mm=float(args.get("feature_size_mm", 3.0)),
            segmentation_algorithm=str(args.get("segmentation_algorithm", "GrowCut")),
            seed_view=args.get("seed_view"),
            seed_x_norm=(float(args["seed_x_norm"]) if args.get("seed_x_norm") is not None else None),
            seed_y_norm=(float(args["seed_y_norm"]) if args.get("seed_y_norm") is not None else None),
            ras_seed=args.get("ras_seed"),
            seed_segment_id=args.get("seed_segment_id"),
            seed_segment_name=args.get("seed_segment_name"),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        ),
        "segment_edit_sphere": lambda args: _segment_edit_sphere(
            ras_center=args.get("ras_center"),
            radius_mm=float(args["radius_mm"]),
            action=str(args.get("action", "add")),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        ),
        "segment_margin": lambda args: _segment_margin(
            margin_size_mm=float(args["margin_size_mm"]),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        ),
        "segment_smoothing": lambda args: _segment_smoothing(
            smoothing_method=str(args.get("smoothing_method", "MEDIAN")),
            kernel_size_mm=float(args.get("kernel_size_mm", 3.0)),
            gaussian_std_mm=(float(args["gaussian_std_mm"]) if args.get("gaussian_std_mm") is not None else None),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        ),
        "segment_islands": lambda args: _segment_islands(
            operation=str(args.get("operation", "KEEP_LARGEST_ISLAND")),
            minimum_size=int(args.get("minimum_size", 1000)),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        ),
        "segment_logical": lambda args: _segment_logical(
            operation=str(args["operation"]),
            modifier_segment_id=args.get("modifier_segment_id"),
            modifier_segment_name=args.get("modifier_segment_name"),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        ),
        "segment_statistics": lambda args: _segment_statistics(
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        ),
        "center_on_segment": lambda args: _center_on_segment(
            segmentation_id=args.get("segmentation_id"),
            segmentation_name=args.get("segmentation_name"),
            segment_id=args.get("segment_id"),
            segment_name=args.get("segment_name"),
        ),
        "resample_volume_isotropic": lambda args: _resample_volume_isotropic(
            output_spacing_mm=float(args.get("output_spacing_mm", 1.0)),
            source_volume_id=args.get("source_volume_id"),
            source_volume_name=args.get("source_volume_name"),
            output_name=args.get("output_name"),
            interpolation_type=str(args.get("interpolation_type", "linear")),
            make_active=bool(args.get("make_active", True)),
        ),
        "export_segmentation_dicom": lambda args: _export_segmentation_dicom(
            segmentation_id=str(args["segmentation_id"]),
            reference_volume_id=str(args["reference_volume_id"]),
            output_folder=str(args["output_folder"]),
        ),
    }


_BRIDGE_HANDLERS = _build_tool_handlers()


def dispatch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Single stable entry point called by external process via `/slicer/exec`."""

    tool = payload.get("tool")
    args = payload.get("args") or {}

    try:
        handler = _BRIDGE_HANDLERS.get(tool)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool}")
        result = handler(args)
        return {"ok": True, **result}

    except Exception as e:
        return {
            "ok": False,
            "tool": tool,
            "error": str(e),
        }
