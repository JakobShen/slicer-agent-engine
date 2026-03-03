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


BRIDGE_VERSION = "0.3.0"


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
    if v in {"red", "r"}:
        return "Red"
    if v in {"yellow", "y"}:
        return "Yellow"
    if v in {"green", "g"}:
        return "Green"
    # Allow passing already-normalized names
    if v in {"red", "yellow", "green"}:
        return v.title()
    raise ValueError(f"Unsupported view: {view!r} (expected red/yellow/green)")


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


def _safe_slab_type_as_string(sn) -> Optional[str]:
    """Best-effort slab reconstruction type string across Slicer/VTK wrapper variants.

    Some Slicer builds expose `GetSlabReconstructionTypeAsString()` with **no arguments**,
    while others expose it as a static/utility method that requires the integer type.

    We avoid raising here because slab state is often informational.
    """

    if not hasattr(sn, "GetSlabReconstructionTypeAsString"):
        return None

    fn = getattr(sn, "GetSlabReconstructionTypeAsString")
    # Variant A: no-arg
    try:
        return str(fn())
    except Exception:
        pass

    # Variant B: requires the type argument
    try:
        t = int(sn.GetSlabReconstructionType()) if hasattr(sn, "GetSlabReconstructionType") else None
        if t is None:
            return None
        return str(fn(t))
    except Exception:
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

    for v in ("red", "yellow", "green"):
        try:
            sn = _slice_node(v)
            comp = _slice_composite_node(v)
            view_state: Dict[str, Any] = {
                "orientation": sn.GetOrientationString() if hasattr(sn, "GetOrientationString") else str(sn.GetOrientation()),
                "slice_offset": float(sn.GetSliceOffset()),
                "field_of_view": list(sn.GetFieldOfView()) if hasattr(sn, "GetFieldOfView") else None,
                "xyz_origin": list(sn.GetXYZOrigin()) if hasattr(sn, "GetXYZOrigin") else None,
                "background_volume_id": str(comp.GetBackgroundVolumeID()) if comp else None,
                "foreground_volume_id": str(comp.GetForegroundVolumeID()) if comp else None,
                "foreground_opacity": float(comp.GetForegroundOpacity()) if comp and hasattr(comp, "GetForegroundOpacity") else None,
                "linked_control": int(comp.GetLinkedControl()) if comp and hasattr(comp, "GetLinkedControl") else None,
            }

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

            state["views"][v] = view_state
        except Exception as e:
            state["views"][v] = {"error": str(e)}

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
    return {"view": _norm_view(view), "orientation": sn.GetOrientationString() if hasattr(sn, "GetOrientationString") else str(sn.GetOrientation())}


def _get_slice_offset_range(*, view: str) -> Dict[str, Any]:
    logic = _slice_logic(view)
    lo = None
    hi = None
    if hasattr(logic, "GetLowestVolumeSliceOffset") and hasattr(logic, "GetHighestVolumeSliceOffset"):
        lo = float(logic.GetLowestVolumeSliceOffset())
        hi = float(logic.GetHighestVolumeSliceOffset())
    elif hasattr(logic, "GetSliceOffsetRange"):
        r = logic.GetSliceOffsetRange()
        lo = float(r[0])
        hi = float(r[1])
    else:
        raise RuntimeError("SliceLogic does not expose slice offset range in this Slicer build.")

    return {"view": _norm_view(view), "min_offset": lo, "max_offset": hi}


def _set_slice_offset(*, view: str, offset: float) -> Dict[str, Any]:
    sn = _slice_node(view)
    sn.SetSliceOffset(float(offset))
    slicer.app.processEvents()
    return {"view": _norm_view(view), "slice_offset": float(sn.GetSliceOffset())}


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
    return {"view": _norm_view(view), "field_of_view": list(sn.GetFieldOfView()), "slice_offset": float(sn.GetSliceOffset())}


def _zoom_slice_relative(*, view: str, factor: float) -> Dict[str, Any]:
    if factor <= 0:
        raise ValueError("factor must be > 0")
    sn = _slice_node(view)
    if not hasattr(sn, "GetFieldOfView") or not hasattr(sn, "SetFieldOfView"):
        raise RuntimeError("SliceNode does not support field-of-view control in this Slicer build.")
    fov = sn.GetFieldOfView()
    sn.SetFieldOfView(float(fov[0]) / float(factor), float(fov[1]) / float(factor), float(fov[2]))
    slicer.app.processEvents()
    return {"view": _norm_view(view), "field_of_view": list(sn.GetFieldOfView()), "factor": float(factor)}




def _set_field_of_view(*, view: str, field_of_view: Sequence[float]) -> Dict[str, Any]:
    sn = _slice_node(view)
    if not hasattr(sn, "SetFieldOfView"):
        raise RuntimeError("SliceNode does not support SetFieldOfView in this Slicer build.")
    if len(field_of_view) != 3:
        raise ValueError("field_of_view must be length-3 [x,y,z]")
    fx, fy, fz = [float(v) for v in field_of_view]
    sn.SetFieldOfView(fx, fy, fz)
    slicer.app.processEvents()
    return {"view": _norm_view(view), "field_of_view": list(sn.GetFieldOfView())}


def _set_xyz_origin(*, view: str, xyz_origin: Sequence[float]) -> Dict[str, Any]:
    sn = _slice_node(view)
    if not hasattr(sn, "SetXYZOrigin"):
        raise RuntimeError("SliceNode does not support SetXYZOrigin in this Slicer build.")
    if len(xyz_origin) != 3:
        raise ValueError("xyz_origin must be length-3 [x,y,z]")
    x, y, z = [float(v) for v in xyz_origin]
    sn.SetXYZOrigin(x, y, z)
    slicer.app.processEvents()
    return {"view": _norm_view(view), "xyz_origin": list(sn.GetXYZOrigin())}
def _jump_to_ras(*, ras: Sequence[float], centered: bool = True) -> Dict[str, Any]:
    if len(ras) != 3:
        raise ValueError("ras must be length-3 [x,y,z] in RAS (mm)")
    x, y, z = [float(v) for v in ras]
    slicer.util.jumpSlicesToLocation(x, y, z, centered)
    slicer.app.processEvents()
    return {"ras": [x, y, z], "centered": bool(centered)}


def _set_linked_slices(*, enabled: bool) -> Dict[str, Any]:
    results: Dict[str, Any] = {"enabled": bool(enabled), "views": {}}
    for v in ("red", "yellow", "green"):
        try:
            comp = _slice_composite_node(v)
            if comp is None:
                continue
            if hasattr(comp, "SetLinkedControl"):
                comp.SetLinkedControl(1 if enabled else 0)
            if hasattr(comp, "SetHotLinkedControl"):
                comp.SetHotLinkedControl(1 if enabled else 0)
            results["views"][v] = {
                "linked_control": int(comp.GetLinkedControl()) if hasattr(comp, "GetLinkedControl") else None
            }
        except Exception as e:
            results["views"][v] = {"error": str(e)}
    slicer.app.processEvents()
    return results


def _set_layout(*, layout: str) -> Dict[str, Any]:
    lm = _require_layout_manager()
    ln = slicer.app.layoutManager().layoutLogic().GetLayoutNode()

    name = (layout or "").strip().lower()
    # Small, deterministic subset of layouts.
    layout_map = {
        "four_up": slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView,
        "fourup": slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView,
        "one_up_red": slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView,
        "one_up_yellow": slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView,
        "one_up_green": slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpGreenSliceView,
        "three_up": slicer.vtkMRMLLayoutNode.SlicerLayoutThreeUpView,
        "conventional": slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView,
    }
    if name not in layout_map:
        raise ValueError(f"Unsupported layout: {layout!r}. Supported: {sorted(layout_map.keys())}")
    ln.SetViewArrangement(layout_map[name])
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
        for fn in ("SetSlabReconstructionTypeToMax", "SetSlabReconstructionTypeToMaximum", "SetSlabReconstructionTypeToMIP"):
            if hasattr(sn, fn):
                getattr(sn, fn)()
                break
        else:
            if hasattr(sn, "SetSlabReconstructionType"):
                sn.SetSlabReconstructionType(0)
    elif mode_l in {"min", "minimum"}:
        for fn in ("SetSlabReconstructionTypeToMin", "SetSlabReconstructionTypeToMinimum"):
            if hasattr(sn, fn):
                getattr(sn, fn)()
                break
        else:
            if hasattr(sn, "SetSlabReconstructionType"):
                sn.SetSlabReconstructionType(1)
    elif mode_l in {"mean", "avg", "average"}:
        for fn in ("SetSlabReconstructionTypeToMean", "SetSlabReconstructionTypeToAverage"):
            if hasattr(sn, fn):
                getattr(sn, fn)()
                break
        else:
            if hasattr(sn, "SetSlabReconstructionType"):
                sn.SetSlabReconstructionType(2)
    else:
        raise ValueError("mode must be one of: mip/max, min, mean")

    slicer.app.processEvents()

    out: Dict[str, Any] = {
        "view": _norm_view(view),
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
    slicer.mrmlScene.Clear()
    slicer.app.processEvents()
    return {"cleared": True}


def _load_dicom(dicom_dir: str, clear_scene_first: bool = True) -> Dict[str, Any]:
    dicom_dir = os.path.expanduser(dicom_dir)
    dicom_dir = os.path.abspath(dicom_dir)
    if not os.path.isdir(dicom_dir):
        raise ValueError(f"dicom_dir is not a directory: {dicom_dir}")

    if clear_scene_first:
        slicer.mrmlScene.Clear()

    from DICOMLib import DICOMUtils

    loaded_node_ids: List[str] = []
    patient_uids: List[str] = []
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        DICOMUtils.importDicom(dicom_dir, db)
        patient_uids = list(db.patients())
        for patient_uid in patient_uids:
            loaded_node_ids.extend(DICOMUtils.loadPatientByUID(patient_uid))

    # Pick first scalar volume (if any) and set as active
    volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    active_volume_id = None
    active_volume_name = None
    if volumes:
        volume_node = None
        loaded_set = set(loaded_node_ids)
        for v in volumes:
            if v.GetID() in loaded_set:
                volume_node = v
                break
        if volume_node is None:
            volume_node = volumes[0]
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
    }


def _select_volume(volume_id: Optional[str] = None, volume_name: Optional[str] = None) -> Dict[str, Any]:
    node = _get_node_by_id_or_name(node_id=volume_id, node_name=volume_name)
    if not node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
        raise ValueError(f"Node is not a volume: id={node.GetID()} class={node.GetClassName()} name={node.GetName()}")
    _set_active_volume(node)
    slicer.app.processEvents()
    return {"active_volume_id": node.GetID(), "active_volume_name": node.GetName()}


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

    import DICOMSegmentationPlugin

    exporter = DICOMSegmentationPlugin.DICOMSegmentationPluginClass()
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


def dispatch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Single stable entry point called by external process via `/slicer/exec`."""

    tool = payload.get("tool")
    args = payload.get("args") or {}

    try:
        if tool == "ping":
            return {
                "ok": True,
                "bridge_version": BRIDGE_VERSION,
                "slicer_version": slicer.app.applicationVersion,
                "slicer_revision": slicer.app.revision,
            }

        # Scene / data
        if tool == "clear_scene":
            result = _clear_scene()
            return {"ok": True, **result}

        if tool == "load_dicom":
            result = _load_dicom(
                dicom_dir=str(args["dicom_dir"]),
                clear_scene_first=bool(args.get("clear_scene_first", True)),
            )
            return {"ok": True, **result}

        if tool == "select_volume":
            result = _select_volume(
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            )
            return {"ok": True, **result}

        # L0 viewer actuation
        if tool == "get_viewer_state":
            result = _get_viewer_state()
            return {"ok": True, **result}

        if tool == "set_window_level":
            result = _set_window_level(
                window=args.get("window"),
                level=args.get("level"),
                auto=args.get("auto"),
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            )
            return {"ok": True, **result}

        if tool == "set_interpolation":
            result = _set_interpolation(
                interpolate=bool(args["interpolate"]),
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
            )
            return {"ok": True, **result}

        if tool == "set_slice_orientation":
            result = _set_slice_orientation(view=str(args["view"]), orientation=str(args["orientation"]))
            return {"ok": True, **result}

        if tool == "get_slice_offset_range":
            result = _get_slice_offset_range(view=str(args["view"]))
            return {"ok": True, **result}

        if tool == "set_slice_offset":
            result = _set_slice_offset(view=str(args["view"]), offset=float(args["offset"]))
            return {"ok": True, **result}

        if tool == "set_slice_scroll_to":
            result = _set_slice_scroll_to(view=str(args["view"]), scroll_to=float(args["scroll_to"]))
            return {"ok": True, **result}

        if tool == "fit_slice":
            result = _fit_slice(view=str(args["view"]))
            return {"ok": True, **result}

        if tool == "zoom_slice_relative":
            result = _zoom_slice_relative(view=str(args["view"]), factor=float(args["factor"]))
            return {"ok": True, **result}


        if tool == "set_field_of_view":
            result = _set_field_of_view(view=str(args["view"]), field_of_view=args["field_of_view"])
            return {"ok": True, **result}

        if tool == "set_xyz_origin":
            result = _set_xyz_origin(view=str(args["view"]), xyz_origin=args["xyz_origin"])
            return {"ok": True, **result}

        if tool == "jump_to_ras":
            result = _jump_to_ras(ras=args["ras"], centered=bool(args.get("centered", True)))
            return {"ok": True, **result}

        if tool == "set_linked_slices":
            result = _set_linked_slices(enabled=bool(args["enabled"]))
            return {"ok": True, **result}

        if tool == "set_layout":
            result = _set_layout(layout=str(args["layout"]))
            return {"ok": True, **result}

        # L1 quantification / processing
        if tool == "roi_stats_ijk":
            result = _roi_stats_ijk(
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
                ijk_min=args.get("ijk_min"),
                ijk_max=args.get("ijk_max"),
            )
            return {"ok": True, **result}

        if tool == "roi_stats_ras_box":
            result = _roi_stats_ras_box(
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
                ras_min=args.get("ras_min"),
                ras_max=args.get("ras_max"),
            )
            return {"ok": True, **result}

        if tool == "sample_intensity_ras":
            result = _sample_intensity_ras(
                volume_id=args.get("volume_id"),
                volume_name=args.get("volume_name"),
                ras=args.get("ras"),
                method=str(args.get("method", "nearest")),
            )
            return {"ok": True, **result}

        if tool == "measure_distance_ras":
            result = _measure_distance_ras(p1=args.get("p1"), p2=args.get("p2"))
            return {"ok": True, **result}

        if tool == "measure_angle_ras":
            result = _measure_angle_ras(p1=args.get("p1"), p2=args.get("p2"), p3=args.get("p3"))
            return {"ok": True, **result}

        if tool == "measure_area_polygon_ras":
            result = _measure_area_polygon_ras(points=args.get("points"))
            return {"ok": True, **result}

        if tool == "set_thick_slab":
            result = _set_thick_slab(
                view=str(args["view"]),
                enabled=bool(args.get("enabled", True)),
                thickness_mm=float(args.get("thickness_mm", 0.0)),
                mode=str(args.get("mode", "mip")),
            )
            return {"ok": True, **result}

        if tool == "set_fusion":
            result = _set_fusion(
                background_volume_id=str(args["background_volume_id"]),
                foreground_volume_id=str(args["foreground_volume_id"]),
                opacity=float(args.get("opacity", 0.5)),
            )
            return {"ok": True, **result}

        if tool == "compute_subtraction":
            result = _compute_subtraction(
                volume_a_id=str(args["volume_a_id"]),
                volume_b_id=str(args["volume_b_id"]),
                output_name=str(args.get("output_name", "Subtraction")),
            )
            return {"ok": True, **result}

        if tool == "export_segmentation_dicom":
            result = _export_segmentation_dicom(
                segmentation_id=str(args["segmentation_id"]),
                reference_volume_id=str(args["reference_volume_id"]),
                output_folder=str(args["output_folder"]),
            )
            return {"ok": True, **result}

        raise ValueError(f"Unknown tool: {tool}")

    except Exception as e:
        return {
            "ok": False,
            "tool": tool,
            "error": str(e),
        }
