from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests


logger = logging.getLogger(__name__)

_SLICE_VIEW_ALIASES = {
    "axial": "red",
    "red": "red",
    "r": "red",
    "sagittal": "yellow",
    "yellow": "yellow",
    "y": "yellow",
    "coronal": "green",
    "green": "green",
    "g": "green",
}


def _normalize_slice_view(view: str) -> str:
    v = (view or "").strip().lower()
    if v in _SLICE_VIEW_ALIASES:
        return _SLICE_VIEW_ALIASES[v]
    raise ValueError(f"Unsupported view: {view!r} (expected axial/sagittal/coronal)")


class SlicerRequestError(RuntimeError):
    """Raised when Slicer WebServer returns an error or is unreachable."""


@dataclass(frozen=True)
class SlicerClient:
    """HTTP client for 3D Slicer WebServer.

    Scope
    -----
    This client intentionally stays close to the WebServer HTTP API. It provides:

    - Thin wrappers for documented `/slicer/*` endpoints (rendering, MRML, volumes, etc.)
    - A *raw* request helper for forward compatibility
    - An `exec_bridge()` helper to call a **fixed entry** script via `/slicer/exec`

    Notes
    -----
    * Most endpoints require that you enable "Slicer API" in the WebServer module.
    * `/slicer/exec` additionally requires enabling "Slicer API exec".
    * Do **NOT** expose exec to untrusted networks.
    """

    base_url: str = "http://localhost:2016"
    timeout_s: float = 60.0

    # -------------------------
    # Core request plumbing
    # -------------------------

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        expect: Optional[str] = None,
        stream: bool = False,
        timeout_s: Optional[float] = None,
    ) -> requests.Response:
        url = self._url(path)
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                data=data,
                json=json_body,
                headers=headers,
                timeout=float(timeout_s if timeout_s is not None else self.timeout_s),
                stream=stream,
            )
        except Exception as e:
            raise SlicerRequestError(f"Failed to call Slicer WebServer: {method} {url}: {e}") from e

        if resp.status_code >= 400:
            # Slicer WebServer often returns text/plain with stack traces.
            try:
                body = resp.text
            except Exception:
                body = "<unreadable body>"
            raise SlicerRequestError(
                f"Slicer WebServer error {resp.status_code} for {method} {url}: {body[:2000]}"
            )

        if expect is not None:
            ct = (resp.headers.get("Content-Type") or "").lower()
            if expect not in ct:
                # Don't hard-fail; just warn (Slicer sometimes varies).
                logger.warning(
                    "Unexpected content-type for %s %s: %s (expected contains %s)",
                    method,
                    url,
                    ct,
                    expect,
                )

        return resp

    def raw_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        expect: Optional[str] = None,
        stream: bool = False,
        timeout_s: Optional[float] = None,
    ) -> requests.Response:
        """Low-level escape hatch for forward-compatibility.

        You should prefer dedicated helpers when available.
        """

        return self._request(
            method,
            path,
            params=params,
            data=data,
            json_body=json_body,
            headers=headers,
            expect=expect,
            stream=stream,
            timeout_s=timeout_s,
        )

    # -------------------------
    # /slicer/system
    # -------------------------

    def get_system_version(self) -> Dict[str, Any]:
        resp = self._request("GET", "/slicer/system/version", expect="application/json")
        return resp.json()

    def shutdown_system(self) -> Dict[str, Any]:
        """Request Slicer to exit via `DELETE /slicer/system` (if enabled)."""
        resp = self._request("DELETE", "/slicer/system")
        # Some builds return JSON, some return text.
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "message": resp.text}

    # -------------------------
    # Rendering endpoints
    # -------------------------

    def get_slice_png(
        self,
        *,
        view: str = "axial",
        orientation: str = "axial",
        scroll_to: float = 0.5,
        size: int = 512,
        offset: Optional[float] = None,
        copy_slice_geometry_from: Optional[str] = None,
    ) -> bytes:
        """Render a 2D slice view to PNG via `/slicer/slice`.

        Parameters map to WebServer's query params:
        - view: axial/sagittal/coronal
        - orientation: axial/sagittal/coronal
        - scrollTo: normalized [0,1]
        - size: square pixel size
        - offset: (optional) alternative to scrollTo
        - copySliceGeometryFrom: (optional) node id
        """
        slicer_view = _normalize_slice_view(view)

        params: Dict[str, Any] = {
            "view": slicer_view,
            "orientation": orientation,
            "scrollTo": str(scroll_to),
            "size": str(int(size)),
        }
        if offset is not None:
            params["offset"] = str(float(offset))
        if copy_slice_geometry_from is not None:
            params["copySliceGeometryFrom"] = str(copy_slice_geometry_from)

        resp = self._request("GET", "/slicer/slice", params=params, expect="image/png")
        return resp.content

    def save_slice_png(
        self,
        out_path: Union[str, Path],
        *,
        view: str = "axial",
        orientation: str = "axial",
        scroll_to: float = 0.5,
        size: int = 512,
        offset: Optional[float] = None,
        copy_slice_geometry_from: Optional[str] = None,
    ) -> Path:
        png = self.get_slice_png(
            view=view,
            orientation=orientation,
            scroll_to=scroll_to,
            size=size,
            offset=offset,
            copy_slice_geometry_from=copy_slice_geometry_from,
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png)
        return out_path

    def is_slice_rendering_available(self) -> bool:
        """Return True if `/slicer/slice` can render a PNG.

        This endpoint requires a Qt layout manager and slice widgets.
        If Slicer was started with `--no-main-window`, then
        `slicer.app.layoutManager()` is None and `/slicer/slice` typically fails with:
        "'NoneType' object has no attribute 'sliceWidget'".
        """

        try:
            _ = self.get_slice_png(view="axial", orientation="axial", scroll_to=0.5, size=64)
            return True
        except Exception:
            return False

    def assert_slice_rendering_available(self) -> None:
        """Raise a clear error if `/slicer/slice` is not usable."""

        try:
            _ = self.get_slice_png(view="axial", orientation="axial", scroll_to=0.5, size=64)
        except Exception as e:
            msg = str(e)
            if "sliceWidget" in msg or "layoutManager" in msg:
                raise SlicerRequestError(
                    "Slicer WebServer /slicer/slice rendering is unavailable. "
                    "This usually happens when Slicer is started with --no-main-window (no Qt layout manager). "
                    "Close that Slicer/WebServer instance and restart Slicer normally (with a main window), "
                    "or let this script auto-launch Slicer (it avoids --no-main-window by default)."
                ) from e
            # A common failure mode *before* any volume is loaded/selected is:
            #   "'NoneType' object has no attribute 'GetDataType'".
            # This usually indicates that the slice pipeline has no image data yet.
            # We keep this as an error, but provide a clearer hint.
            if "GetDataType" in msg or "GetImageData" in msg:
                raise SlicerRequestError(
                    "Slicer WebServer /slicer/slice returned an internal error while rendering. "
                    "This can happen if no volume is loaded/selected in the slice views yet. "
                    "Load a volume (e.g., via DICOM import) and retry. "
                    f"Original error: {msg}"
                ) from e
            raise

    def assert_slice_widgets_available(self) -> None:
        """Check that Qt slice widgets exist (does NOT require a loaded volume).

        The `/slicer/slice` endpoint depends on the Qt layout manager and slice widgets.
        However, `/slicer/slice` may legitimately return HTTP 500 *before* any volume is
        loaded (e.g. "GetDataType" on None image data). For startup readiness we therefore
        validate the GUI infrastructure via `/slicer/exec`.
        """

        script = (
            "import slicer\n"
            "lm = slicer.app.layoutManager()\n"
            "has_lm = lm is not None\n"
            "has_red = False\n"
            "has_logic = False\n"
            "if has_lm:\n"
            "  w = lm.sliceWidget('Red')\n"
            "  has_red = w is not None\n"
            "  if has_red:\n"
            "    has_logic = w.sliceLogic() is not None\n"
            "__execResult = {'ok': bool(has_lm and has_red and has_logic), 'has_layout_manager': bool(has_lm), "
            "'has_red_slice_widget': bool(has_red), 'has_red_slice_logic': bool(has_logic)}\n"
        )

        res = self.exec_python(script)
        if not res.get("ok"):
            raise SlicerRequestError(
                "Slicer GUI slice widgets are not available yet. "
                "This usually means Slicer has not finished initializing the main window/layout. "
                f"Details: {res}"
            )

    def get_screenshot_png(self) -> bytes:
        """Capture a full application screenshot via `/slicer/screenshot` (PNG)."""
        resp = self._request("GET", "/slicer/screenshot", expect="image/png")
        return resp.content

    def save_screenshot_png(self, out_path: Union[str, Path]) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(self.get_screenshot_png())
        return out_path

    def get_threeD_png(self, *, look_from_axis: Optional[str] = None) -> bytes:
        """Render the 3D view to PNG via `/slicer/threeD`.

        `look_from_axis` may be one of: left/right/anterior/posterior/superior/inferior.
        """
        params: Dict[str, Any] = {}
        if look_from_axis is not None:
            params["lookFromAxis"] = str(look_from_axis)
        resp = self._request("GET", "/slicer/threeD", params=params, expect="image/png")
        return resp.content

    def save_threeD_png(self, out_path: Union[str, Path], *, look_from_axis: Optional[str] = None) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(self.get_threeD_png(look_from_axis=look_from_axis))
        return out_path

    def get_timeimage_png(self, *, color: str = "pink") -> bytes:
        """Return a small 'time image' via `/slicer/timeimage` (often used for latency testing)."""
        resp = self._request("GET", "/slicer/timeimage", params={"color": str(color)}, expect="image/png")
        return resp.content

    def save_timeimage_png(self, out_path: Union[str, Path], *, color: str = "pink") -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(self.get_timeimage_png(color=color))
        return out_path

    # -------------------------
    # /slicer/gui
    # -------------------------

    def put_gui(self, *, contents: Optional[str] = None, viewers_layout: Optional[str] = None) -> Dict[str, Any]:
        """Set UI properties via `PUT /slicer/gui`.

        This is usually unnecessary for headless benchmark rendering, but is part of the API.
        """
        body: Dict[str, Any] = {}
        if contents is not None:
            body["contents"] = contents
        if viewers_layout is not None:
            body["viewersLayout"] = viewers_layout
        resp = self._request("PUT", "/slicer/gui", json_body=body, expect="application/json")
        return resp.json()

    # -------------------------
    # MRML endpoints
    # -------------------------

    def mrml_list_nodes(self, *, classname: Optional[str] = None) -> Any:
        params = {"classname": classname} if classname else None
        resp = self._request("GET", "/slicer/mrml", params=params, expect="application/json")
        return resp.json()

    def mrml_names(self, *, classname: Optional[str] = None) -> Any:
        params = {"classname": classname} if classname else None
        resp = self._request("GET", "/slicer/mrml/names", params=params, expect="application/json")
        return resp.json()

    def mrml_ids(self, *, classname: Optional[str] = None) -> Any:
        params = {"classname": classname} if classname else None
        resp = self._request("GET", "/slicer/mrml/ids", params=params, expect="application/json")
        return resp.json()

    def mrml_properties(self, *, node_id: str) -> Any:
        resp = self._request("GET", "/slicer/mrml/properties", params={"id": node_id}, expect="application/json")
        return resp.json()

    def mrml_set_properties(self, *, node_id: str, properties: Dict[str, Any]) -> Any:
        """Set MRML node properties via `PUT /slicer/mrml?id=...`.

        The exact settable properties depends on node class.
        """
        resp = self._request(
            "PUT",
            "/slicer/mrml",
            params={"id": node_id},
            json_body=properties,
        )
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "message": resp.text}

    def mrml_file_get(self, *, node_id: str, file_name: str) -> Any:
        """Save an MRML node to a file on the Slicer machine via `GET /slicer/mrml/file`.

        NOTE: This does NOT download the file content; it triggers a save on the Slicer host.
        """
        resp = self._request("GET", "/slicer/mrml/file", params={"id": node_id, "fileName": file_name})
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "message": resp.text}

    def mrml_file_post(self, *, file_name: str) -> Any:
        """Load an MRML node from a file on the Slicer machine via `POST /slicer/mrml/file`."""
        resp = self._request("POST", "/slicer/mrml/file", params={"fileName": file_name})
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "message": resp.text}

    def mrml_file_delete(self, *, node_id: Optional[str] = None, file_name: Optional[str] = None) -> Any:
        """Delete an MRML node and/or its file via `DELETE /slicer/mrml/file`.

        WebServer doc: `DELETE /mrml/file (id, fileName)`.
        """
        params: Dict[str, Any] = {}
        if node_id is not None:
            params["id"] = node_id
        if file_name is not None:
            params["fileName"] = file_name
        resp = self._request("DELETE", "/slicer/mrml/file", params=params if params else None)
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "message": resp.text}

    # -------------------------
    # Volumes / transforms
    # -------------------------

    def list_volumes(self) -> List[Dict[str, str]]:
        """Return list of scalar/labelmap volumes as {name,id}."""
        resp = self._request("GET", "/slicer/volumes", expect="application/json")
        return resp.json()

    def volume_selection(self, *, cmd: str) -> Any:
        resp = self._request("GET", "/slicer/volumeSelection", params={"cmd": cmd}, expect="application/json")
        return resp.json()

    def list_gridtransforms(self) -> Any:
        resp = self._request("GET", "/slicer/gridtransforms", expect="application/json")
        return resp.json()

    def download_volume_nrrd(self, *, node_id: str, out_path: Union[str, Path]) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resp = self._request("GET", "/slicer/volume", params={"id": node_id}, stream=True)
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return out_path

    def download_gridtransform_nrrd(self, *, node_id: str, out_path: Union[str, Path]) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # The docs sometimes typo this endpoint. Use the canonical one.
        resp = self._request("GET", "/slicer/gridtransform", params={"id": node_id}, stream=True)
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return out_path

    def upload_volume_nrrd(self, *, nrrd_path: Union[str, Path]) -> Any:
        nrrd_path = Path(nrrd_path)
        data = nrrd_path.read_bytes()
        resp = self._request(
            "POST",
            "/slicer/volume",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "message": resp.text}

    # -------------------------
    # Fiducials
    # -------------------------

    def list_fiducials(self) -> Any:
        resp = self._request("GET", "/slicer/fiducials", expect="application/json")
        return resp.json()

    def put_fiducial(self, *, node_id: str, r: float, a: float, s: float) -> Any:
        params = {"id": node_id, "r": str(float(r)), "a": str(float(a)), "s": str(float(s))}
        resp = self._request("PUT", "/slicer/fiducial", params=params, expect="application/json")
        return resp.json()

    # -------------------------
    # Tracking
    # -------------------------

    def get_tracking(self, *, m: Optional[str] = None, q: Optional[str] = None, p: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if m is not None:
            params["m"] = m
        if q is not None:
            params["q"] = q
        if p is not None:
            params["p"] = p
        resp = self._request("GET", "/slicer/tracking", params=params if params else None, expect="application/json")
        return resp.json()

    # -------------------------
    # Sample data
    # -------------------------

    def load_sampledata(self, *, name: str) -> Any:
        resp = self._request("GET", "/slicer/sampledata", params={"name": name}, expect="application/json")
        return resp.json()

    # -------------------------
    # DICOMweb bridge
    # -------------------------

    def access_dicomweb_study(self, payload: Dict[str, Any]) -> Any:
        """POST /slicer/accessDICOMwebStudy

        Payload JSON keys are defined by Slicer WebServer.
        """
        resp = self._request(
            "POST",
            "/slicer/accessDICOMwebStudy",
            json_body=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            return resp.json()
        except Exception:
            return {"ok": True, "message": resp.text}

    # -------------------------
    # Scene lifecycle
    # -------------------------

    def clear_scene(self) -> Dict[str, Any]:
        """Clear MRML scene.

        WebServer historically had `DELETE /slicer/mrml`, but this is not guaranteed.
        We try it first, then fall back to exec if enabled.
        """
        try:
            self._request("DELETE", "/slicer/mrml")
            return {"ok": True, "method": "DELETE /slicer/mrml"}
        except Exception as e:
            # Fallback: exec
            try:
                return self.exec_python(
                    "import slicer\nslicer.mrmlScene.Clear()\n__execResult={'ok': True, 'method': 'exec'}\n"
                )
            except Exception:
                raise

    # -------------------------
    # Exec endpoint (unsafe; use sparingly)
    # -------------------------

    def exec_python(self, source: str, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Execute Python in Slicer via `/slicer/exec`.

        The code must set `__execResult` to a JSON-serializable dict.
        """
        resp = self._request(
            "POST",
            "/slicer/exec",
            data=source.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            expect="application/json",
            timeout_s=timeout_s,
        )
        try:
            return resp.json()
        except Exception as e:
            raise SlicerRequestError(f"Failed to parse /slicer/exec response as JSON: {resp.text[:2000]}") from e

    def is_exec_enabled(self) -> bool:
        """Return True if the `/slicer/exec` endpoint is available.

        Note: If exec is disabled in WebServer (Advanced -> Slicer API exec unchecked), Slicer typically
        responds with HTTP 500 and a message such as: "unknown command b'/exec'".
        """

        try:
            _ = self.exec_python("__execResult = {'ok': True}\n")
            return True
        except Exception:
            return False

    def assert_exec_enabled(self) -> None:
        """Raise a clear error if `/slicer/exec` is unavailable."""

        if not self.is_exec_enabled():
            raise SlicerRequestError(
                "Slicer WebServer exec endpoint is unavailable. "
                "Enable WebServer -> Advanced -> 'Slicer API exec' in Slicer and restart the WebServer."
            )

    def exec_bridge(
        self,
        *,
        bridge_dir: Union[str, Path],
        tool: str,
        args: Dict[str, Any],
        session_id: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Call a *fixed-entry* bridge module inside Slicer via `/slicer/exec`.

        This avoids long inline python snippets and keeps the Slicer-side surface auditable.

        The bridge module must expose `dispatch(payload: dict) -> dict`.
        """
        payload = {
            "tool": tool,
            "args": args,
            "session_id": session_id,
            "ts": time.time(),
        }
        payload_json = json.dumps(payload, ensure_ascii=False)
        payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("ascii")
        bridge_dir = Path(bridge_dir).resolve()

        source = f"""import json, base64, importlib.util, pathlib
bridge_dir = pathlib.Path(r\"{str(bridge_dir)}\")
bridge_path = bridge_dir / "slicer_agent_bridge.py"
spec = importlib.util.spec_from_file_location("slicer_agent_bridge_runtime", str(bridge_path))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load bridge module from: {{bridge_path}}")
sab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sab)
payload = json.loads(base64.b64decode(\"{payload_b64}\").decode('utf-8'))
__execResult = sab.dispatch(payload)
"""
        return self.exec_python(source, timeout_s=timeout_s)

    # -------------------------
    # /dicom (DICOMweb) raw proxy
    # -------------------------

    def dicomweb_request(
        self,
        method: str,
        dicom_path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[str, bytes]] = None,
        stream: bool = False,
    ) -> requests.Response:
        """Raw request to the `/dicom` subtree.

        Slicer exposes its internal DICOM database as DICOMweb services.
        Support varies by Slicer version and configuration.
        """
        if not dicom_path.startswith("/"):
            dicom_path = "/" + dicom_path
        if not dicom_path.startswith("/dicom"):
            dicom_path = "/dicom" + dicom_path
        return self._request(method, dicom_path, params=params, headers=headers, data=data, stream=stream)
