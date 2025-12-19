import json, uuid, time
from typing import Any, Dict, Optional, List
import numpy as np
import h5py
import hashlib
import os

_UTF8_STR_DTYPE = h5py.string_dtype(encoding="utf-8")

def _to_jsonable(obj: Any) -> Any:
    """Make obj JSON-serializable: convert numpy scalars/arrays and other simple types."""
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj

def _canonical_json(d: Dict[str, Any]) -> str:
    """Stable JSON encoding for hashing/fingerprints."""
    return json.dumps(_to_jsonable(d), sort_keys=True, separators=(",", ":"))

def _read_utf8_scalar(ds) -> str: 
    v = ds[()]
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8")
    return str(v) 


class H5ResultStore:
    """
    Generic hierarchical result store:
      /runs/<uuid>/
    """
    SCHEMA_VERSION = "1.1"

    def __init__(self, path: str, mode: str = "a"):
        self._h = h5py.File(path, mode)
        ra = self._h.attrs
        if "schema_version" not in ra:
            ra["schema_version"] = self.SCHEMA_VERSION
            ra["created_at"]     = str(time.time())
            ra["tool"]           = "H5ResultStore"

    def close(self): self._h.close()
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    # ---------- write ----------
    def new_run(
        self,
        *,
        params: Dict[str, Any],     
        tags: Optional[Dict[str, Any]] = None,
        dedupe_on_fingerprint: bool = False
    ) -> str:
        meta = {"params": params, "tags": tags or {}, "created_at": time.time()}
        meta_json = _canonical_json(meta)
        meta_hash = hashlib.sha256(meta_json.encode("utf-8")).hexdigest()

        if dedupe_on_fingerprint:
            for rid in self.list_runs():
                old = self.get_meta(rid)
                if old.get("fingerprint") == meta_hash:
                    return rid  

        run_id = str(uuid.uuid4())
        run_grp = self._h.create_group(f"/runs/{run_id}")
        run_grp.create_group("data")
        meta["fingerprint"] = meta_hash

        run_grp.create_dataset("attrs/json", data=json.dumps(_to_jsonable(meta)), dtype=_UTF8_STR_DTYPE)
        return run_id

    def save_array(
        self,
        run_id: str,
        name: str,
        array: np.ndarray,
        *,
        compression: str = "gzip",
        compression_opts: int = 4,
        chunks: bool = True,
        overwrite: bool = False,
    ):
        g = self._h[f"/runs/{run_id}/data"]
        if name in g:
            if not overwrite:
                raise ValueError(f"Dataset '{name}' exists for run {run_id}. Set overwrite=True to replace.")
            del g[name]
        g.create_dataset(name, data=array, compression=compression, compression_opts=compression_opts, chunks=chunks)

    # ---------- read / query ----------
    def list_runs(self) -> List[str]:
        return list(self._h.get("/runs", {}).keys()) if "/runs" in self._h else []

    def get_meta(self, run_id: str) -> Dict[str, Any]:
        raw = _read_utf8_scalar(self._h[f"/runs/{run_id}/attrs/json"])
        return json.loads(raw)

    def list_arrays(self, run_id: str) -> List[str]:
        return list(self._h[f"/runs/{run_id}/data"].keys())

    def load_array(self, run_id: str, name: str) -> np.ndarray:
        return self._h[f"/runs/{run_id}/data/{name}"][()]

    def query(self, *, where: Optional[Dict[str, Any]] = None) -> List[str]:
        where = {k: str(v) for k, v in (where or {}).items()}
        out = []
        for rid in self.list_runs():
            meta = self.get_meta(rid)
            ok = True
            for k, v in where.items():
                cur = meta
                for part in k.split("."):
                    if not isinstance(cur, dict) or part not in cur:
                        ok = False; break
                    cur = cur[part]
                if not ok or str(cur) != v:
                    ok = False; break
            if ok:
                out.append(rid)
        return out
    
    @staticmethod
    def merge_files(source_paths: List[str], dest_path: str, delete_sources: bool = False):
        """
        Merge all runs from source_paths into dest_path.
        Does not support deduplication (blind copy).
        """
        # Ensure dest exists
        with H5ResultStore(dest_path, "a") as _: pass
        
        with h5py.File(dest_path, "a") as f_dest:
            dest_runs = f_dest.require_group("runs")
            
            for src in source_paths:
                if not os.path.exists(src): continue
                
                with h5py.File(src, "r") as f_src:
                    if "runs" not in f_src: continue
                    
                    src_runs = f_src["runs"]
                    for run_id in src_runs.keys():
                        # Copy group /runs/<run_id> from src to dest
                        # If run_id collision, we skip or error? 
                        # UUIDs practically guarantee unique, so direct copy is safe.
                        if run_id in dest_runs:
                            print(f"Warning: Run ID {run_id} collision during merge. Skipping.")
                            continue
                            
                        f_src.copy(f"runs/{run_id}", dest_runs, name=run_id)
        
        if delete_sources:
            for src in source_paths:
                try:
                    os.remove(src)
                except OSError:
                    pass