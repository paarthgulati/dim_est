import json, uuid, time
from typing import Any, Dict, Optional, List
import numpy as np
import h5py
import hashlib

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
    # primitives (str, int, float, bool, None) pass through
    return obj

def _canonical_json(d: Dict[str, Any]) -> str:
    """Stable JSON encoding for hashing/fingerprints."""
    return json.dumps(_to_jsonable(d), sort_keys=True, separators=(",", ":"))

class H5ResultStore:
    """
    Generic hierarchical result store:
      /runs/<uuid>/
          data/<name>        # ndarray datasets
          attrs/json         # JSON (params, tags, created_at, fingerprint)
    Root attrs: schema_version, created_at, tool
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
        params: Dict[str, Any],     # e.g. {"opt": opt_params, "data": data_params, "critic": base_critic_params}
        tags: Optional[Dict[str, Any]] = None,
        dedupe_on_fingerprint: bool = False
    ) -> str:
        meta = {"params": params, "tags": tags or {}, "created_at": time.time()}
        meta_json = _canonical_json(meta)
        meta_hash = hashlib.sha256(meta_json.encode("utf-8")).hexdigest()

        if dedupe_on_fingerprint:
            # simple linear scan (fine for modest number of runs)
            for rid in self.list_runs():
                old = self.get_meta(rid)
                if old.get("fingerprint") == meta_hash:
                    return rid  # reuse existing run

        run_id = str(uuid.uuid4())
        run_grp = self._h.create_group(f"/runs/{run_id}")
        run_grp.create_group("data")
        meta["fingerprint"] = meta_hash
        run_grp.create_dataset("attrs/json", data=np.string_(json.dumps(_to_jsonable(meta))))
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
        raw = self._h[f"/runs/{run_id}/attrs/json"][()].decode("utf-8")
        return json.loads(raw)

    def list_arrays(self, run_id: str) -> List[str]:
        return list(self._h[f"/runs/{run_id}/data"].keys())

    def load_array(self, run_id: str, name: str) -> np.ndarray:
        return self._h[f"/runs/{run_id}/data/{name}"][()]

    def query(self, *, where: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Exact-match query on dotted keys into meta:
          where={'params.opt.batch_size': 128, 'tags.estimator': 'infonce'}
        """
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
