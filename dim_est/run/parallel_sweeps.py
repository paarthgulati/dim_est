import os
import uuid
from typing import List, Callable, Dict, Any
from joblib import Parallel, delayed
from ..utils.h5_result_store import H5ResultStore

def _worker_wrapper(func: Callable, kwargs: Dict[str, Any], temp_dir: str) -> str:
    """
    Executes 'func' (e.g. run_dsib_finite) but forces it to write to a 
    unique temp H5 file in temp_dir.
    Returns the path to the temp file.
    """
    # Create unique filename
    unique_id = str(uuid.uuid4())
    temp_file = os.path.join(temp_dir, f"worker_{unique_id}.h5")
    
    # Inject outfile into kwargs
    kwargs_copy = kwargs.copy()
    kwargs_copy["outfile"] = temp_file
    
    # Run
    # We assume func has signature matching run_dsib_*(..., outfile=...)
    func(**kwargs_copy)
    
    return temp_file

def run_sweep_parallel(
    func: Callable,
    sweep_configs: List[Dict[str, Any]],
    final_outfile: str,
    n_jobs: int = -1,
    temp_dir: str = ".temp_sweeps"
):
    """
    Runs a list of configurations in parallel using joblib.
    
    Args:
        func: The runner function (run_dsib_finite or run_dsib_infinite)
        sweep_configs: List of dicts, each containing arguments for 'func'.
                       (Do not include 'outfile' here; it is handled automatically).
        final_outfile: Path to the final merged HDF5.
        n_jobs: Number of parallel workers (-1 = all CPUs).
        temp_dir: Directory to store temporary HDF5 files.
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Starting parallel sweep with {len(sweep_configs)} jobs across {n_jobs} workers...")
    
    # 1. Scatter (Execute in Parallel)
    temp_files = Parallel(n_jobs=n_jobs)(
        delayed(_worker_wrapper)(func, cfg, temp_dir) 
        for cfg in sweep_configs
    )
    
    print("All jobs completed. Merging results...")
    
    # 2. Gather (Merge HDF5)
    # Filter out None if any failed (though joblib raises error by default)
    temp_files = [f for f in temp_files if f and os.path.exists(f)]
    
    H5ResultStore.merge_files(temp_files, final_outfile, delete_sources=True)
    
    # Cleanup dir if empty
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
        
    print(f"Parallel sweep merged into: {final_outfile}")