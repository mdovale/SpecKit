import os
import multiprocessing

# Ensure OpenBLAS does not oversubscribe CPU cores
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Set multiprocessing start method to "spawn" (if not already set)
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn", force=True)