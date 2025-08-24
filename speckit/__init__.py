import os
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, (os.cpu_count() or 1))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from .analysis import (
    compute_spectrum, 
    compute_single_bin, 
    lpsd, 
    SpectrumAnalyzer, 
    SpectrumResult
)
