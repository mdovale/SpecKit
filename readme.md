Very fast implementation of the LPSD/LTF algorithm.

Compute times depend highly on the parameters used in the algorithm, especially `Jdes` (desired number of frequencies) and `Kdes` (desired number of averages).

Pure-Python compute times on Apple M3 Max (16 cores, no detrending), using default parameters:

- Time series with 1 million points: 2.3 seconds

- Time series with 10 million points: 16.3 seconds

- Time series with 100 million points: 161 seconds

C implementation using `ctypes` is roughly twice as fast.

As of yet, C implementation is not managing memory correctly. Beware when processing very long time series. 

Algorithm by Gerhard Heinzel and Michael Troebs: https://doi.org/10.1016/j.measurement.2005.10.010

Miguel Dovale, AEI (2024)

![Multiprocessing times](https://gitlab.aei.uni-hannover.de/midova/spectools/uploads/d978646e6567760481fed9235ef0551a/parallel_lpsd.pdf)