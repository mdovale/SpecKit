Author: Miguel Dovale
Email: spectools@pm.me

# Spectools Library

The Spectools Library is a comprehensive Python package designed for advanced spectral analysis and signal processing tasks. It offers a collection of modules for performing time-series analysis, windowing, spectral estimation, and system analysis for Single-Input Single-Output (SISO) and Multiple-Input Single-Output (MISO) systems. The library is modular and built for scalability, allowing users to handle both small and large datasets with flexibility and precision.

***

## Features

### Spectral Analysis:

- [ ] Implements the LPSD/LTF algorithms for power spectrum and power spectral density estimation from digitized time series with an approximately log-spaced frequency axis.
- [ ] On multi-channel data, computes the cross spectrum, cross-spectral density, transfer function, coherence, etc.
- [ ] Supports both single-frequency and full-spectrum analysis.
- [ ] Comprehensive set of statistical error definitions.

### Window Functions:
- [ ] Provides a diverse range of flat-top window functions with varying sidelobe levels and differentiability.
- [ ] Optimized overlap values for each window type, ensuring minimal spectral leakage.

### System Analysis:
- [ ] Implements algorithms for optimal spectral analysis of SISO and MISO systems, with options for analytic or numerical solutions.

### Auxiliary Tools:
- [ ] Utilities for detrending, truncation, resampling, and time-shifting of time series data.
- [ ] Flexible tools for loading, aligning, and resampling time series datasets across multiple input files.

### Parallelization:
- [ ] Integration with multiprocessing.

***

## Modules

1. `LTFObject` class (`spectools.ltf`)

	•	Provides functions for spectral estimation using logarithmic time-frequency (LTF) and logarithmic power spectral density (LPSD) algorithms.
	
2. Spectral Analysis (`spectools.lpsd`)

	•	Provides functions for spectral analysis using the LTFObject class.
    
	•	Key Functions:
	
    •	`ltf`: Returns an LTFObject with all relevant spectral estimates in the full Fourier spectrum.
	
    •	`ltf_single_bin`: Returns an LTFObject with all relevant spectral estimates for a single frequency bin.

3. Schedulers (`spectools.schedulers`)

	•	Scheduler algorithms for spectral analysis on approximately log-spaced Fourier frequencies.

	•	Key Functions:
	
    •	`lpsd_plan`: "Classic" `lpsd` scheduler (see Ref. Troebs, Heinzel 2006).
	
    •	`ltf_plan`: New `ltf` scheduler implementing `bmin` and `Lmin` (see Ref. Heinzel 2008).
	
    •	new_ltf_plan: Work in progress.

4. Windowing Functions (`spectools.flattop`)

	•	A collection of flat-top window functions optimized for spectral analysis.
	
    •	Fast Decaying Windows: `SFT3F`, `SFT4F`, `SFT5F`.
	
    •	Minimum Sidelobe Windows: `SFT3M`, `SFT4M`, `SFT5M`.
	
    •	Special Windows: `FTNI`, `FTHP`, `Matlab`, `HFT` family.
	
    •	Key Features:
	
    •	`win_dict`: A dictionary of supported window functions.
	
    •	`olap_dict`: Optimized overlap values for each window type.

5. System Analysis (`spectools.analyzer`)

	•	Tools for SISO and MISO system spectral analysis.
	
    •	Key Functions:
	
    •	`SISO_optimal_spectral_analysis`: Estimates the ASD of a SISO system’s output with input influence subtracted.
	
    •	`MISO_analytic_optimal_spectral_analysis`: Analytic solution for MISO systems.
	
    •	`MISO_numeric_optimal_spectral_analysis`: Numeric solver for MISO systems with large dimensions.

6. Digital Signal Processing Tools (`spectools.dsp`)

	•	Utility functions for handling time-series data and managing spectral analysis workflows.
	
    •	Key Functions:
	
    •	`polynomial_detrend`: Very fast polynomial detrending with numpy.
	
    •	`crop_data`: Crops data to a specified range.
	
    •	`df_timeshift`: Time-shifts columns in a DataFrame.
	
    •	`multi_file_timeseries_resampler`: Aligns and resamples time series data across multiple files.

***

## References

1. **Bendat**  
   *Random Data: Analysis and Measurement Procedures*
   DOI: [10.1002/9781118032428](https://doi.org/10.1002/9781118032428)

2. **Bendat, Piersol**  
   *Engineering Applications of Correlation and Spectral Analysis*  
   ISBN: 978-0-471-57055-4

3. **Troebs, Heinzel**  
   *Improved spectrum estimation from digitized time series on a logarithmic frequency axis*  
   DOI: [10.1016/j.measurement.2005.10.010](https://doi.org/10.1016/j.measurement.2005.10.010)
   
4. **Heinzel**  
   *`lpsd` revisited: `ltf`*  
   AEI Hannover, 2008/02/07 1.1

5. **Heinzel, Rudiger, Schilling**  
   *Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new flat-top windows*  
   Max Planck Publication Repository, 2002