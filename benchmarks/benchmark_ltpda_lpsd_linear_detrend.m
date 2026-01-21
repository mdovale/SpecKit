function benchmark_ltpda_lpsd_linear_detrend()
% benchmark_ltpda_lpsd_linear_detrend
%
% Benchmarks LTPDA ao/lpsd performance using linear detrending (Order = 1)
% on 10 million points of uniformly distributed random data.
%
% LPSD parameters requested:
%   fs   = 2
%   Jdes = 1000
%   Kdes = 100
%   bmin = 1.0   (NOTE: not exposed as an ao/lpsd plist key in LTPDA docs)
%   Lmin = 1
%   win  = "hann"  (mapped to LTPDA window name 'Hanning')
%   olap = 0.75  (fractional; LTPDA uses percentage, so 75 in plist = 0.75 fractional)
%
% See LTPDA docs for ao/lpsd parameter keys and detrending Order.
%
% Output:
%   - Prints timing statistics for:
%       * Order = 1 (linear detrend)  [main requested case]
%       * Order = 0 (mean removal)    [reference]
%       * Order = -1 (no detrend)     [reference]
%
% Notes:
%   - This script avoids file I/O to benchmark the spectral estimation itself.
%   - It constructs a tsdata AO with FS set, so no explicit x-vector is stored.

  %% --- Configuration ---
  fs   = 2;
  Jdes = 1000;
  Kdes = 100;
  bmin = 1.0; %#ok<NASGU>  % Not used: LTPDA ao/lpsd does not list 'bmin' as a plist key.
  Lmin = 1;

  % User requested "hann"; LTPDA window options list uses 'Hanning'.
  winName = 'Hanning';

  olap = 75;  % LTPDA uses percentage (75 = 75% = 0.75 fractional overlap)

  N = 10000000;   % 10 million points
  rng(0, 'twister');  % reproducible

  % Benchmark repetition settings
  nWarmup = 1;
  nRuns   = 10;      % increase if you want tighter statistics (at the cost of time)

  fprintf('--- LTPDA ao/lpsd benchmark (linear detrending) ---\n');
  fprintf('N = %d samples, fs = %.6g Hz\n', N, fs);
  fprintf('LPSD params: Jdes=%d, Kdes=%d, Lmin=%d, Win=%s, Olap=%.2f, Scale=ASD\n', ...
          Jdes, Kdes, Lmin, winName, olap);
  fprintf('Detrending: Order=1 (linear)\n\n');

  %% --- Generate uniform random data (centered) ---
  % Uniform in (-0.5, +0.5); still "uniformly distributed" but zero-mean-ish.
  y = rand(N, 1) - 0.5;

  %% --- Build AO (tsdata) without explicit xvals ---
  % Use "From XY Values" constructor keys: TYPE, FS, YVALS.
  % With FS set, any XVALS (if passed) are ignored and an evenly-sampled x is implied.
  plAO = plist( ...
    'type',   'tsdata', ...
    'fs',     fs, ...
    'yvals',  y, ...
    'yunits', 'rad', ...
    'name',   sprintf('uniform_rand_N%d_fs%.6g', N, fs) ...
  );

  ch = ao(plAO);

    % --- LPSD parameter list (common) ---
    plCommon = plist( ...
    'SCALE', 'ASD', ...
    'WIN',   'Hanning', ...
    'OLAP',  75, ...
    'KDES',  100, ...
    'JDES',  1000, ...
    'LMIN',  1 ...
    );
    
    % Create variants by setting ORDER using pset (recommended)
    plNoDetrend   = pset(plCommon, 'ORDER', -1);  % no detrending
    plMeanDetrend = pset(plCommon, 'ORDER',  0);  % subtract mean
    plLinDetrend  = pset(plCommon, 'ORDER',  1);  % subtract linear fit (requested)

  %% --- Warm-up (JIT, caches, etc.) ---
  % Use a small dataset for warm-up to avoid long computation times.
  N_warmup = 50000;  % Small dataset sufficient to trigger compilation
  y_warmup = rand(N_warmup, 1) - 0.5;
  plAO_warmup = plist( ...
    'type',   'tsdata', ...
    'fs',     fs, ...
    'yvals',  y_warmup, ...
    'yunits', 'rad', ...
    'name',   sprintf('uniform_rand_N%d_fs%.6g_warmup', N_warmup, fs) ...
  );
  ch_warmup = ao(plAO_warmup);
  
  fprintf('Warm-up runs: %d per order (orders: -1, 0, 1) on %d samples\n', nWarmup, N_warmup);
  for order_warmup = [-1, 0, 1]
    plWarmup = pset(plCommon, 'ORDER', order_warmup);
    for k = 1:nWarmup
      tmp = lpsd(ch_warmup, plWarmup); %#ok<NASGU>
      clear tmp
    end
  end
  clear ch_warmup plAO_warmup y_warmup plWarmup
  fprintf('Warm-up done.\n\n');

  %% --- Benchmark helper ---
  % Returns per-run times in seconds.
  function tvec = bench_lpsd(pl, label)
    tvec = zeros(nRuns, 1);
    fprintf('Benchmark: %s (nRuns=%d)\n', label, nRuns);
    for i = 1:nRuns
      t0 = tic;
      out = lpsd(ch, pl); %#ok<NASGU>
      tvec(i) = toc(t0);
      clear out
      fprintf('  run %d/%d: %.3f s\n', i, nRuns, tvec(i));
    end
    fprintf('\n');
  end

  %% --- Run benchmarks ---
  t_lin  = bench_lpsd(plLinDetrend,  'Order=1 (linear detrend)  [requested]');
  t_mean = bench_lpsd(plMeanDetrend, 'Order=0 (mean removal)    [reference]');
  t_none = bench_lpsd(plNoDetrend,   'Order=-1 (no detrend)     [reference]');

  %% --- Report summary ---
  fprintf('--- Summary (seconds) ---\n');
  report_stats('Order=1 (linear)', t_lin);
  report_stats('Order=0 (mean)  ', t_mean);
  report_stats('Order=-1 (none) ', t_none);

  fprintf('\nDone.\n');

  %% --- Local stats printer ---
  function report_stats(name, t)
    fprintf('%s: mean=%.3f, median=%.3f, std=%.3f, min=%.3f, max=%.3f\n', ...
      name, mean(t), median(t), std(t), min(t), max(t));
  end

end