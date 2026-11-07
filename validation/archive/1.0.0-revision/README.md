# Default NGs validation

Validated on 2026-10-30 with MATLAB R2024b Update 3. The unmodified `run_demo()` defaults were used: NGs Per0.5, ten complete solver executions, 300 sweeps maximum, and twenty k-means restarts per execution.

| Metric | Mean | Sample standard deviation |
|---|---:|---:|
| ACC | 0.9960000000 | 2.34056e-16 |
| NMI | 0.9860691161 | 1.28198e-16 |
| ARI | 0.9899699009 | 1.17028e-16 |
| Purity | 0.9960000000 | 2.34056e-16 |
| Fscore | 0.9919599208 | 0 |
| Precision | 0.9919198449 | 0 |
| Recall | 0.9920000000 | 2.34056e-16 |

All 10 executions ended with `max_iterations`, not `tolerance`. All final F matrices, objectives and stationarity diagnostics were finite. Column orthogonality, F orthogonality, positive anchor degrees and masked weight formulas were checked.
The maximum final primal residual was 1.12846e-10; the maximum final stationarity diagnostic was 0.00724479. These values do not establish general convergence.

All 10 final partitions were identical up to label permutation. Floating-point-level sample deviations are retained rather than rounded to zero in the machine-readable files. All seventy metric values were independently recomputed using a separate evaluation implementation; means and sample deviations were independently checked.

The ten per-run models and labels are produced by `run_demo` under `results/`; the compact CSV and verification records here document this validation. No historical manuscript table is overwritten or relabeled as this release.
