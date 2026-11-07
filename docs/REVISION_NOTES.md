# Revision 1.1.0-paper-optimization

This release updates the corrected October 26 demo to use the current manuscript's frozen-distance C update and primal-residual stopping rule. The final full economy-SVD representation remains an intentional exception to Algorithm 1, preserving the earlier clustering protocol. The manuscript and response letter are unchanged. Historical benchmark tables and tensor-layout diagnostics used earlier implementations and are not reproduced by this release.

## Reviewer concerns covered by code

| Concern | Release behavior | Evidence |
|---|---|---|
| R1-1: m larger than a view's feature dimension | Normalize, then pad to max(d,m); use economy SVD with m columns | Low-dimensional solver test and orthogonality diagnostics |
| R1-6: c <= m and m <= d feasibility | Enforce c <= m <= n; define the feature-space bound using working dimensions | A case with d < c <= m is accepted; c > m and m > n are rejected |
| R1-5: nonnegative weights and near-zero reconstruction errors | Masked errors, explicit zero-error branch, scaled inverse-error weights | Zero, multiple-zero and tiny positive error tests |
| R2-7: reproducibility/code availability | Bundled default NGs data, one entry point, ten full repetitions, saved models/labels/seeds, public repository target | Default run and verification records |

Working feature dimensions, the nonnegative weight constraint, and the complete repetition protocol are documented in the README and saved experiment metadata.

## Optimization changes

* **C update:** Compute normalized distances K from the old C and current F, freeze K, then apply one columnwise simplex projection to `M-alpha/(mu*V)*K`. This exactly solves the convex frozen-distance surrogate. It replaces the original degree-dependent objective's projected-gradient/Armijo loop; surrogate descent is not an original-objective descent guarantee.
* **Stopping:** Stop only when the maximum absolute entry of the two primal residuals is at most `tolPrimal`, or when `maxIter` is exhausted. The corresponding stop reasons are `primal_tolerance` and `max_iterations`. Multiplier updates use the current sweep's mu; the capped penalty growth occurs before the stopping check, including on the terminating sweep.
* **Zero anchor degrees:** Use reciprocal square root zero at exactly zero, leaving all positive degrees unchanged. Zero rows are permitted without flooring degrees or perturbing C. The original graph gradient and standalone stationarity utility remain restricted to positive anchor degrees.
* **Diagnostics and options:** `cSurrogateChange` and `cSurrogateProjectedGradient` replace the old C-objective diagnostic names. Objective and `relativeChange` remain diagnostics. The solver does not compute stationarity; `cdagtl_stationarity` remains available separately. The obsolete options `tolChange`, `tolStationarity`, `cMaxIter`, and `cTolerance` have been removed; `muGrowth` must exceed one.

## Retained behavior

* Normalize and pad each view to `max(d_original,m)` so anchors have m orthonormal columns.
* Use the standard **mean** Fourier-slice TNN and matching proximal threshold, without additional core shrinkage.
* Compute reconstruction and weight errors only on observed samples; handle exactly zero and tiny positive errors explicitly.
* Keep orthonormal F separate from degree-normalized coordinates, the existing initialization, and the A, Z, J, F, D, and weight updates.
* Train without ground-truth labels and record separate clustering seeds. The default NGs protocol performs ten complete training-and-clustering repetitions, each with 20 k-means restarts.
* Form the final representation from every left singular vector returned by the full economy SVD of the transposed mean view graph. The demo retains `m < n`; the direct training API permits `m == n`.

Tests now check the frozen surrogate's optimality, zero-degree continuation, primal-only stopping, penalty order, and the retained full-SVD representation, alongside the existing dimension, tensor, weight, masking, metric, and experiment-orchestration checks. Version 1.0.0 validation records are archived under `validation/archive/1.0.0-revision/`; new validation belongs to this version and must not replace historical manuscript results.

## Experimental boundaries

Ten repetitions mean ten calls to the complete solver and final clustering. They do not mean ten independent masks or necessarily ten distinct trained representations. A deterministic update sequence may give identical representations, and distinct k-means seeds may still give identical partitions. No zero standard deviation is replaced or suppressed.

The graph problem remains nonconvex despite the exact frozen-surrogate C step. Primal feasibility and an exhausted iteration budget are not stationarity certificates, and this release does not establish a general multi-block ADMM convergence result. Saved stop reasons and residuals distinguish these outcomes.

The sample-mode tensor prior is order-dependent in general. The earlier comparison in the response letter used historical updates and automatic core shrinkage; it is not an ablation of this implementation.
