# Revision release

This release is based on the corrected October 26 demo. It is separate from the later candidate that restored historical updates and only repaired dimensions. It must not be confused with the historical implementation used for the submitted benchmark tables or the tensor-layout diagnostics in the response letter.

## Reviewer concerns covered by code

| Concern | Release behavior | Evidence |
|---|---|---|
| R1-1: m larger than a view's feature dimension | Normalize, then pad to max(d,m); use economy SVD with m columns | Low-dimensional solver test and orthogonality diagnostics |
| R1-6: c <= m and m <= d feasibility | Enforce c <= m <= n; define the feature-space bound using working dimensions | A case with d < c <= m is accepted; c > m and m > n are rejected |
| R1-5: nonnegative weights and near-zero reconstruction errors | Masked errors, explicit zero-error branch, scaled inverse-error weights | Zero, multiple-zero and tiny positive error tests |
| R2-7: reproducibility/code availability | Bundled default NGs data, one entry point, ten full repetitions, saved models/labels/seeds, public repository target | Default run and verification records |

The code cannot replace the associated explanations in the manuscript. In particular, the working feature space and nonnegative weight constraint must be stated explicitly, and this release must be identified separately from the historical table-generating implementation.

## Preserved corrections from the old corrected demo

1. Standard **mean** Fourier-slice TNN and its matching proximal threshold; no additional historical core shrinkage.
2. Reconstruction errors restricted to observed samples, including the weight subproblem.
3. Correct orthonormal F separated from degree-normalized coordinates.
4. Positive-degree graph initialization and C updates that reject zero-degree candidates.
5. Derivative of anchor degrees in the normalized spectral C objective, with Armijo descent.
6. Same mu in each sweep's primal and multiplier updates, increasing it only for the next sweep.
7. Actual objective, residual, stationarity and change diagnostics; exact iteration budget and honest stop reason.
8. Label-free training API and separate clustering seeds.

## Experimental boundaries

Ten repetitions mean ten calls to the complete solver and final clustering. They do not mean ten independent masks or necessarily ten distinct trained representations. A deterministic update sequence may give identical representations, and distinct k-means seeds may still give identical partitions. No zero standard deviation is replaced or suppressed.

The graph problem remains nonconvex, the C update is inexact, and the current implementation does not come with a general multi-block ADMM convergence proof. A small primal residual or a completed iteration budget alone is not a stationarity certificate. The release records all diagnostics needed to distinguish these outcomes.

The sample-mode tensor prior is order-dependent in general. Retaining it here does not settle the broader sample-permutation issue raised in R3-1. The earlier comparison in the response letter used historical updates and automatic core shrinkage; it is not presented as an ablation of this corrected implementation.
