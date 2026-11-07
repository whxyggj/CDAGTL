# CDAGTL

MATLAB implementation of **Consistent and Diverse Anchor Graph Tensor Learning for Incomplete Multi-view Clustering**.

Version **1.1.0-paper-optimization** aligns the C update and stopping rule with the current manuscript revision while retaining feature-space padding, stable nonnegative view weights, and ten complete experiment repetitions. The final full economy-SVD representation is intentionally retained from the earlier clustering protocol. Historical manuscript tables and tensor-layout diagnostics used earlier implementations; this release does not claim to reproduce them.

## Quick start

Tested with MATLAB R2024b Update 3. The solver, metrics and bundled k-means use base MATLAB; Statistics and Machine Learning Toolbox is not required.

From this repository folder:

```matlab
addpath(pwd);
result = run_demo();
```

The default command loads the bundled **NGs** dataset at **missing rate 0.5** and executes **10 complete training-and-clustering repetitions**. Each repetition calls `cdagtl_fit` again and performs one final k-means call with 20 internal restarts. Restarts are not counted as complete repetitions.

Results are written to a new directory under `results/`. An existing result directory is never overwritten.

```matlab
result.mean         % ACC, NMI, ARI, Purity, Fscore, Precision, Recall
result.std          % sample standard deviations across all 10 executions
result.runs(1).stopReason
result.outputDir
```

## Exact default protocol

| Setting | Default |
|---|---|
| Dataset | `data/my_NGs_Per0.5.mat` |
| Complete repetitions | 10 |
| Missing-view pattern and sample order | Fixed across repetitions |
| Anchors | `m = 2 * numClusters` (10 on NGs) |
| Alpha / beta | 0.03125 / 1 |
| Training seeds | 202610251 through 202610260 |
| Clustering seeds | Training seed + 10000, modulo 2^32 |
| Internal k-means restarts | 20 per complete repetition |
| Training iteration budget | 300 |
| Tensor arrangement | Anchors by views by samples |
| Tensor norm | Mean Fourier-slice nuclear norms |

The mean and **sample** standard deviation (`std(values,0,1)`, denominator R-1) are computed from complete repetitions, without choosing the best run. A single requested repetition reports an undefined standard deviation, not a fabricated zero. A failed execution stops with a saved `failure.json`; it is never silently omitted from the statistics.

The retained optimizer can produce the same learned representation across training seeds. In particular, its randomly initialized anchors are replaced by the first Procrustes update before influencing the other primal updates; the initialization still appears in the first change diagnostic. Different seeds do not guarantee different solutions. Final clustering has its own recorded seeds. Zero empirical standard deviation is possible even when the complete pipeline is executed ten times. This protocol measures variability conditional on the supplied fixed missing-view pattern, not across newly sampled masks.

## Custom runs

```matlab
solver = struct('anchors',10,'alpha',0.03125,'beta',1, ...
                'seed',202610251,'maxIter',300);
experiment = struct('repetitions',10,'clusteringRestarts',20, ...
                    'outputDir',fullfile(pwd,'results','my_experiment'));
result = run_demo(fullfile('data','my_NGs_Per0.5.mat'),solver,experiment);
```

Use a new output directory each time. Training options are listed in `cdagtl_fit.m`; experiment options are `repetitions`, `clusteringRestarts`, and `outputDir`. `solver.seed` is the base seed, incremented once per complete execution.

The direct, label-free training API is:

```matlab
[model,history] = cdagtl_fit(data,observed,numClusters,solver);
```

Here `data{v}` is features by samples and `observed` is a samples-by-views logical matrix. Every sample and every view must have an observation. Ground-truth labels are used by the demo only to obtain the prescribed cluster count and to evaluate the final labels; they are not passed to `cdagtl_fit`.

The MAT-file interface expects `data`, `index`, and `truelabel`. `index{v}` contains observed sample indices or a logical observation mask. **Missing rate = 1 - Per** for the supplied benchmark filenames: `Per0.1` means missing rate 0.9. The nominal rate, fraction of samples with any missing view, and fraction of missing sample-view entries are recorded separately.

## Feature dimensions and weights

For each view, the normalized data are embedded with trailing zeros into working dimension `d_work(v)=max(d_original(v),m)`. Thus `A{v}` has `d_work(v)` rows and `m` orthonormal columns. Require `numClusters <= m <= numSamples`. A six-dimensional view can therefore be used with ten classes and twenty anchors, without removing the view.

Padding preserves the data Gram matrix, distances, and norms, but does not add information or increase data rank. The enlarged anchor parameterization can change reconstruction losses and fitted graphs. The original and working dimensions are recorded in `model.embedding`; padding is not claimed to reproduce the historical unpadded experiments.

The final clustering representation uses every left singular vector returned by `svd(fused','econ')`, where `fused` is the mean of the learned view graphs. This full economy-SVD step is an intentional exception to the current paper's Algorithm 1, retained to preserve the earlier clustering protocol. `run_demo` requires `m < numSamples`: at equality the full basis is square and orthogonal, with constant pairwise row distances. The direct training API accepts the mathematically feasible `m == numSamples` case, but its full representation should not be used for clustering then.

View weights solve `min sum_v w_v^2*t_v`, subject to `w >= 0` and `sum(w)=1`, where `t_v` is the squared reconstruction error **only on observed columns**:

* If any `t_v` is exactly zero, weight is distributed equally over the zero-error views and is zero elsewhere.
* Otherwise `q_v=min(t)/t_v`, followed by `w=q/sum(q)`, is algebraically equivalent to normalized inverse errors without forming potentially overflowing reciprocals.

These are explicit zero-error and small-error safeguards; no epsilon is inserted into the objective. Missing stored values are excluded from every reconstruction loss, including weight estimation.

## Solver updates and diagnostics

The update order is A, Z, J, F, C, D, weights, multipliers, penalty growth, then the stopping check. The standard mean-TNN proximal map, orthonormal spectral embedding, masked weight errors, and same-sweep penalty in primal and multiplier updates are retained. There is no additional core-tensor shrinkage.

For C, normalized distances `K` are computed using the old C's degrees and held fixed. One columnwise simplex projection solves the resulting convex surrogate:

```matlab
C = cdagtl_simplex(M - alpha/(mu*V)*K);
```

Here `M` is the mean of `Z{v}-D{v}+Pi{v}/mu`. The surrogate is `alpha*sum(K.*C,'all') + mu*V/2*norm(C-M,'fro')^2`. `history.cSurrogateChange` and `history.cSurrogateProjectedGradient` describe this frozen-distance problem. Surrogate descent does not imply descent of the original degree-dependent C objective or of the full objective.

Exactly zero anchor degrees use reciprocal square root zero; positive degrees are unchanged. Zero-degree anchor rows are allowed, and C is not perturbed to remove them. The original graph-gradient calculation, used only by the optional standalone `cdagtl_stationarity` diagnostic, requires strictly positive anchor degrees.

Training stops when `history.primalMax <= tolPrimal`, with `stopReason='primal_tolerance'`, or exhausts `maxIter`, with `stopReason='max_iterations'`. The maximum residual covers both `Z{v}-C-D{v}` and the tensor constraint `Z-J`. After updating the multipliers, `mu` grows by `muGrowth > 1`, capped at `muMax`, before this stopping check. Objective and `history.relativeChange` remain diagnostics; stationarity is not computed during training. A small primal residual certifies constraint satisfaction to the chosen tolerance, not stationarity or general convergence of nonconvex multi-block ADMM.

The sample-mode FFT is an algebraic prior. The standard tensor term couples view columns within each Fourier slice, but is not generally invariant to arbitrary sample permutations. No physical cyclic sample relationship is asserted. See `docs/REVISION_NOTES.md` for the distinction from historical experiments.

## Saved outputs and metrics

* `run_001.mat` through `run_010.mat`: complete model, history, labels, truth, mask, metrics and run metadata.
* `per_run.csv`: every repetition's seeds, timings, stop reason and seven metrics.
* `summary.csv`: mean and sample standard deviation of each metric.
* `protocol.json`, `progress.json`, `summary.json`, `summary.mat`: machine-readable protocol and results.

ACC uses an optimal one-to-one label assignment; NMI is `2*MI/(Htruth+Hpred)`; ARI is adjusted Rand index; F-score, precision and recall use unordered sample pairs. All seven metrics are computed from saved labels. The assignment solver and metric implementation require no external toolbox.

## Verification

```matlab
addpath('tests');
report = run_tests();
```

Tests cover padding and orthogonality, simplex feasibility, FFT-mean proximal scaling, positive-degree spectral gradients, frozen-C surrogate optimality, zero-degree continuation, primal-only stopping and penalty order, retained full-SVD representations, zero/tiny reconstruction-error weights, missing-value isolation, seed handling, known clustering metrics, exhaustive small assignment problems, and complete repeated executions with persisted models. Release validation records are stored in `validation/`; previous-version records are preserved in `validation/archive/1.0.0-revision/`. Large per-run models are generated locally under the ignored `results/` directory.

## Code availability and citation

Repository: https://github.com/whxyggj/CDAGTL

Please cite **Consistent and Diverse Anchor Graph Tensor Learning for Incomplete Multi-view Clustering** by Baicheng Pan, Yimin Fu, Hangjun Che, Aiping Lu, and Michael K. Ng. The manuscript is under revision; no publication DOI is asserted here. See `CITATION.cff` and the repository revision history for the exact code version.
