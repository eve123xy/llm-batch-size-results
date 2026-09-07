# Empirical batch-size model

R(N) = 1 + 0.00761684 (N-1) + 0.00378599 max(N-16, 0).

R is the mean across paired repeats and two engine lifetimes of the within-repeat median, over matched stable positions, of ITL(N)/ITL(1). It is not the ratio of overall mean ITLs. N=1 is exactly anchored at 1.

Statistical specification: Z[j,r,N] = a(N-1) + b max(N-16,0) + error[j,r,N], where Z is the paired relative increase. Equal-weight least squares; errors within a seed block may correlate across sizes and engine lifetimes and may be heteroskedastic. No iid-token assumption.

Coefficients: a=0.007616845, b=0.003785992. Conditional 95% paired-seed bootstrap intervals: a=[0.007497371, 0.007784860], b=[0.003518061, 0.003964001]. 10,000 resamples of 12 seed blocks, preserving both engines and all sizes together. These intervals condition on the observed engine lifetimes, model, and chosen knot; no model-selection or cross-job uncertainty is included.

| N | Observed increase (%) | Formula increase (%) | Observed minus formula (pp) |
|---|---|---|---|
| 2 | 0.9536 | 0.7617 | +0.1919 |
| 4 | 2.8155 | 2.2851 | +0.5304 |
| 8 | 5.7323 | 5.3318 | +0.4006 |
| 16 | 11.1195 | 11.4253 | -0.3058 |
| 32 | 29.6698 | 29.6698 | -0.0000 |

The fit RMSE is 0.3383 percentage points; maximum error is 0.5304 pp (about 0.52% of total ITL at measured sizes). Systematic residuals exceed repeat-mean uncertainty at several sizes. This is an engineering approximation, not a statistically adequate exact law.

The N=16 knot is a descriptive, post-data choice. Only N=32 lies above it, so the extra slope is identified by one size, and its shape/breakpoint is unvalidated. Leave-N=32-out hinge fitting is rank deficient; the minimum-norm cross-validation number in the comparison JSON must NOT be used to claim predictive validity. Cross-engine validation reuses the same six sizes and does not validate intermediate sizes.

For accurate summaries at measured sizes prefer the saturated categorical model E[Z|N]=theta_N using the observed increases above; a compact smooth/hinge formula sacrifices this fidelity. Linear interpolation between observed sizes is also an unvalidated interpolation assumption.

For approximate absolute latency use T(N) approximately T(1)*R(N), calibrating T(1) in the same run. The observed baseline means of run medians were 4.1902 and 4.1292 ms. A fixed 4.16 ms baseline is only a rough reference; do not conflate baseline variation with the paired effect.

Scope: N in {1,2,4,8,16,32}; one H100 allocation, two engine launches, Llama-3.1-8B AWQ, 256 input and 512 output tokens, stable positions 258–767. Same target prompt/seed does not ensure identical generated history. No inference to longer context/output, other hardware/models, N>32, power, or energy.

Candidate comparison (RMSE of five nonbaseline size means, percentage points): linear 1.3374; logarithmic 6.2058; quadratic 0.6147; power 0.8256; descriptive hinge 0.3383. Model comparison is descriptive, not a held-out confirmation of the selected shape.
