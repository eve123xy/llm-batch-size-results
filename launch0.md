# Batch-size repeat analysis

Engine lifetime: full_8b_awq_17132827_launch0; 12 repeats per N.

| N | Mean of run medians (ms) | Run-median CV (%) | Paired change vs N=1 (%) | Bootstrap 95% interval (%) | Identical outputs / pairs |
|---|---|---|---|---|---|
| 1 | 4.190213 | 0.0640 | +0.0000 | [+0.0000, +0.0000] | 12/12 |
| 2 | 4.226377 | 0.0518 | +0.8866 | [+0.8406, +0.9316] | 4/12 |
| 4 | 4.307086 | 0.0279 | +2.8906 | [+2.8493, +2.9246] | 5/12 |
| 8 | 4.428278 | 0.0698 | +5.7795 | [+5.7421, +5.8175] | 4/12 |
| 16 | 4.656266 | 0.0846 | +11.1064 | [+11.0078, +11.1871] | 5/12 |
| 32 | 5.433516 | 0.0892 | +29.5665 | [+29.4080, +29.7049] | 5/12 |

Intervals resample whole paired repeats within one engine lifetime, not individual tokens. They are exploratory, unadjusted for multiple comparisons, and do not measure between-job uncertainty. Balanced positions do not guarantee balanced carryover effects.

Scope: one H100, Llama-3.1-8B AWQ, 256 input / 512 output tokens, stable absolute positions 258–767. Target prompt and within-repeat seed are fixed across N; generated histories can diverge. This measures same-input/seed workloads, not an identical forced token trajectory.

All formal batches passed trace validation and showed zero audited capture/compile increments. Two full warmups per size precede each engine lifetime. Compare independent engine lifetimes separately; two starts on one allocation do not establish cross-node or cross-day variability.
