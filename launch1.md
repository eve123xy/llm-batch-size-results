# Batch-size repeat analysis

Engine lifetime: full_8b_awq_17132827_launch1; 12 repeats per N.

| N | Mean of run medians (ms) | Run-median CV (%) | Paired change vs N=1 (%) | Bootstrap 95% interval (%) | Identical outputs / pairs |
|---|---|---|---|---|---|
| 1 | 4.129154 | 1.3555 | +0.0000 | [+0.0000, +0.0000] | 12/12 |
| 2 | 4.170741 | 1.4287 | +1.0206 | [+0.7423, +1.4990] | 4/12 |
| 4 | 4.240548 | 1.4033 | +2.7403 | [+2.6572, +2.8228] | 6/12 |
| 8 | 4.361348 | 1.3101 | +5.6851 | [+5.5749, +5.8086] | 5/12 |
| 16 | 4.590785 | 1.4394 | +11.1325 | [+10.7745, +11.6904] | 4/12 |
| 32 | 5.360169 | 1.1499 | +29.7731 | [+29.5258, +30.0001] | 6/12 |

Intervals resample whole paired repeats within one engine lifetime, not individual tokens. They are exploratory, unadjusted for multiple comparisons, and do not measure between-job uncertainty. Balanced positions do not guarantee balanced carryover effects.

Scope: one H100, Llama-3.1-8B AWQ, 256 input / 512 output tokens, stable absolute positions 258–767. Target prompt and within-repeat seed are fixed across N; generated histories can diverge. This measures same-input/seed workloads, not an identical forced token trajectory.

All formal batches passed trace validation and showed zero audited capture/compile increments. Two full warmups per size precede each engine lifetime. Compare independent engine lifetimes separately; two starts on one allocation do not establish cross-node or cross-day variability.
