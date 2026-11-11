# Lion's gate

`Lion::gate` decides per coordinate whether the sign step runs.
Here we describe what we tried instead and what we have not tested yet.

## Liang's cautious mask

Ref: Kaizhao Liang, Lizhang Chen, Bo Liu & Qiang Liu (2024).
Cautious Optimizers: Improving Training with One Line of Code.
<https://arxiv.org/abs/2411.16085v4>

Our gate skips the sign step on `m·g ≤ 0 && |m| > 1e-6`; weight decay runs either way.
Liang's canonical mask skips on `c·g ≤ 0`, which expands to `m·g ≤ -g²(1-β₁)/β₁`, or `-g²/9` at our default `β₁ = 0.9`.
That threshold is looser than `m·g ≤ 0`: Liang steps through mild reversals in `(-g²/9, 0]` that we skip.
A coordinate with `|m| ≤ 1e-6` takes the step regardless, because the skip needs `|m| > 1e-6`.

We tried Liang's mask at 490 HCE parameters across two retunes on separate seeds:

```text
Soul evaltune-sweep vs b58e13e770adfe9b4015af7ab738e11ff948b797
Elo    -6.24 ± 6.42
LLR    -2.54 (-2.47, 2.91) [0.00, 5.00]
Conf   8.0+0.08 16mb 1t
Games  5286 W: 1492 L: 1587 D: 2207
Penta  [167, 654, 1064, 623, 135]
```
https://asylum.red/tests/3197

```text
Soul evaltune-sweep vs b58e13e770adfe9b4015af7ab738e11ff948b797
Elo    -1.53 ± 4.13
LLR    -2.50 (-2.47, 2.91) [0.00, 5.00]
Conf   8.0+0.08 16mb 1t
Games  12734 W: 3658 L: 3714 D: 5362
Penta  [376, 1528, 2590, 1522, 351]
```
https://asylum.red/tests/3198

Liang also rescales the surviving update by `φ/mean(φ)`, which sets each active step to `lr·dim/nnz`.
We left that rescale out because it gives up Lion's uniform step size and couples every coordinate's 
update magnitude to how many other coordinates happened to fire.

Without the rescale, the gate's pass rate sets the step length `‖Δθ‖₁ = eff_lr · (total - skipped - dead)` directly, exact for a uniform `lr` mask.
A wider gate takes a longer step, so the two runs above moved step length and mask shape together.

A retry has to pin one of the two or it measures their sum. `GateTally` measures that gap.

## Magma's per-block scoring

Ref: Taejong Joo, Wenhan Xia, Cheolmin Kim, Ming Zhang & Eugene Ie (2026).
On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.
<https://arxiv.org/abs/2602.15322v1>

Magma skips whole blocks at random and scales each surviving block's update by an EMA of `sigmoid(cossim(m, g) / tau)`.\
The `Δᵀ·H·Δ` regularization toward flatter minima comes from the skipping.

We collapsed the blocks into a single global cosine similarity across all 430 HCE parameters
(stable decay, batch size 32768, `tau` 0.15, Big3 7.1M, 2000 epochs):

```text
Elo    -0.67 ± 5.39
LLR    -1.17 (-2.47, 2.91) [0.00, 5.00]
Conf   8.0+0.08s 16mb 1t
Games  8240 W: 2499 L: 2515 D: 3226
```

384 of those 430 parameters are PSQT, so a global cosine mostly gates the rest of the eval on PSQT alignment.
Splitting by HCE term does not help at this scale either: on a 1D scalar term, `cossim(m, g)` is just `sign(m·g) ∈ {-1, +1}`,
which reduces to the per-coordinate sign check we already run. From `D = 2` up the score depends on the angle between `m` and `g`, which the coordinate signs don't determine.

## Per-coordinate step adaptation

Ref: Christian Igel & Michael Hüsken (2003).
Empirical Evaluation of the Improved Rprop Learning Algorithms. Neurocomputing 50, 105-123.
<https://christian-igel.github.io/paper/EEotIRLA.pdf>

Our disagreement gate is half of iRprop⁻. When a coordinate's derivative reverses sign from the previous step,
iRprop⁻ shrinks that coordinate's step size by `η⁻` (growing it by `η⁺` when the signs agree)
and skips the update for that step. In the paper (§2.4, Fig. 3), the skip is a side effect of zeroing the current derivative
so `sign(0) = 0` drops the weight update. Adapting the step size is the core of the algorithm; we took the skip and left the step size fixed.

Wiring step adaptation itself is not a big deal: `lr_mask` is already per-parameter and threaded through `update`, just held constant, and `m·g` is already computed by the gate.
Doing so gives up Lion's uniform step magnitude per coordinate and changes `‖Δθ‖₁` over the run, so it has the same step-length confound as the mask comparisons.

If we try it, we should put `η⁻` near 0.9 rather than the paper's 0.5. That said 0.5 is for noiseless full-batch gradients.
RSPSA uses 0.8 to 0.9 because its gradients are stochastic, and minibatch gradients sit between the two.
Halving the step on every reversal would freeze a coordinate on a short run of noisy batches.

## Ablating the gate

I don't remember testing whether the gate helps at all.

Turning the gate off changes direction filtering and `‖Δθ‖₁` together, so a clean comparison has
to rescale `lr` by the stepped fraction from `GateTally` to hold step length fixed.

It's technically cheap to check; we want to make sure whether the loss curves will be separate.
