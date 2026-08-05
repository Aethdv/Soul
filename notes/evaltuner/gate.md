# Lion's gate

`Lion::gate` decides, per coordinate, whether the sign step fires. Everything below is what
was tried instead and what it cost. The mechanism itself lives on the function.

## Liang's cautious mask

Ref: Kaizhao Liang, Lizhang Chen, Bo Liu & Qiang Liu (2024).
Cautious Optimizers: Improving Training with One Line of Code.
<https://arxiv.org/abs/2411.16085v4>

Ours withholds the sign step on `m·g ≤ 0 && |m| > 1e-6`; decay fires either way. Liang's
canonical mask withholds on c·g ≤ 0, which expands to m·g ≤ -g²(1-β₁)/β₁, so -g²/9 at our
default β₁ = 0.9: a subset of ours on the m·g comparison, so it steps on reversals we sit out,
while our epsilon steps where it would not. 

Attempted at 490 HCE parameters, two retunes on separate seeds:

```text
  Elo   | -6.24 ± 6.43 (95%)
  SPRT  | 8.0+0.08s Threads=1 Hash=16MB
  LLR   | -2.54 (-2.47, 2.91) [0.00, 5.00]
  Games | N: 5286 W: 1492 L: 1587 D: 2207
  https://asylum.red/test/5761/

  Elo   | -1.53 ± 4.13 (95%)
  SPRT  | 8.0+0.08s Threads=1 Hash=16MB
  LLR   | -2.50 (-2.47, 2.91) [0.00, 5.00]
  Games | N: 12734 W: 3658 L: 3714 D: 5362
  https://asylum.red/test/5762/
```

Liang pairs the mask with a φ/mean(φ) rescale, which we leave out: it would set the surviving step
to lr·dim/nnz, forfeiting the uniform magnitude Lion is built on and pricing every coordinate off
a global statistic.

Leaving it out is not free. Gate width sets ‖Δθ‖₁ directly, so a wider gate is also a longer step,
and the two runs above differ in step length as well as in mask shape. Any retry pins one of the
two, or it buys another confounded result. `GateCensus` exists to price that: `band` and
`canonical_only` are the two directions in which the gates disagree, and the counts are the
step length.

## Magma's per-block scoring

Ref: Taejong Joo, Wenhan Xia, Cheolmin Kim, Ming Zhang & Eugene Ie (2026).
On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.
<https://arxiv.org/abs/2602.15322v1>

Magma scores per parameter block. Ours collapsed that to one global cossim over 430 HCE
parameters, 384 of them PSQT, which set the gate for everything else.

```text
  Elo   | -0.67 ± 5.39 (95%)
  SPRT  | 8.0+0.08s Threads=1 Hash=16MB
  LLR   | -1.17 (-2.47, 2.91) [0.00, 5.00]
  Games | N: 8240 W: 2499 L: 2515 D: 3226
  https://asylum.red/test/4378/
```

Worth revisiting per group, with more HCE terms or at NNUE scale.
