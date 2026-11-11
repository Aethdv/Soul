## Worth Trying.

### Soft WDL Targets

We take a position's target as a blend of the game result and `sigmoid(K * search_score)`.\
I'd swap that second half for `w + d/2` out of `wdl_model(search_score, material)`.

Single global `K` applies the same win-probability curve at full material as in a pawn endgame.\
And since WDL model's curve is a function of material, the target would have phase awareness for free and `K` stops doing a job it wasn't supposed to do.

### Chamfer Distance

Linear combination of Chebyshev (`w=0`) and Manhattan (`w=64`):\
`d(a,b,w) = 64 ⋅ max(Δf,Δr) + w ⋅ min(Δf,Δr)`

Power-of-two base so downstream normalization is just a shift (`>> 6`).\
I tried `round((w/100) * min)` in Clockwork a very long time ago, but at `min=1` it binned all of `w` into 0 or 1 and killed the asymmetry.\
Keeping the distance unrounded in scaled integer space fixes that.

A 1D Chebyshev table treats `(3,0)` and `(3,3)` as identical. The transpose-symmetric `(min, max)` triangle table tells them apart,
at 36 params per piece type (64 for a full asymmetric `(Δf,Δr)` table). Chamfer compresses the 36-entry triangle into a single knob.

The promising application I can see is within King tropism, where diagonal vs orthogonal map to how sliders attack the king.

Signing `w` per piece type works because the `w` term weights the diagonal part of the distance, which is `0` for every square a rook attacks.\
But the full `max(Δf,Δr)` for every square a bishop attacks. If tropism penalizes distance, `w > 0` makes `(3,3)` cost more than `(3,0)`,
rewarding rooks on the same file or rank. For bishops, `w < 0` makes `(3,3)` cheaper than `(3,0)`, folding diagonal alignment and proximity into the same parameter.

### Rational Winnable Gate (Smooth Endgame Scaling)

Stockfish Classical's `winnable` adjusts endgame scores via `eg + sign(eg) * max(complexity, -abs(eg))`, which is the soft-thresholding operator `sign(eg) * max(abs(eg) + complexity, 0)`.
Stockfish folks only got away with that formulation because they tuned HCE via SPSA on Fishtest. Inside an offline gradient tuner like Lion, that clamp has two structural flaws:

1. Whenever `complexity < -abs(eg)`, the clamp zeroes out the derivative with respect to both `eg` and the `complexity` weights.
   If a won endgame gets clamped to 0.00, the complexity weights receive zero gradient to ever un-clamp it.

2. Step discontinuity at `eg = 0` when `complexity > 0`: at `complexity = +40`, `eg = +1cp` jumps to +41cp while `eg = -1cp` jumps to -41cp,
   opening a `2 * complexity` chasm across zero. Above the clamp, `d(eg_out) / d(complexity)` is a flat `sign(eg)`, so a +1cp coinflip pushes
   the complexity weights just as hard as a +400cp position.

Both go away if `complexity` is split into non-negative drive `D >= 0` (pawns, both flanks, passed pawns, infiltration, horizontal king separation `Δfile`)
and positive draw friction `F > 0` (base bias, vertical king opposition `Δrank`, single-flank / OCB penalties), then gated rationally:

`eg_out = eg * (abs(eg) + D) / (abs(eg) + F)`

Rewriting the fraction makes the exact relation to Stockfish:\
`eg_out = eg + sign(eg) * (D - F) * (abs(eg) / (abs(eg) + F))`

Here `D - F` is Stockfish's `complexity`, and `abs(eg) / (abs(eg) + F)` replaces the hard `max()` clamp with a smooth hyperbolic gate.\
At large `|eg|`, it approaches the same additive shift `eg + sign(eg) * (D - F)`.\
Near `eg = 0`, it cleanly passes through the origin with slope `D / F` instead of jumping, cannot flip the sign of `eg`,\
and scales the parameter gradient by `d(eg_out)/dD = eg / (abs(eg) + F)` so near-equal positions stop corrupting the complexity weights.

### Rational Mobility

Standard mobility tables spend 66 parameters across N/B/R/Q, one per move-count bucket, just to approximate four concave curves,
leaving the rare high-mobility buckets in the tails starved for gradient.

I'd trade each piece's lookup table for a discrete Michaelis-Menten curve:\
`Bonus(moves) = Offset + (Span * moves) / (moves + K_s)`

`Span` sets the dynamic range from `moves=0` to the asymptote, and `K_s` sets the move count required to cover half that range.\
Every position updates the full curve, therefore no bucket should starve.

In case of an unshifted curve `(Span * moves) / (moves + K_s)` is pinned to `0` at `moves=0` and monotone, so it cannot represent
the negative penalties standard tables assign to trapped pieces.

Two ways to handle that:

1. When piece material values or uncentered PSQTs are tuned alongside mobility, `Offset` is strictly collinear with piece value and PSQT mean.
   In that regime, two parameters per piece (`Span`, `K_s`) should suffice and the material/PSQT mean should absorb the zero-move penalty.

2. When material and PSQT means are pinned, or if mobility needs to remain zero-centered in isolation for `eval` trace readability,
   `Offset` costs a third parameter per piece, 12 across N/B/R/Q.

### Soft Reliability Weighting

The volatility filter hard-drops `|static_eval - search_score| > volatility_threshold`.\
`volatility_adaptive` shifts the threshold by `(piece_count - 10) * 2`, floored at 0; once cut, the position is just gone.\
I'd try weighting the loss instead, with trust proportional to agreement: calm positions dominate the gradient, volatile ones whisper, and every position stays in.

### Search-Bootstrapped Targets

Tuning static eval to predict WDL from a static position optimizes for single-ply snapshots, whereas search consumes the eval ply to ply and wants consistency across transitions.

TD(λ) flavor: blend a position's target with the discounted score of the line that followed it in its own game, then refine in rounds.\
The loop is tune, regenerate scores with the new eval, retune. A controlled fixed point rather than in-loop feedback.

Game outcomes need to stay mixed in as the anchor so a stale search teacher cannot cap the ceiling, which is very important.

Fairly nichè known territory (TD-leaf, KnightCap).

### Dense Gauss-Newton / Levenberg-Marquardt (Over L-BFGS)

At a few hundred HCE parameters, the `N × N` Hessian is under a megabyte of memory. L-BFGS exists for the parameter counts where that doesn't fit in memory, we don't have this problem.

If I go second-order, I'd skip the L-BFGS history approximation and accumulate the exact Gauss-Newton matrix `H = Jᵀ⋅W⋅J` (with `W` the per-sample loss weights)
and gradient `g` in one streaming pass over the dataset, then solve `(H + λI) ⋅ Δθ = -g` directly via Cholesky.

Explicit ridge regularization `λI` (or `λ ⋅ diag(H)` for Marquardt) tuned on held-out loss replaces the implicit regularization that Lion gets from minibatch noise and EMA.

Two structural catches apply no matter the solver: `.trunc()` in the forward pass (`training.rs`, staircase documented in `scale.rs`) makes the true integer surface non-smooth, and tuning phase weights makes the MG/EG taper bilinear rather than convex. A second-order solve on the smooth float surrogate will still hit that quantization floor when weights are truncated back to integers.
