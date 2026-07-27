## Worth Trying.

### Soft WDL Targets

A position's target is a blend of the game result and `sigmoid(k·score)`.
Swap that second half for `w + d/2` out of `wdl_model(score, material)`.

One global K asserts 100cp means the same win probability at full material and in a pawn endgame.
It doesn't; the WDL model already scales by material, so the target gets phase awareness for free and K stops carrying a job it was never shaped for.

The soft form is the whole point. No threshold to place, every position enters at its own confidence,
and draws land near 0.5 on their own rather than being told where 0.5 sits.
Bucketing by a win-probability cut does the same thing worse: it throws away the confidence it just computed.

Needs entries carrying real search scores; an outcome-only set has nothing to convert.

Honest trim: the coefficients are Stockfish's, fitted on Stockfish's score scale.
They read sane on Soul's, and a refit on Soul's own results is the eventual right answer.
That refit pays three times over, since contempt and TM want the same curve.

### Aeth Distance, or tunable anisotropic distance.

Convex blend of Chebyshev (w=0) and Manhattan (w=100):

`d(a,b,w) = max(Δf,Δr) + (w/100)·min(Δf,Δr)`

Parameterized octile distance (standard A* heuristic), not novel, there's just no name for the *tunable* form, hence the label.
Not Minkowski either (that curves via the exponent; this is linear).

Integer form to ship:

`d(a,b,w) = 100·max(Δf,Δr) + w·min(Δf,Δr)`

The round was the trap. At min=1, `round((w/100)·min)` bins all of w to 0 or 1, killing the asymmetry.
Scaling by 100 keeps it integer without rounding, and the tropism coefficient absorbs the scale.

A Chebyshev-indexed bonus array dominates it wherever params are free, and Aeth Distance's one edge is a single param that keeps the asymmetry a table throws away ((3,0) ≈ (3,3) to a table).

Real application is King tropism. Metric choice genuinely matters there; diagonal-vs-straight maps to how bishops/rooks reach the king, and one knob beats a 2D (Δf,Δr) table.

### Soft reliability weighting

The vol filter hard-drops `|static − search| > t`, the position's just gone.
Weight it instead: trust ∝ agreement, calm positions dominate and volatile ones whisper.
Same shape as the instance-confidence blend, moved off the target onto the gradient weight.
Uses all the data, one knob, one SPRT. Not novel, the thing HCE tuners skip.
Needs scored data; outcomes-only has nothing to disagree with.

### Search-bootstrapped targets

Tune eval to predict WDL from a static position, and search uses it ply to ply; different objective. Search wants consistency across plies, not just a calibrated win-prob.
Offline only, precompute scores once (the `score` field already is one), never a live search in the loop; that's the expense and the moving-target divergence both at once.

TD(λ) flavour: blend a position's target with the discounted score of the line that followed it in its own game.
Refine by rounds; tune, regen scores with the new eval, retune; a controlled fixed point, not in-loop feedback.

Game results are timeless; a search score is a teacher, and a stale teacher caps you right there. The result-blend is the anchor.
Known (TD-leaf, KnightCap), just nobody bothers in hand-crafted tuning.

### L-BFGS / second-order

490 HCE params, near-convex, tiny.
Full-batch L-BFGS lands a sharper optimum in tens of iterations where Lion takes thousands of noisy minibatch steps.
And the catch: `.trunc()` is non-smooth, the phase taper is bilinear not convex, and the SGD noise + EMA are doing implicit regularization a sharp fit throws away.
Better train loss, and maybe worse Elo.


## Already Poked At.

### MAGMA (Momentum-Aligned Gradient Masking)

*Ref: Joo et al. (2026) — On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.*  
<https://arxiv.org/abs/2602.15322v1>

Global cosine similarity between momentum and gradient vectors,
gating Lion's sign-update magnitude via `sigmoid(cossim / tau)`; suppressing updates when the optimizer as a whole oscillates induces implicit `Δᵀ·H·Δ` regularization toward flatter minima.

Tested: 430 HCE params, stable decay, batch_size 32768, tau 0.15, Big3 7.1M, 2000 epochs.  
Result: −0.67 ± 5.39 Elo (8240 SPRT games) — neutral, as expected in retrospect.
<https://asylum.red/test/4378/>

384 of 430 params are PSQT, so global cossim is basically just a PSQT proxy.
At low dimensionality, small groups collapse cossim into a near-binary switch anyway; the per-parameter disagreement gate already handles individual oscillators at the right granularity,
making this redundant at HCE scale.

Probably revisit at NNUE scale where per-group dimensionality is large enough for a meaningful collective alignment signal.
Per-group cossim, tau per group, group-selective application (skip D < 4).

### Per-Coordinate Step Adaptation On The Gate We Already Have

The disagreement gate is half of iRprop⁻ and it kept the wrong half.

*Ref: Igel & Hüsken (2003) — Empirical Evaluation of the Improved Rprop Learning Algorithms. Neurocomputing 50, 105-123.*
<https://christian-igel.github.io/paper/EEotIRLA.pdf>

iRprop⁻ does two things when a coordinate's derivative reverses sign against the previous step: it skips that coordinate's update, and it shrinks that coordinate's step size by η⁻,
growing it by η⁺ when the signs agree instead. The skip is a side effect of how they implement it, zeroing the stored derivative so `sign(0) = 0` drops the weight update (§2.4, Fig. 3).
The step adaptation is the actual algorithm. We do the skip and never touch the step.

`lr_mask` is already per-parameter and already threaded through `update`, just static. Adapting it on `m · g`, the product the gate computes anyway, is a handful of lines.

Costs Lion's uniform update magnitude, which is the property the whole optimizer is built around; giving it up per-coordinate should be measured rather than argued.
Gate width already sets `‖Δθ‖₁`, so this compounds with the confound noted below.

η⁻ near 0.9, not the 0.5 the paper uses. That 0.5 is for noiseless backprop gradients; RSPSA runs 0.8 to 0.9 because its gradients are stochastic, and minibatch gradients sit between the two.
A 0.5 factor compounds a coordinate into immobility on a run of bad luck.

### Ablate The Disagreement Gate

Nobody has ever tested whether the gate helps.

Both cautious-mask runs (5761, 5762) tested *replacing* `m · g ≤ 0` with Liang's canonical `c · g ≤ 0`, a strictly wider skip. Neither compared the gate against no gate.
It’s been sitting in the baseline since before the mask experiments without ever being isolated.

The confound is the one already written in `lion.rs`: gate width sets `‖Δθ‖₁ = eff_lr · (total − skipped − dead)`, so a narrower gate is also a longer step,
and removing the gate entirely changes both at once. Pinning step length across the comparison means rescaling `lr` by the stepped fraction, the number `GateCensus` now reports.

Cheaper as an offline sweep than as an SPRT. Two tuner runs on identical seeds and data, gate on and off with `‖Δθ‖₁` pinned, compared on held-out loss first; SPRT only if the loss curves separate.
