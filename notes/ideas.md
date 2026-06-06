## Worth Trying.

### Eval-Derived Outcome Classification

Random-restart positions all get `result=0.5` right now — blend=1.0 needs to carry everything.
Classifying by WDL threshold instead:

```text
wdl model says ≥76% win  → win
               ≥76% loss → loss
               else      → draw
```

The WDL model scales by material; 100cp at 78 material is a nothingburger, same score at 17 material is crushing. Phase-aware for free, can't complain.
Then yolo `wdl_blend=0.5`.

Failure mode: Soul's eval is very often wrong at 16k soft nodes, so.. the WDL prediction is wrong.
But. At least, the threshold isn't fighting the material scaling problem on top of the eval accuracy problem.

Blocked on data; needs scored positions carrying the random-restarts, and the set on hand is stale outcomes-only.
`wdl.rs` (SF coeffs) reads sane on Soul's scale, so the axis isn't tilted — datagen's the only gate.

### Aeth Distance, or tunable anisotropic distance.

Convex blend of Chebyshev (w=0) and Manhattan (w=100):

`d(a,b,w) = max(Δf,Δr) + (w/100)·min(Δf,Δr)`

Parameterized octile distance (standard A* heuristic), not novel — there's just no name for the *tunable* form, hence the label.
Not Minkowski either (that curves via the exponent; this is linear).

Integer form to ship:

`d(a,b,w) = 100·max(Δf,Δr) + w·min(Δf,Δr)`

The round was the trap. At min=1, `round((w/100)·min)` bins all of w to 0 or 1, killing the asymmetry.
Scaling by 100 keeps it integer without rounding, and the tropism coefficient absorbs the scale.

A Chebyshev-indexed bonus array dominates it wherever params are free, and Aeth Distance's one edge is a single param that keeps the asymmetry a table throws away ((3,0) ≈ (3,3) to a table).

Real application is King tropism. Metric choice genuinely matters there; diagonal-vs-straight maps to how bishops/rooks reach the king, and one knob beats a 2D (Δf,Δr) table.

### Soft reliability weighting

The vol filter hard-drops `|static − search| > t` — the position's just gone.
Weight it instead: trust ∝ agreement, calm positions dominate and volatile ones whisper.
Same shape as the instance-confidence blend, moved off the target onto the gradient weight.
Uses all the data, one knob, one SPRT. Not novel — the thing HCE tuners skip.
Needs scored data; outcomes-only has nothing to disagree with.

### Search-bootstrapped targets

Tune eval to predict WDL from a static position, and search uses it ply to ply — different objective. Search wants consistency across plies, not just a calibrated win-prob.
Offline only — precompute scores once (the `score` field already is one), never a live search in the loop; that's the expense and the moving-target divergence both at once.

TD(λ) flavour: blend a position's target with the discounted score of the line that followed it in its own game.
Refine by rounds — tune, regen scores with the new eval, retune; a controlled fixed point, not in-loop feedback.

Game results are timeless; a search score is a teacher, and a stale teacher caps you right there. The result-blend is the anchor.
Known (TD-leaf, KnightCap), just nobody bothers in hand-crafted tuning.

### L-BFGS / second-order

488 HCE params, near-convex — tiny.
Full-batch L-BFGS lands a sharper optimum in tens of iterations where Lion takes thousands of noisy minibatch steps.
And the catch: `.trunc()` is non-smooth, the phase taper is bilinear not convex, and the SGD noise + EMA are doing implicit regularization a sharp fit throws away.
Better train loss, and maybe worse Elo. Measure.


## Already Poked At.

### MAGMA (Momentum-Aligned Gradient Masking)

*Ref: Joo et al. (2026) — On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.*  
<https://arxiv.org/abs/2602.15322v1>

Global cosine similarity between momentum and gradient vectors,
gating Lion's sign-update magnitude via `sigmoid(cossim / tau)` — suppressing updates when the optimizer as a whole oscillates induces implicit `Δᵀ·H·Δ` regularization toward flatter minima.

Tested: 430 HCE params, stable decay, batch_size 32768, tau 0.15, Big3 7.1M, 2000 epochs.  
Result: −0.67 ± 5.39 Elo (8240 SPRT games) — neutral, as expected in retrospect.
<https://asylum.red/test/4378/>

384 of 430 params are PSQT, so global cossim is basically just a PSQT proxy.
At low dimensionality, small groups collapse cossim into a near-binary switch anyway — the per-parameter disagreement gate already handles individual oscillators at the right granularity,
making this redundant at HCE scale.

Probably revisit at NNUE scale where per-group dimensionality is large enough for a meaningful collective alignment signal.
Per-group cossim, tau per group, group-selective application (skip D < 4).
