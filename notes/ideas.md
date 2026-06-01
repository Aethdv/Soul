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

### Aeth Distance, or tunable anisotropic distance.

Convex blend of Chebyshev (w=0) and Manhattan (w=100):

`d(a,b,w) = max(Δf,Δr) + (w/100)·min(Δf,Δr)`

Parameterized octile distance (standard A* heuristic), not novel — there's just no name for the *tunable* form, hence the label.
Not Minkowski either (that curves via the exponent; this is linear).

Integer form to ship:

`d(a,b,w) = 100·max(Δf,Δr) + w·min(Δf,Δr)`

The round was the trap. At min=1, `round((w/100)·min)` bins all of w to 0 or 1, killing the asymmetry. Scaling by 100 keeps it integer without rounding, and the tropism coefficient absorbs the scale.

A Chebyshev-indexed bonus array dominates it wherever params are free, and Aeth Distance's one edge is a single param that keeps the asymmetry a table throws away ((3,0) ≈ (3,3) to a table).

Real application is King tropism. Metric choice genuinely matters there; diagonal-vs-straight maps to how bishops/rooks reach the king, and one knob beats a 2D (Δf,Δr) table.


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
