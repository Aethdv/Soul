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
