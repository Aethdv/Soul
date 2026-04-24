# Adding New Eval Terms

Step-by-step recipe for wiring a new linear eval term.

See [`EVAL_TUNER.md`](EVAL_TUNER.md) for the tuner architecture.

---

## What a term is

A term is a zero-sized type implementing `LinearTerm`:

- `fn apply<T: EvalMath>` — forward pass. Reads features and params, writes its contribution into `Accumulators`.
- `fn scatter` — backward pass. Consumes the combiner-derived upstream for its bucket and writes `∂loss/∂param` into the term's param slots.
- `type Upstream` — `TaperPair` for pre-tapered buckets (`bonus`, `mobility`); `f64` for scalar buckets (`king_safety`, `xray`).

The `register_terms!` macro in `src/engine/eval.rs` wires terms into `apply_all_terms` and `scatter_all_terms`.

Adding a term is one macro line plus the impl — no edits to the eval pipeline or to `tape.rs`.

---

## Linearity rule

`LinearTerm` assumes `∂bucket/∂param` is a pure feature coefficient: `bucket += param · feature_expr`. The feature expression can be any combination of features; the parameter must appear linearly.

| Pattern | Fast tape works? |
|---|---|
| `param · feature_count` | Yes |
| `param · (feature_a * feature_b)` | Yes — feature non-linearity is free |
| `param · (feature² / threshold)` | Yes |
| `param_a · param_b · feature` | **No** — parameter non-linearity, breaks scatter |
| `sigmoid(param_a + const) · feature` | **No** — param inside an activation |

Parameter non-linearity belongs in the `Combiner` (as a bucket-level activation) or a future `NonlinearTerm` escape hatch. The oracle test will catch drift if you cross the line.

---

## Recipe: bishop pair

A tapered bonus for holding both bishops. When white has the pair and black doesn't, add `mg_bonus · phase + eg_bonus · eg_phase` (and mirror for black).

### 1. Add the params to `define_tunables!`

In `src/engine/eval_params.rs`:

```rust
#[macro_export]
macro_rules! define_tunables {
    ($macro:ident) => {
        $macro! {
            // ...
            (w_bishop_pair_mg, Scalar, $crate::engine::eval_params::LAYOUT.bishop_pair_offset),
            (w_bishop_pair_eg, Scalar, $crate::engine::eval_params::LAYOUT.bishop_pair_offset + 1),
        }
    }
}
```

One entry feeds all three sites: the engine's `EvalParams<i32>`, the autograd `EvalParams<DualNode>`, and the tuner's f64 loader.

### 2. Extend `Layout`

Same file:

```rust
pub struct Layout {
    // ...
    pub xray_offset: usize,
    pub xray_len: usize,
    pub bishop_pair_offset: usize,
    pub bishop_pair_len: usize,
}

const fn calc_layout() -> Layout {
    // ...
    let xray_offset = king_safety_offset + safety_len;
    let xray_len = 1;
    let bishop_pair_offset = xray_offset + xray_len;
    let bishop_pair_len = 2;

    Layout {
        // ...
        xray_offset,
        xray_len,
        bishop_pair_offset,
        bishop_pair_len,
    }
}
```

Then update `EvalParams::<i32>::from_const()` in `src/engine/eval.rs` to load the compile-time defaults for the new params.

### 3. Write the term

Place it in a sensible file — new module `src/engine/terms/bishop_pair.rs`, or co-located if it belongs with existing logic (e.g. pawn terms beside other pawn code):

```rust
use crate::{
    core::{defs::TOTAL_PHASE, psqt::LAYOUT},
    engine::{
        autograd::EvalMath,
        combiner::Accumulators,
        eval::{EvalParams, SharedFeatures},
        term::{LinearTerm, TaperPair},
    },
};

pub struct BishopPairTerm;

impl LinearTerm for BishopPairTerm {
    type Upstream = TaperPair;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(
        features: &SharedFeatures,
        params: &EvalParams<T>,
        phase: T,
        acc: &mut Accumulators<T>,
    ) {
        let feature = T::from_i32(features.bishop_pair_diff);
        let eg_phase = T::from_i32(TOTAL_PHASE) - phase;
        let tapered = params.w_bishop_pair_mg * phase + params.w_bishop_pair_eg * eg_phase;
        acc.bonus += (tapered * feature / T::from_i32(TOTAL_PHASE)).trunc();
    }

    #[inline]
    fn scatter(features: &SharedFeatures, upstream: TaperPair, grads: &mut [f64]) {
        let feature = features.bishop_pair_diff as f64;
        grads[LAYOUT.bishop_pair_offset] += upstream.d_mg * feature;
        grads[LAYOUT.bishop_pair_offset + 1] += upstream.d_eg * feature;
    }
}
```

`apply` uses `+=` on `acc.bonus` because the bonus bucket is shared across every simple tapered term. `scatter` reads `upstream.d_mg` / `upstream.d_eg` directly — the combiner already folded in the phase fractions and loss derivative, so scatter is just `upstream · ∂bucket/∂param`.

### 4. Extend `SharedFeatures`

Both `apply` and `scatter` read `features.bishop_pair_diff`, so it lives on `SharedFeatures` (single source, computed once per position). In `src/engine/eval.rs`:

```rust
pub struct SharedFeatures {
    pub openness: i32,
    pub data: MobilityData,
    pub xray_ortho: i32,
    pub bishop_pair_diff: i32,
}

impl SharedFeatures {
    pub fn compute(board: &Position) -> Self {
        // ... existing extraction ...
        let bishop_pair_diff = pair_i32(board.pieces(PieceType::Bishop, Color::White))
                             - pair_i32(board.pieces(PieceType::Bishop, Color::Black));
        Self { /* ... */ bishop_pair_diff }
    }
}

fn pair_i32(bb: Bitboard) -> i32 { i32::from(bb.more_than_one()) }
```

### 5. Register the term

In `src/engine/eval.rs`:

```rust
crate::register_terms! {
    crate::engine::mobility::MobilityTerm => mobility,
    crate::engine::mobility::KingSafetyTerm => king_safety,
    crate::engine::terms::bishop_pair::BishopPairTerm => bonus,
    XrayTerm => xray,
}
```

The right-hand side names the `BucketUpstreams` field that feeds this term's `scatter`. A tapered bonus routes to `bonus`.

### 6. Verify

```sh
cargo test -p tuner --release oracle
```

The per-term oracle tests in `tuner/src/evaltune/tape.rs` run each `LinearTerm` in isolation. If your scatter disagrees with the `DualNode` AD oracle by more than `1e-3` on any fen, the matching test fails and names the term.

### 7. SPRT

Commit with a `Bench: XXXXX` trailer (CI and OB requires it). Then SPRT the branch against base :)

---

## Choosing the right bucket

For a pre-tapered, pure linear-sum term, route to `bonus`. No new bucket needed.

If the term has its own combiner shape — a separate activation, its own taper logic — then it earns a new bucket.<br>
Add a field to `Accumulators`, a matching field to `BucketUpstreams`, and update `LinearCombiner::forward` + `backward`.<br>
Bucket additions touch every combiner impl, so only add one when a term's combiner treatment is genuinely different.

Multi-bucket terms (one term writing several accumulator fields from shared features) already exist — `KingSafetyTerm` writes both `safety_us` and `safety_them`.
Scatter handles the per-side sign; `Upstream` is whatever the combiner produces for the shared block.

---

## Gradient cheat sheet

For a term routed to bucket `B` with upstream `U`, where `bucket += param · feature_expr(p)`:

```
grads[p] += U · feature_expr(p)
```

| Bucket routing | `Upstream` type | Scatter formula |
|---|---|---|
| `bonus` (shared tapered bonus) | `TaperPair { d_mg, d_eg }` | `grads[mg] += d_mg · feature`, `grads[eg] += d_eg · feature` |
| `mobility` (internal openness+phase blend) | `TaperPair` | Same shape, multiply by openness fraction — see `MobilityTerm::scatter` |
| `king_safety` (raw bucket, combiner tapers) | `f64` | `grads[p] += upstream · feature_diff` |
| `xray` (raw bucket, combiner tapers) | `f64` | `grads[xray] += upstream · feature` |

The combiner's `backward` has already folded in the loss derivative, STM sign, and taper. Scatter's only job is `upstream · ∂bucket/∂param`.

If the shape isn't on the table, derive `∂bucket/∂param` by hand from your `apply` formula. The oracle test catches mistakes — run it.
