# Adding an Eval Term

Wiring a new linear eval term, end to end.
Architecture lives in [`EVAL_TUNER.md`](EVAL_TUNER.md).

---

## What a term is

A zero-sized type implementing `LinearTerm`:

- `apply<T: EvalMath>` — forward. Read features and params, write the contribution into `Accumulators`.
- `scatter` — backward. Take the combiner's upstream for the bucket, write `∂loss/∂param` into the term's slots.
- `type Upstream` — `TaperPair` for pre-tapered buckets (`bonus`, `mobility`), `f64` for the scalar ones (`king_safety`, `xray`).

`register_terms!` in `eval.rs` stitches every term into `apply_all_terms` / `scatter_all_terms`, which feed both the engine and the board-based tuner gradient off the same impl.

Most terms are a **tapered bonus** — a feature dotted with an `(mg, eg)` weight pair, summed into the `bonus` bucket.
Those never get a hand-written impl; they're one row of `tapered_bonus_term!`.
You hand-write a `LinearTerm` only for a **"novel shape"** — own bucket, own combiner activation, MG-only taper, multi-bucket output. Those are below.

The macros stop at the engine and the board path.
They do not reach the **cached SoA path** (`src/tools/dataset/gradient.rs`) — a hand-rolled mirror that packs features into `i8` so the epoch loop over `.soul.zst` data skips recomputing the spatial tensor.
It doesn't go through `LinearTerm`, so you mirror the term there yourself (Step 5).
Forget it and the term is right on raw EPD and silently wrong on encoded data — the exact bug the bishop-pair gap once hid.

---

## The linearity rule

`LinearTerm` assumes `∂bucket/∂param` is a pure feature coefficient: `bucket += param · feature_expr`. The feature expression can be anything; the *parameter* has to stay linear.

| Pattern                            | Fast tape holds?                               |
|------------------------------------|------------------------------------------------|
| `param · feature_count`            | Yes                                            |
| `param · (feature_a * feature_b)`  | Yes — feature non-linearity is free            |
| `param · (feature² / threshold)`   | Yes                                            |
| `param_a · param_b · feature`      | **No** — two params multiplied, scatter breaks |
| `sigmoid(param + const) · feature` | **No** — param inside an activation            |

Parameter non-linearity belongs in the `Combiner` as a bucket-level activation, or a future `NonlinearTerm`. Cross the line and the oracle catches you.

---

## Recipe: a tapered bonus (bishop pair)

A bonus for the pair: white holds both bishops and black doesn't, add `mg · phase + eg · eg_phase`, mirrored for black. Scalar case — one feature, one weight pair.

### 1. The feature → `SharedFeatures`

Computed once per position, read by every consumer. In `eval.rs`:

```rust
pub struct SharedFeatures {
    // ...
    pub bishop_pair_diff: i32, // +1 white pair, −1 black pair, 0 neither
}

impl SharedFeatures {
    pub fn compute(board: &Position) -> Self {
        // ... existing extraction ...
        let w_pair = i32::from(board.pieces(PieceType::Bishop, Color::White).more_than_one());
        let b_pair = i32::from(board.pieces(PieceType::Bishop, Color::Black).more_than_one());
        let bishop_pair_diff = w_pair - b_pair;
        Self { /* ... */ bishop_pair_diff }
    }
}
```

`compute` is White-relative. The cached path flips to STM in Step 5 — don't bake the sign in here.

### 2. The weights and the slot map

All in `eval_params.rs`, one line each:

```rust
// defaults
define_weight_params! {
    // ...
    BISHOP_PAIR_WEIGHTS = [V(33), V(85)], // [MG, EG]
}

// slot map — the Layout struct and prefix-sum offsets are generated; order IS the map
define_layout! {
    // ...
    bishop_pair = BISHOP_PAIR_WEIGHTS.len(),
}

// typed EvalParams fields — one row feeds engine i32, tuner f64, and oracle DualNode
macro_rules! define_tunables {
    ($macro:ident) => { $macro! {
        // ...
        (w_bp_mg, Scalar, bishop_pair_offset, 0),
        (w_bp_eg, Scalar, bishop_pair_offset, 1),
    }};
}
```

Then the compile-time defaults in `EvalParams::<i32>::from_const()` (`eval.rs`):

```rust
w_bp_mg: BISHOP_PAIR_WEIGHTS[0],
w_bp_eg: BISHOP_PAIR_WEIGHTS[1],
```

### 3. The term — one macro row

No hand-written `apply`/`scatter` for a bonus. Scalar feature on a contiguous `(mg, eg)` slot:

```rust
tapered_bonus_term! {
    BishopPairTerm = scalar(bishop_pair_diff, w_bp_mg, w_bp_eg, bishop_pair_offset);
    // array form — N buckets, separate MG/EG blocks:
    // PassedPawnTerm = array(passed_pawn, passed_mg, passed_eg, passed_mg_offset, passed_eg_offset, 6);
}
```

Declare the marker struct with its peers, not buried in the macro:

```rust
/// Tapered bonus for holding both bishops (~9 Elo).
pub struct BishopPairTerm;
```

### 4. Register it

```rust
crate::register_terms! {
    // ...
    BishopPairTerm => bonus,
}
```

The right side is the `BucketUpstreams` field that feeds this term's `scatter`. A tapered bonus routes to `bonus`.

### 5. Mirror the cached path

`gradient.rs` packs features into `i8` at startup and runs its own forward/backward over the bytes — it never calls your `LinearTerm`, so mirror it or `.soul.zst` training is wrong:

- `FeatureSlots`: a `pub bishop_pair: Vec<i8>` field and its `with_capacity` line.
- `push_entry`: pack from the `SharedFeatures` value with the STM flip — `self.bishop_pair.push((sf.bishop_pair_diff * sign) as i8)`.
                A white-minus-black diff negates for Black; a side-symmetric metric swaps halves.
- `eval_soul_cached`: the forward contribution, `bp · (mg·mg_w + eg·eg_w)`, truncated.
- `accumulate_gradient_cached`: the scatter — `grads[bp_offset] += gradient · bp · mg_w`, `+1` for eg.

### 6. Oracle test, then run it

Add a per-term test in `tape.rs` — a `term_for` arm, a `test_*_term_oracle`, and a fen that imbalances the feature (without one the term is never exercised, which is how the gap stays invisible).
Then:

```sh
make oracle
```

The per-term tests run each `LinearTerm` against the `DualNode` oracle; the encoded test checks the cached path. Drift past `1e-3` on any fen fails and names the term.

### 7. Bench, then SPRT

`make bench`, commit with the `Bench: XXXXX` trailer (CI and OB demand it), SPRT against base. :)

---

## Novel shapes

When the term isn't a plain tapered bonus — its own bucket, a combiner activation, MG-only taper, multiple output buckets — skip the macro and hand-write the `LinearTerm`.

The live examples:

- **`XrayTerm`** (`eval.rs`) — scalar, MG-only, `f64` upstream, writes the `xray` bucket. The smallest hand-written term, and the one to copy.
- **`KingSafetyTerm`** (`mobility.rs`) — one feature pass, two buckets (`safety_us`, `safety_them`); attacker weights indexed by attacker count.
- **`MobilityTerm`** (`mobility.rs`) — interpolates the open and closed weight vectors by openness, then tapers.

Steps 1, 2, 4, 5, 6, 7 are unchanged. Only Step 3 becomes the hand-written impl instead of a macro row.

---

## Choosing the bucket

Pre-tapered, pure linear sum → `bonus`. No new bucket.

A term earns its own bucket only when its combiner treatment is genuinely different — a separate activation, its own taper.
That costs a field on `Accumulators`, a matching field on `BucketUpstreams`, and an edit to `LinearCombiner::forward` + `backward`.
Bucket additions touch every combiner impl, so add one deliberately.

Multi-bucket terms already exist: `KingSafetyTerm` writes `safety_us` and `safety_them` from shared features, and scatter carries the per-side sign.
`Upstream` is whatever the combiner hands the shared block.

---

## Gradient cheat sheet

A term on bucket `B` with upstream `U`, where `bucket += param · feature_expr(p)`:

```
grads[p] += U · feature_expr(p)
```

|               Bucket              |         `Upstream`         |                                Scatter                                |
|-----------------------------------|----------------------------|-----------------------------------------------------------------------|
| `bonus` (tapered bonus)           | `TaperPair { d_mg, d_eg }` | `grads[mg] += d_mg · feature`, `grads[eg] += d_eg · feature`          |
| `mobility` (openness+phase blend) | `TaperPair`                | same shape, times the openness fraction — see `MobilityTerm::scatter` |
| `king_safety` (combiner tapers)   | `f64`                      | `grads[p] += upstream · feature_diff`                                 |
| `xray` (combiner tapers)          | `f64`                      | `grads[xray] += upstream · feature`                                   |

The combiner's `backward` already folded in the loss derivative, the STM sign, and the taper, so scatter is only `upstream · ∂bucket/∂param`.
A `tapered_bonus_term!` row generates all of it — the table matters when you hand-write a novel shape.
If your shape isn't here, derive `∂bucket/∂param` from your `apply` by hand; the oracle catches the slips.
