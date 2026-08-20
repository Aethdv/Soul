# Adding an Eval Term

Wiring a new linear eval term, end to end.
Architecture is in [`EVAL_TUNER.md`](EVAL_TUNER.md).

---

## What a term is

A zero-sized type implementing `LinearTerm`:

- `apply<T: EvalMath>` reads `SharedFeatures` and `EvalParams` from a live board, writing into `Accumulators`.
- `apply_input` writes the same buckets from a packed `FeatureRecord`, which has no board and no `EvalParams`.
- `scatter` takes the combiner's upstream for the bucket and writes `∂loss/∂param` into the term's slots.
- `type Upstream` is whatever the combiner hands that bucket: `TaperPair` for the pre-tapered ones,
  `f64` for `xray`, `KingSafetyUpstream` for king safety, which holds the shelter derivative plus one
  danger derivative per side.

Nearly every term is a tapered bonus, a feature dotted with an `(mg, eg)` pair into the `bonus`
bucket, and those are one roster row with no hand-written impl. You write a `LinearTerm` yourself
only for a novel shape: its own bucket, its own activation, an MG-only taper, multiple outputs.

## The linearity rule

`LinearTerm` assumes `∂bucket/∂param` is a pure feature coefficient, `bucket += param · feature_expr`.
The feature expression can be anything; the *parameter* has to stay linear.

| Pattern | Holds? | Why |
|---|---|---|
| `param · feature_count` | Yes | The parameter is linear. |
| `param · (feat_a · feat_b)` | Yes | Feature non-linearity is free. |
| `param · (feat² / scale)` | Yes | Arithmetic the parameter never enters. |
| `TABLE[feature_index]` | Yes | A feature choosing an index is still `param · 1`. |
| `param_a · param_b · feat` | No | Two parameters multiplied; the scatter has nowhere to put it. |
| `sigmoid(param + c) · feat` | No | Parameter inside an activation. |

One shape did cross the line, and where it went is where the next one goes. King danger accelerates,
`pressure + pressure²·curvature/DANGER_SCALE`, and `pressure` is itself a sum of weighted features,
so the square multiplies parameters together and `curvature` is a tunable on top. It lives in
`LinearCombiner::forward`, not in `KingSafetyTerm`: the term feeds `danger_us` and `danger_them` raw,
the combiner curves each side after the sum, and `backward` folds the slope
`1 + 2p·curvature/DANGER_SCALE` into the term's upstream. Leave it in a term and the oracle catches
you, but only if a test position exercises it.

The gauge that holds the eval's overall scale multiplies every slot by one factor, which is right
only where the score is degree one in that parameter. `curvature` multiplies `pressure²`, so
`Gauge::slot_scale` hands it the reciprocal, and the next non-linear parameter needs its own arm
there. Nothing breaks loudly when it is missing; the gauge only stops being a rescale.

---

## Recipe: a tapered bonus

`bishop_pair`, the scalar case. Five edits and a verification.

### 1. The feature, in `eval.rs`

Computed once per position into `SharedFeatures`, read by every consumer.

```rust
pub struct SharedFeatures {
    /// +1 / 0 / -1 depending on side holding >= 2 bishops.
    pub bishop_pair_diff: i32,
}

impl SharedFeatures {
    pub fn compute(board: &Position) -> Self {
        let w_pair = i32::from(board.pieces(PieceType::Bishop, Color::White).more_than_one());
        let b_pair = i32::from(board.pieces(PieceType::Bishop, Color::Black).more_than_one());
        let bishop_pair_diff = w_pair - b_pair;

        Self { /* ... */ bishop_pair_diff }
    }
}
```

Always White-relative. The side-to-move flip happens at packing.

### 2. The weights, in `eval_params.rs`

```rust
define_weight_params! {
    bishop_pair = [V(40), V(74)], // [MG, EG]
}
```

The paste-block test reproduces this block byte for byte, so grouping and trailing comments are part
of the contract.

### 3. The roster row, in `eval.rs`

```rust
bonus_terms! {
    bishop_pair = scalar(BishopPairTerm, bishop_pair_diff, bishop_pair_mg, bishop_pair_eg); // ~9 Elo
    passed_pawn = array(PassedPawnTerm, passed_pawn, passed_pawn_mg, passed_pawn_eg, 6);    // ~15 Elo
}
```

Block name, term type, the `SharedFeatures` field it reads, the two `EvalParams` fields it
multiplies, and for an array its width. Offsets are not on the row: `paste` emits
`bishop_pair_offset` from the block name at each use site.

That row writes the term struct and its `LinearTerm` impl, both `TermSource` bridges, the record's
packing, the `register_terms!` entry, both `define_tunables!` rows with the `EvalParams` fields and
their loaders, the `define_layout!` block name, and the oracle test binding.

### 4. The packed field, in `gradient.rs`

```rust
pub struct FeatureRecord {
    pub bishop_pair_diff: i8,
}
```

**Name it exactly as the `SharedFeatures` field.** One roster column spells both sides of the copy,
`$rec.$field = ($sf.$field * $sign) as i8`, so a mismatch is a build error rather than a silent
mispack.

That `$sign` is the side-to-move flip, and a bare negation covers it only because every roster
feature is white-minus-black. A per-side metric has none to flip; mobility and safety swap their
us/them halves, packed by hand in `from_entry` before the roster's arm runs. No roster row is
per-side today, and if yours is, it is a novel shape before it is a packing problem.

### 5. The annotation, in `evaltuner/src/report.rs`

```rust
("bishop_pair", "[MG, EG]"),
```

The paste block's trailing comment, keyed by block name. Comments are not tokens, so this is the one
thing no macro can reach. A stale entry fails `the_paste_block_reproduces_eval_params` rather than
misprinting quietly.

### 6. Oracle, bench, SPRT

```sh
make oracle
make bench
```

`test_bonus_terms_oracle` runs each roster term alone against the `DualNode` oracle, so a new row is
covered the moment it exists. `test_encoded_block_coverage_oracle` bumps every block and names any
whose slots never move the cached eval, which is what catches a missed record field. Add a FEN to
`FENS` in `tape.rs` if none of the existing ones imbalance your feature; a term that never fires is a
term the oracle cannot check.

Then commit with the `Bench: XXXXX` trailer and SPRT against base.

---

## Novel shapes

When the term is not a plain tapered bonus, everything the roster wrote comes back by hand.

- **`XrayTerm`** in `eval.rs` is the smallest and the one to copy: scalar, MG-only, `f64` upstream,
  one bucket.
- **`KingSafetyTerm`** in `mobility.rs` writes four buckets from one feature pass, with the attacker
  weight indexed by attacker count.
- **`MobilityTerm`** in `mobility.rs` interpolates the open and closed weight vectors by openness,
  then tapers.

Steps 1, 2, 4 and 5 are unchanged. In place of step 3 you write the `LinearTerm` impl, both
`TermSource` bridges, a `register_terms!` line inside `register_bonus!`, the `define_tunables!` rows
each naming the const its field loads from, and the block name in `define_layout!`'s core list in
slot order.

A hand-written term says the same thing three times and the three have to agree. `KingSafetyTerm`'s
attacker weight is `weak / 10` inside `SideMetrics::pressure`, which `apply` and `apply_input` both
call, and again as a bare `/ 10.0` in `scatter`, because a derivative cannot go through `pressure`.
Change the divisor in one place and the forward pass and the gradient disagree, silently, on every
position with a king attacker.

---

## Gradient cheat sheet

The combiner's `backward` has already folded in the loss derivative, the side-to-move sign and the
taper, so scatter is only `upstream · ∂bucket/∂param`.

| Bucket | `Upstream` | Scatter |
|---|---|---|
| `bonus` | `TaperPair { d_mg, d_eg }` | `grads[mg] += d_mg · feat`, `grads[eg] += d_eg · feat` |
| `mobility` | `TaperPair` | the same, times the openness fraction |
| `xray` | `f64` | `grads[xray] += upstream · feat` |
| `king_safety` | `KingSafetyUpstream` | shelter slots take `shelter · feat_diff`; each side's attacker weight takes its own `danger_us` / `danger_them` |

The danger halves arrive separately because the combiner curves each before differencing them, so
`∂block/∂danger` depends on which king it belongs to.

A roster row generates all of this. The table matters when you hand-write a novel shape, and if your
shape is not here, derive `∂bucket/∂param` from your `apply`; the oracle catches the slips.
