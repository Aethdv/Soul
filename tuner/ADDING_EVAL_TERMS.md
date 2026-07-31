# Adding an Eval Term

Wiring a new linear eval term, end to end.
Architecture lives in [`EVAL_TUNER.md`](EVAL_TUNER.md).

---

## What a term is

A zero-sized type implementing `LinearTerm`:

- `apply<T: EvalMath>`: forward from a board. Read features and params, write the contribution into `Accumulators`.
- `apply_input`: the same buckets from extracted features, for the packed record, which has no board and no `EvalParams`.
- `scatter`: backward. Take the combiner's upstream for the bucket, write `∂loss/∂param` into the term's slots.
- `type Upstream`: whatever the combiner hands that bucket. `TaperPair` for the pre-tapered ones (`bonus`, `mobility`), plain `f64` for `xray`, and `KingSafetyUpstream` for king safety, which carries the shelter derivative plus one danger derivative per side.

Most terms are a **tapered bonus**: a feature dotted with an `(mg, eg)` weight pair, summed into the `bonus` bucket.
Those never get a hand-written impl; they are one row of the roster.
You hand-write a `LinearTerm` only for a **novel shape**: own bucket, own combiner activation, MG-only taper, multi-bucket output. Those are further down.

---

## The roster

`bonus_terms!` in `eval.rs` holds one row per tapered bonus:

```rust
bishop_pair = scalar(BishopPairTerm, bishop_pair_diff, bishop_pair_mg, bishop_pair_eg); // ~9 Elo
phalanx     = array(PhalanxTerm, phalanx, phalanx_mg, phalanx_eg, 6);                   // ~5 Elo
```

Block name, term type, the `SharedFeatures` field it reads, the two `EvalParams` fields it multiplies, and for an array its width.
Offsets are not on the row: `paste` emits `bishop_pair_offset` and `phalanx_mg_offset` from the block name at each use site.

From that one row the build writes:

- the term struct and its `LinearTerm` impl
- the `TermSource` bridges, from a board and from the packed record
- the record's packing, and the `register_terms!` entry
- both `define_tunables!` rows, the `EvalParams` fields, `load_tunable` and `from_const`
- the `define_layout!` block name, and the oracle test

---

## The linearity rule

`LinearTerm` assumes `∂bucket/∂param` is a pure feature coefficient: `bucket += param · feature_expr`. The feature expression can be anything; the *parameter* has to stay linear.

|              Pattern               |               Direct path holds?              |
|------------------------------------|-----------------------------------------------|
| `param · feature_count`            | Yes                                           |
| `param · (feature_a * feature_b)`  | Yes, feature non-linearity is free            |
| `param · (feature² / threshold)`   | Yes                                           |
| `param_a · param_b · feature`      | **No**, two params multiplied, scatter breaks |
| `sigmoid(param + const) · feature` | **No**, param inside an activation            |

Two live shapes look like they cross and don't.
`ATTACKER[count]` reads a different weight per attacker count, which is a feature choosing an index; given the index it is still `param · 1`, and scatter writes to the slot the count selected.
The `feature² / threshold` row is the same thing from the other side: arithmetic the parameter never enters.

One shape did cross, and where it ended up is where the next one goes.
King danger accelerates, `pressure + pressure² · curvature / DANGER_SCALE`, and `pressure` is a sum of weighted features, so the square alone multiplies parameters together; `curvature` is a tunable on top of that.
It sits in `LinearCombiner::forward`, not in `KingSafetyTerm`: the term feeds `danger_us` and `danger_them` raw, the combiner curves each side after the sum, and `backward` folds the slope `1 + 2p · curvature / DANGER_SCALE` into the term's upstream.
Anything else with a parameter inside an activation goes the same way, into the combiner or into a future `NonlinearTerm`.
Leave it in a term and the oracle catches you.

---

## Recipe: a tapered bonus (bishop pair)

A bonus for the pair: white holds both bishops and black doesn't, add `mg · phase + eg · eg_phase`, mirrored for black. Scalar case: one feature, one weight pair.

Five edits, in the order they make sense, plus verification.

### 1. The feature → `SharedFeatures` (`eval.rs`)

Computed once per position, read by every consumer. The field, the math, and the struct literal that returns it:

```rust
pub struct SharedFeatures {
    // ...
    /// +1/0/−1 per side's `more_than_one()`.
    pub bishop_pair_diff: i32,
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

`compute` is White-relative. The cached path flips to STM in step 4; don't bake the sign in here.

### 2. The values (`eval_params.rs`)

One row. The block name generates the const, `bishop_pair` → `BISHOP_PAIR`:

```rust
define_weight_params! {
    // ...
    bishop_pair = [V(39), V(72)], // [MG, EG]
}
```

This block is machine-written: `write_params` renders it and `the_paste_block_reproduces_eval_params` diffs the render against the file, so the trailing comment has to match what the printer emits.
No `define_layout!` row, no `define_tunables!` rows, no `from_const` line. The roster supplies all three.

### 3. The roster row (`eval.rs`)

```rust
bonus_terms! {
    // ...
    bishop_pair = scalar(BishopPairTerm, bishop_pair_diff, bishop_pair_mg, bishop_pair_eg); // ~9 Elo
}
```

Array form for a bucketed feature, MG and EG living in separate blocks:

```rust
passed_pawn = array(PassedPawnTerm, passed_pawn, passed_pawn_mg, passed_pawn_eg, 6); // ~15 Elo
```

The width is matched as a literal, so a term at any width other than 6 has no arm, and the build stops on that row.
That is deliberate: a captured width would let a term's array run past its own block into the next one, and nothing else checks it.

### 4. The record field (`gradient.rs`)

`FeatureRecord` packs features into `i8` so the epoch loop never recomputes the spatial tensor.
The packing and both `TermSource` bridges come from the roster; the field does not, because `macro_rules!` cannot expand in struct-field position:

```rust
pub struct FeatureRecord {
    // ...
    pub bishop_pair_diff: i8,
}
```

**Name it exactly as the `SharedFeatures` field.** One roster column spells both sides of the copy, `$rec.$field = ($sf.$field * $sign) as i8`, so a mismatch is a build error rather than a silent mispack.
The sign is the STM flip, and a bare negation covers it only because every roster feature is white-minus-black.
A per-side metric has no sign to flip; mobility and safety swap their us/them halves instead, packed by hand in `from_entry` before the roster's arm ever runs.
No roster row is per-side, and if yours is, it is a novel shape before it is a packing problem.

### 5. The annotation (`tuner/src/evaltune/report.rs`)

The paste block's trailing comment, keyed by block name:

```rust
("bishop_pair", "[MG, EG]"),
```

Comments are not tokens, so no macro can carry this one. A stale entry fails `the_paste_block_reproduces_eval_params` rather than misprinting quietly.

### 6. Oracle, bench, SPRT

```sh
make oracle
```

`test_bonus_terms_oracle` walks the roster and runs each term alone against the `DualNode` oracle, so a new row is covered the moment it exists.
`test_encoded_block_coverage_oracle` bumps every block by 100 and fails naming any whose slots never move the cached eval, which is what catches a missed record field.
Add a FEN to `FENS` in `tape.rs` if none of the existing ones imbalance your feature; a term that never fires is a term the oracle cannot check.

Then `make bench`, commit with the `Bench: XXXXX` trailer, and SPRT against base.

---

## Choosing the bucket

Pre-tapered, pure linear sum → `bonus`. No new bucket.

A term earns its own bucket only when its combiner treatment is genuinely different: a separate activation, its own taper.
That costs a field on `Accumulators`, a matching field on `BucketUpstreams`, and an edit to both halves of `LinearCombiner`, which is the only `Combiner` in the tree and the one every path collapses through.

Multi-bucket terms exist: `KingSafetyTerm` writes four, and the per-side sign rides in on the upstream, so its scatter adds both halves the same way.
`Upstream` is whatever the combiner hands the shared block.

---

## Novel shapes

When the term isn't a plain tapered bonus, it leaves the roster and everything the roster wrote comes back by hand.

The live examples:

- **`XrayTerm`** (`eval.rs`): scalar, MG-only, `f64` upstream, writes the `xray` bucket. The smallest hand-written term, and the one to copy.
- **`KingSafetyTerm`** (`mobility.rs`): one feature pass, four buckets. Shelter into `safety_us` / `safety_them`, attacker pressure into `danger_us` / `danger_them` for the combiner to curve; the attacker weight is indexed by attacker count.
- **`MobilityTerm`** (`mobility.rs`): interpolates the open and closed weight vectors by openness, then tapers.

Steps 1, 2, 4 and 5 are unchanged. In place of step 3 you write:

- the `LinearTerm` impl, with `apply`, `apply_input` and `scatter`,
- `TermSource<YourTerm> for SharedFeatures` and `for FeatureRecord`,
- a `register_terms!` line inside `register_bonus!`, beside the ones already there,
- the `define_tunables!` rows in `eval_params.rs`, each carrying the const its field loads from,
- the block name in `define_layout!`'s core list, in slot order.

A hand-written term says the same thing three times, and the three have to agree.
`KingSafetyTerm` clamps its attacker count in `apply`, again in `apply_input`, and again in `scatter`, all three off `ATTACKER.len() - 1` rather than the 5 it currently equals.
The cached copy carried that literal for a while and was correct the whole time, which is how it survived: a bound the engine owns, written out again where nothing checks it.

---

## Gradient cheat sheet

A term on bucket `B` with upstream `U`, where `bucket += param · feature_expr(p)`:

```
grads[p] += U · feature_expr(p)
```

|               Bucket              |            `Upstream`            |                                Scatter                                |
|-----------------------------------|----------------------------------|-----------------------------------------------------------------------|
| `bonus` (tapered bonus)           | `TaperPair { d_mg, d_eg }`       | `grads[mg] += d_mg · feature`, `grads[eg] += d_eg · feature`          |
| `mobility` (openness+phase blend) | `TaperPair`                      | same shape, times the openness fraction; see `MobilityTerm::scatter`  |
| `king_safety` (shelter + danger)  | `KingSafetyUpstream`             | shelter slots take `shelter · feature_diff`; each side's attacker weight takes its own `danger_us` / `danger_them` |
| `xray` (combiner tapers)          | `f64`                            | `grads[xray] += upstream · feature`                                   |

The danger halves arrive separately because the combiner curves each one before differencing them, so `∂block/∂danger` depends on which king it belongs to.

The combiner's `backward` already folded in the loss derivative, the STM sign, and the taper, so scatter is only `upstream · ∂bucket/∂param`.
A roster row generates all of it: the table matters when you hand-write a novel shape.
If your shape isn't here, derive `∂bucket/∂param` from your `apply` by hand; the oracle catches the slips.
