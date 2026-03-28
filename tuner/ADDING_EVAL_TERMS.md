# Adding New Eval Terms

*The step-by-step recipe for wiring new evaluation parameters.*

**→ See [EVAL_TUNER.md](EVAL_TUNER.md) for the architecture overview.**

---

## The Recipe

Say you're adding a **bishop pair bonus**: a weight that rewards having both bishops.

### 1. Add the parameter to `define_tunables!`

In `src/engine/eval_params.rs`:

```rust
#[macro_export]
macro_rules! define_tunables {
    ($macro:ident) => {
        $macro! {
            // ... existing fields ...
            (w_bishop_pair, Scalar, $crate::engine::eval_params::LAYOUT.bishop_pair_offset)
        }
    }
}
```

This macro automatically synchronizes the engine and autograd types.

Update `EvalParams::<i32>::from_const()` in `src/engine/eval.rs` to load from your new constant.

### 2. Add the constant to `eval_params.rs`

```rust
pub const BISHOP_PAIR_BONUS: i32 = 30; // initial guess, tuner will optimize
```

### 3. Use it in `evaluate_generic`

```rust
let bp_bonus = if board.has_bishop_pair(Color::White) && !board.has_bishop_pair(Color::Black) {
    params.w_bishop_pair
} else if board.has_bishop_pair(Color::Black) && !board.has_bishop_pair(Color::White) {
    -params.w_bishop_pair
} else {
    T::zero()
};

score += bp_bonus;
```

The feature here is `+1` if white has both bishops and black doesn't,\
`-1` if reversed,\
`0` if symmetric.\
It comes from the board, not from the weight value.

### 4. Register in the parameter layout

The tuner optimizes a flat `&[f64]` array. To map your parameter into this contiguous block, add its length and offset to `Layout` in `src/engine/eval_params.rs`:

```rust
pub struct Layout {
    // ... existing fields ...
    pub xray_offset:        usize,
    pub xray_len:           usize,
    pub bishop_pair_offset: usize,
    pub bishop_pair_len:    usize,
}

const fn calc_layout() -> Layout {
    // ...
    let bishop_pair_len = 1;

    // ...
    let xray_offset = king_safety_offset + safety_len;
    let bishop_pair_offset = xray_offset + xray_len;

    Layout {
        // ...
        bishop_pair_offset,
        bishop_pair_len,
    }
}
```

### 5. Add the gradient formula in `eval_linear_grad`

This is the production training path (`tuner/src/evaltune/tape.rs`). We bypass dual-number automatic differentiation here purely for speed. 

Because the evaluation is a strictly linear combination of its parameters ($score = \sum w_i \cdot x_i$), the partial derivative with respect to any weight ($\frac{\partial y}{\partial w_i}$) is simply its feature coefficient ($x_i$). Doing this explicitly in `f64` arithmetic avoids the massive overhead of allocating and tracking dual numbers for millions of positions.

In `eval_linear_grad`, compute the feature coefficient and multiply it by the upstream loss derivative `d` (which already wraps the sigmoid derivative and STM perspective flip):

```rust
// Bishop pair gradient
let bp_offset = psqt::LAYOUT.bishop_pair_offset;
let bp_feature = if board
    .pieces(PieceType::Bishop, Color::White)
    .more_than_one()
    && !board
        .pieces(PieceType::Bishop, Color::Black)
        .more_than_one()
{
    1.0
} else if board
    .pieces(PieceType::Bishop, Color::Black)
    .more_than_one()
    && !board
        .pieces(PieceType::Bishop, Color::White)
        .more_than_one()
{
    -1.0
} else {
    0.0
};

param_grads[bp_offset] += d * bp_feature;
```

### 6. The oracle is automatic

You don't need to write backward passes or manually seed dual numbers. Because you registered the parameter in the `define_tunables!` macro (Step 1), `eval_dual_fused` (the AD correctness oracle) automatically instantiates your parameter as a `DualNode` with its gradient tracking slot initialized.

The dual path is slow but mathematically perfect. The linear path (Step 5) is blazing fast but requires hand-written feature extraction.

### 7. Verify

Run the tuner. Before the first epoch, the training loop runs a Split-Brain Gradient Trap: it evaluates a batch of positions using both `eval_linear_grad` and `eval_dual_fused`. 

If your hand-derived feature coefficient in Step 5 deviates from the AD oracle by even `1e-4`, the tuner panics and identifies the exact parameter index. If it boots, your math is proven correct.

---

## The Gradient Cheat Sheet

For a term parameterized as `params.foo · feature`:

| Context | Gradient applied to `param_grads` |
|---------|----------------|
| **Direct addition** | `d · feature` |
| **Tapered** | `d · feature · t_mg` (MG) <br> `d · feature · t_eg` (EG) |
| **Mobility (Openness interpolated)** | `d · t_mg · feature · (openness / 1024.0)` (Open MG) <br> `d · t_eg · feature · (openness / 1024.0)` (Open EG) <br> `d · t_mg · feature · (closedness / 1024.0)` (Closed MG) <br> `d · t_eg · feature · (closedness / 1024.0)` (Closed EG) |
| **King Safety** | `d · t_mg · feature_diff` |

*(Where `d = outer_deriv · stm_sign`, precomputed at the top of `eval_linear_grad`)*
