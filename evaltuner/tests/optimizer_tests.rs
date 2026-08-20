use evaltuner::lion::{Lion, Masks};

#[test]
fn lion_converges_on_a_quadratic() {
    let n = 10;
    let mut params = vec![10.0; n];
    let masks = Masks::uniform(n, 0.99);

    let mut optimizer = Lion::new(n, 0.9, 0.1, 0.0);

    for iter in 0..200 {
        // f(x) = Σ x_i², so ∇f = 2x.
        let grads: Vec<f64> = params.iter().map(|&p| 2.0 * p).collect();

        let decay = 1.0 - (iter as f64 / 200.0);
        optimizer.set_lr(0.1 * decay.max(0.05));
        optimizer.update(&mut params, &grads, &masks);
    }

    let final_norm: f64 = params.iter().map(|x| x * x).sum::<f64>().sqrt();
    // Starting norm is 31.6, ten parameters at 10.0. Over 200 iterations with the rate decaying
    // linearly from 0.1 to a floor of 0.005, Lion reaches under 0.5. The threshold is empirical:
    // tighten it if the optimizer gets stronger.
    assert!(final_norm < 0.5, "did not converge: final norm {final_norm}");
}

#[test]
fn a_fixed_parameter_never_moves() {
    let mut params = vec![1.0, 2.0, 3.0];
    let grads = vec![1.0, 1.0, 1.0];
    let masks = Masks { fixed: vec![false, true, false], ..Masks::uniform(3, 0.99) };
    let original_middle = params[1];

    let mut optimizer = Lion::new(3, 0.9, 0.1, 0.0);
    optimizer.update(&mut params, &grads, &masks);
    assert_eq!(params[1], original_middle, "the fixed parameter moved");
    assert_ne!(params[0], 1.0, "the free parameter before it did not");
    assert_ne!(params[2], 3.0, "the free parameter after it did not");
}
