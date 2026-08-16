use tuner::evaltune::lion::Lion;

#[test]
fn test_lion_quadratic_convergence() {
    // Test Lion on simple quadratic: f(x) = sum(x_i²)
    let n = 10;
    let mut params = vec![10.0; n];
    let decay_mask = vec![1.0; n];
    let fixed_mask = vec![false; n];
    let beta2 = vec![0.99; n];
    let clip_mask = vec![(f64::NEG_INFINITY, f64::INFINITY); n];

    let mut optimizer = Lion::new(n, 0.9, 0.1, 0.0);

    for iter in 0..200 {
        // ∇f = 2x
        let grads: Vec<f64> = params.iter().map(|&p| 2.0 * p).collect();

        let decay = 1.0 - (iter as f64 / 200.0);
        optimizer.set_lr(0.1 * decay.max(0.05));

        let lr_mask = vec![1.0; params.len()];
        optimizer.update(&mut params, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &clip_mask);
    }

    let final_norm: f64 = params.iter().map(|x| x * x).sum::<f64>().sqrt();
    // Starting norm ≈ 31.6 (10 params at 10.0). After 200 iterations with
    // linear LR decay from 0.1 down to the 0.05 floor, Lion reliably reaches
    // < 0.5. The threshold is empirical: tighten if the optimizer is strengthened.
    assert!(final_norm < 0.5, "Lion failed to converge: final_norm={}", final_norm);
}

#[test]
fn test_lion_respects_fixed_mask() {
    let mut params = vec![1.0, 2.0, 3.0];
    let grads = vec![1.0, 1.0, 1.0];
    let decay_mask = vec![1.0; 3];
    let fixed_mask = vec![false, true, false];
    let beta2 = vec![0.99; 3];
    let clip_mask = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];

    let original_middle = params[1];

    let mut optimizer = Lion::new(3, 0.9, 0.1, 0.0);
    let lr_mask = vec![1.0; params.len()];
    optimizer.update(&mut params, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &clip_mask);

    assert_eq!(params[1], original_middle, "Fixed parameter should not change");
    assert_ne!(params[0], 1.0, "Unfixed parameter should change");
    assert_ne!(params[2], 3.0, "Unfixed parameter should change");
}
