use tuner::{evaltune::lion::Lion, searchtune::cmaes::CmaEs};

#[test]
fn test_lion_quadratic_convergence() {
    // Test Lion on simple quadratic: f(x) = sum(x_i²)
    let n = 10;
    let mut params = vec![10.0; n];
    let mut momentum = vec![0.0; n];
    let decay_mask = vec![1.0; n];
    let fixed_mask = vec![false; n];
    let beta2 = vec![0.99; n];

    let mut optimizer = Lion::new(0.9, 0.1, 0.0);

    for iter in 0..200 {
        // ∇f = 2x
        let grads: Vec<f64> = params.iter().map(|&p| 2.0 * p).collect();

        let decay = 1.0 - (iter as f64 / 200.0);
        optimizer.set_lr(0.1 * decay.max(0.05));

        let lr_mask = vec![1.0; params.len()];
        optimizer.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask);
    }

    let final_norm: f64 = params.iter().map(|x| x * x).sum::<f64>().sqrt();
    // Starting norm ≈ 31.6 (10 params at 10.0). After 200 iterations with
    // cosine-decayed LR from 0.1 to 0.005, Lion reliably reaches < 0.5.
    // The threshold is empirical — tighten if the optimizer is strengthened.
    assert!(final_norm < 0.5, "Lion failed to converge: final_norm={}", final_norm);
}

#[test]
fn test_lion_respects_fixed_mask() {
    let mut params = vec![1.0, 2.0, 3.0];
    let mut momentum = vec![0.0; 3];
    let grads = vec![1.0, 1.0, 1.0];
    let decay_mask = vec![1.0; 3];
    let fixed_mask = vec![false, true, false];
    let beta2 = vec![0.99; 3];

    let original_middle = params[1];

    let optimizer = Lion::new(0.9, 0.1, 0.0);
    let lr_mask = vec![1.0; params.len()];
    optimizer.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask);

    assert_eq!(params[1], original_middle, "Fixed parameter should not change");
    assert_ne!(params[0], 1.0, "Unfixed parameter should change");
    assert_ne!(params[2], 3.0, "Unfixed parameter should change");
}

#[test]
fn test_cmaes_sphere_function() {
    // Test CMA-ES on sphere function: f(x) = ||x||²
    let n = 5;
    let mut cmaes = CmaEs::new(n, 1.5);
    cmaes.set_mean(&vec![3.0; n]); // Start away from optimum
    let mut rng = fastrand::Rng::new();

    for _generation in 0..50 {
        let pop = cmaes.sample_population(&mut rng);

        // Evaluate: maximize negative ||x||²
        // CMA-ES is a maximizer, so we negate the sphere function to drive the search toward the origin.
        let fitness: Vec<f64> = pop
            .iter()
            .map(|x| {
                let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
                -sum_sq
            })
            .collect();

        cmaes.update(&pop, &fitness, &fitness, 0.01);
    }

    let final_norm: f64 = cmaes.mean().iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(final_norm < 1.0, "CMA-ES should converge close to origin, got {}", final_norm);
}

#[test]
fn test_pentanomial_symmetry() {
    use tuner::searchtune::pentanomial::Pentanomial;

    // WW-DD-LL should give 50% score
    let penta1 = Pentanomial { ww: 10, dd: 0, ll: 10, wd: 0, wl: 0, ld: 0 };
    assert!((penta1.score() - 0.5).abs() < 0.01, "Symmetric W/L should give 50%");

    // All WW should give 100%
    let penta2 = Pentanomial { ww: 20, dd: 0, ll: 0, wd: 0, wl: 0, ld: 0 };
    assert!((penta2.score() - 1.0).abs() < 0.01, "All wins should give 100%");

    // WD-LD should give 50%
    let penta3 = Pentanomial { ww: 0, dd: 0, ll: 0, wd: 10, wl: 0, ld: 10 };
    assert!((penta3.score() - 0.5).abs() < 0.01, "WD-LD should give 50%");

    // WL only: Win-Loss pairs score exactly 0.5 each, same as DD
    let penta_wl = Pentanomial { wl: 10, ww: 0, dd: 0, ll: 0, wd: 0, ld: 0 };
    assert!((penta_wl.score() - 0.5).abs() < 0.01, "WL pairs should give 50%");
}

#[test]
fn test_fisher_std_err() {
    use tuner::searchtune::pentanomial::Pentanomial;

    // High certainty (many games)
    let p_certain = Pentanomial { ww: 100, dd: 0, ll: 100, wd: 0, wl: 0, ld: 0 };
    let se_certain = p_certain.fisher_std_err();

    // Low certainty (few games)
    let p_uncertain = Pentanomial { ww: 1, dd: 0, ll: 1, wd: 0, wl: 0, ld: 0 };
    let se_uncertain = p_uncertain.fisher_std_err();

    assert!(se_certain < se_uncertain, "More games should imply lower SE");
    assert!(se_certain > 0.0, "SE should be positive");
}

#[test]
fn test_expected_improvement() {
    use tuner::searchtune::optimizer::expected_improvement;

    // Case 1: Mean > Best, low uncertainty -> High improvement
    let ei_high = expected_improvement(100.0, 1.0, 50.0);
    assert!(ei_high > 45.0, "EI should be close to mean-best for confident improvement");

    // Case 2: Mean < Best, low uncertainty -> ~0 improvement
    let ei_none = expected_improvement(0.0, 1.0, 50.0);
    assert!(ei_none < 0.001, "EI should be zero for confident failure");

    // Case 3: Mean < Best, HIGH uncertainty -> Positive improvement (Exploration)
    let ei_explore = expected_improvement(0.0, 100.0, 50.0);
    assert!(ei_explore > 0.1, "EI should value exploration of uncertain failures");
}
