#[test]
fn test_dual_basic_polynomial() {
    use super::dual::DualNode;

    // f(x, y) = 3x² + 2y  at  x = 2, y = 3
    // ∂f/∂x = 6x = 12
    // ∂f/∂y = 2
    let x = DualNode::seed(2.0, 0);
    let y = DualNode::seed(3.0, 1);
    let c3 = DualNode::constant(3.0);
    let c2 = DualNode::constant(2.0);

    let x_sq = x * x;
    let term1 = c3 * x_sq;
    let term2 = c2 * y;
    let f = term1 + term2;
    assert_eq!(f.val, 18.0);
    assert_eq!(f.grad[0], 12.0);
    assert_eq!(f.grad[1], 2.0);
}

#[test]
fn test_dual_division_and_negation() {
    use super::dual::DualNode;

    // f(x) = -x / 3  at  x = 9
    // f(9) = -3
    // f'(x) = -1/3 ≈ -0.33333334
    let x = DualNode::seed(9.0, 0);
    let c3 = DualNode::constant(3.0);
    let f = -x / c3;
    assert!((f.val - (-3.0)).abs() < 1e-6);
    assert!((f.grad[0] - (-1.0 / 3.0)).abs() < 1e-6);
}

#[test]
fn test_dual_vs_finite_difference() {
    use super::dual::DualNode;

    // f(x, y) = (x·y + x) / (y - 1)  at  x = 3, y = 5
    //
    // f(3, 5) = (15 + 3) / 4 = 4.5
    // ∂f/∂x   = (y + 1) / (y - 1) = 6/4 = 1.5
    // ∂f/∂y   = -2x / (y - 1)²    = -6/16 = -0.375
    let x = DualNode::seed(3.0, 0);
    let y = DualNode::seed(5.0, 1);
    let c1 = DualNode::constant(1.0);

    let numerator = x * y + x;
    let denominator = y - c1;
    let f = numerator / denominator;
    // Exact analytical checks
    assert!((f.val - 4.5).abs() < 1e-6);
    assert!((f.grad[0] - 1.5).abs() < 1e-6);
    assert!((f.grad[1] - (-0.375)).abs() < 1e-6);

    // Numerical finite difference check (central difference)
    let eps = 1e-4_f64;
    let fx = |xv: f64, yv: f64| -> f64 { (xv * yv + xv) / (yv - 1.0) };
    let fd_dx = (fx(3.0 + eps, 5.0) - fx(3.0 - eps, 5.0)) / (2.0 * eps);
    let fd_dy = (fx(3.0, 5.0 + eps) - fx(3.0, 5.0 - eps)) / (2.0 * eps);
    assert!((f64::from(f.grad[0]) - fd_dx).abs() < 1e-3);
    assert!((f64::from(f.grad[1]) - fd_dy).abs() < 1e-3);
}
