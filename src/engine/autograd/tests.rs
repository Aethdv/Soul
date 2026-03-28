//! Unit tests for the forward-mode automatic differentiation logic.

#[test]
fn test_dual_basic_polynomial() {
    use super::dual::DualNode;

    // f(x, y) = 3x² + 2y at x=2, y=3
    // df/dx = 6x = 12
    // df/dy = 2
    let x = DualNode::seed(2.0, 0);
    let y = DualNode::seed(3.0, 1);
    let c3 = DualNode::constant(3.0);
    let c2 = DualNode::constant(2.0);

    let x_sq = x * x;
    let term1 = c3 * x_sq;
    let term2 = c2 * y;
    let f = term1 + term2;

    assert_eq!(f.val, 18.0);
    assert_eq!(f.grad[0], 12.0); // df/dx = 6 · 2.0
    assert_eq!(f.grad[1], 2.0); // df/dy = 2
}

#[test]
fn test_dual_division_and_negation() {
    use super::dual::DualNode;

    // f(x) = -x / 3 at x = 9
    // f = -3, df/dx = -1/3
    let x = DualNode::seed(9.0, 0);
    let c3 = DualNode::constant(3.0);

    let f = -x / c3;

    assert!((f.val - (-3.0)).abs() < 1e-10);
    assert!((f64::from(f.grad[0]) - (-1.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_dual_vs_finite_difference() {
    use super::dual::DualNode;

    // More complex: f(x, y) = (x · y + x) / (y - 1) at x=3, y=5
    // f = (15 + 3) / 4 = 4.5
    // df/dx = (y + 1) / (y - 1) = 6/4 = 1.5
    // df/dy = (x·(y-1) - (x·y+x)) / (y-1)² = (x·y - x - x·y - x) / (y-1)²
    //       = -2x / (y-1)² = -6/16 = -0.375
    let x = DualNode::seed(3.0, 0);
    let y = DualNode::seed(5.0, 1);
    let c1 = DualNode::constant(1.0);

    let numerator = x * y + x;
    let denominator = y - c1;
    let f = numerator / denominator;

    assert!((f.val - 4.5).abs() < 1e-10);
    assert!((f64::from(f.grad[0]) - 1.5).abs() < 1e-5);
    assert!((f64::from(f.grad[1]) - (-0.375)).abs() < 1e-5);

    // Verify against finite differences
    let eps = 1e-6;
    let fx = |xv: f64, yv: f64| -> f64 { (xv * yv + xv) / (yv - 1.0) };

    let fd_dx = (fx(3.0 + eps, 5.0) - fx(3.0 - eps, 5.0)) / (2.0 * eps);
    let fd_dy = (fx(3.0, 5.0 + eps) - fx(3.0, 5.0 - eps)) / (2.0 * eps);

    assert!((f64::from(f.grad[0]) - fd_dx).abs() < 1e-3);
    assert!((f64::from(f.grad[1]) - fd_dy).abs() < 1e-3);
}
