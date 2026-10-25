use super::dual::DualNode;

#[test]
fn polynomial_gradient() {
    let x = DualNode::seed(2.0, 0);
    let y = DualNode::seed(3.0, 1);
    let c3 = DualNode::constant(3.0);
    let c2 = DualNode::constant(2.0);

    let x_sq = x * x;
    let term1 = c3 * x_sq;
    let term2 = c2 * y;
    let f = term1 + term2;

    assert_eq!(f.val, 18.0, "3x² + 2y at (2, 3)");
    assert_eq!(f.grad[0], 12.0, "∂f/∂x of 3x² + 2y at (2, 3)");
    assert_eq!(f.grad[1], 2.0, "∂f/∂y of 3x² + 2y at (2, 3)");
    assert!(f.grad[2..].iter().all(|&g| g == 0.0), "slots other than x and y stay zero");
}

#[test]
fn division_and_negation_gradient() {
    let x = DualNode::seed(9.0, 0);
    let c3 = DualNode::constant(3.0);
    let f = -x / c3;

    assert!((f.val + 3.0).abs() < 1e-6, "value of -x/3 at x = 9");
    assert!((f.grad[0] + 1.0 / 3.0).abs() < 1e-6, "∂f/∂x of -x/3 at x = 9");
}

#[test]
fn rational_function_gradient() {
    let x = DualNode::seed(3.0, 0);
    let y = DualNode::seed(5.0, 1);
    let c1 = DualNode::constant(1.0);

    let numerator = x * y + x;
    let denominator = y - c1;
    let f = numerator / denominator;

    assert!((f.val - 4.5).abs() < 1e-6, "value of (xy + x)/(y - 1) at (3, 5)");
    assert!((f.grad[0] - 1.5).abs() < 1e-6, "∂f/∂x of (xy + x)/(y - 1) at (3, 5)");
    assert!((f.grad[1] + 0.375).abs() < 1e-6, "∂f/∂y of (xy + x)/(y - 1) at (3, 5)");
}
