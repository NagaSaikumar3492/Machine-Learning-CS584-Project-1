import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.lasso_homotopy import LassoHomotopy

# Test 1: Collinear Data - one feature should be suppressed

def test_lasso_collinear_data():
    X = np.random.randn(20, 2)
    X[:, 1] = X[:, 0]  # Perfect collinearity
    y = np.random.randn(20)
    model = LassoHomotopy(lambda_val=0.1)
    model.fit(X, y)

    assert np.isclose(model.coefficients[0], 0, atol=1e-2) or np.isclose(model.coefficients[1], 0, atol=1e-2), \
        "At least one of the collinear coefficients should be near zero."

# Test 2: Single Feature - model should still work

def test_lasso_single_feature():
    X = np.random.randn(20, 1)
    y = 3 * X[:, 0] + np.random.randn(20) * 0.1
    model = LassoHomotopy(lambda_val=0.1)
    model.fit(X, y)

    assert model.coefficients.shape == (1,), "Should have exactly one coefficient"
    assert np.abs(model.coefficients[0]) > 0, "Coefficient should not be zero"

# Test 3: Noisy Data - model shouldn't crash and should return coefficients

def test_lasso_noisy_data():
    X = np.random.randn(50, 5)
    true_coefficients = np.array([1, -2, 3, 0, 0])
    y = X @ true_coefficients + np.random.randn(50) * 10
    model = LassoHomotopy(lambda_val=0.5)
    model.fit(X, y)

    assert model.coefficients is not None, "Coefficients should not be None"

# Test 4: High Regularization - should zero out all coefficients

def test_lasso_high_regularization():
    X = np.random.randn(30, 5)
    y = np.random.randn(30)
    model = LassoHomotopy(lambda_val=1000)
    model.fit(X, y)

    assert np.allclose(model.coefficients, 0), "All coefficients should be zero under high regularization."

if __name__ == "__main__":
    pytest.main()

