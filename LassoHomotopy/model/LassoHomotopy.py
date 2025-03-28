import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class LassoHomotopy:
    def __init__(self, lambda_val=0.1):
        self.lambda_val = lambda_val
        self.coefficients = None
        self.loss_values = []

    def fit(self, X, y, max_iter=1000, tol=1e-4):
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        active_set = set()
        residual = y - X @ self.coefficients
        prev_loss = np.inf

        for _ in range(max_iter):
            correlations = X.T @ residual
            max_corr_idx = np.argmax(np.abs(correlations))

            if np.abs(correlations[max_corr_idx]) < self.lambda_val:
                print("Converged based on the lambda threshold value.")
                break

            active_set.add(max_corr_idx)
            X_active = X[:, list(active_set)]
            theta_active = np.linalg.pinv(X_active) @ y
            self.coefficients[list(active_set)] = theta_active.flatten()
            residual = y - X @ self.coefficients
            loss = np.mean(residual ** 2)
            self.loss_values.append(loss)

            if abs(prev_loss - loss) < tol:
                print("Converged based on the tolerance value.")
                break
            prev_loss = loss
        else:
            print("Reached the max no of iterations.")

    def predict(self, X):
        return X @ self.coefficients

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"The Mean Squared Error value will be: {mse:.4f}")
        print(f"The R² Score will be: {r2:.4f}")
        return mse, r2

    def plot_coefficients(self):
        plt.figure(figsize=(8, 5))
        sns.barplot(x=np.arange(len(self.coefficients)), y=self.coefficients, hue=np.arange(len(self.coefficients)), palette="viridis", legend=False)
        plt.xlabel("Feature Index")
        plt.ylabel("Coefficient Value")
        plt.title(f"LASSO Coefficients (λ={self.lambda_val})")
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.show()

    def plot_predictions(self, X, y):
        y_pred = self.predict(X)
        plt.figure(figsize=(6, 6))
        plt.scatter(y, y_pred, alpha=0.7)
        plt.plot(y, y, color="red", linestyle="--")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("LASSO Predictions vs. Actual")
        plt.show()

    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals, color="blue")
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()

    
    def plot_true_vs_learned(self, true_coefficients):
        """Ploting the graph btw true coefficients vs. learned coefficients."""
        plt.figure(figsize=(8, 5))
        plt.scatter(true_coefficients, self.coefficients, alpha=0.7, color='blue')
        plt.plot(true_coefficients, true_coefficients, linestyle="--", color="red", label="Ideal Fit")
        plt.xlabel("True Coefficients")
        plt.ylabel("Learned Coefficients")
        plt.title("True vs. Learned Coefficients")
        plt.legend()
        plt.show()

    def plot_coefficient_comparison(self, true_coefficients):
        """Ploting the graph btw coefficient comparison (true vs learned)."""
        plt.figure(figsize=(8, 5))
        plt.plot(true_coefficients, label="True Coefficients", marker='o')
        plt.plot(self.coefficients, label="Learned Coefficients", marker='x')
        plt.xlabel("Feature Index")
        plt.ylabel("Coefficient Value")
        plt.title("Coefficient Comparison")
        plt.legend()
        plt.show()

    def plot_convergence(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.loss_values)), self.loss_values, marker='o', linestyle='-', color="green")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Convergence of LASSO Homotopy Algorithm")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    print("Machine Learning Assignment...")

    # Generate synthetic dataset
    np.random.seed(42)
    X = np.random.randn(50, 10)
    true_coefficients = np.array([1.5, -2.0, 0, 0, 3.0, 0, 0, -1.2, 0, 2.5])
    y = X @ true_coefficients + np.random.randn(50) * 0.1  # Add small Gaussian noise

    # Initialize and train the model
    model = LassoHomotopy(lambda_val=0.1)
    model.fit(X, y)

    # Evaluate performance
    model.evaluate(X, y)

    # Print learned coefficients
    print("true coefficients:", true_coefficients)
    print("Learned Coefficients:", model.coefficients)
    print(f"Number of Non-Zero Coefficients: {np.sum(model.coefficients != 0)}")

    # Plotting results
    model.plot_coefficients()
    model.plot_predictions(X, y)
    model.plot_residuals(y, model.predict(X))
    model.plot_convergence()
    
    model.plot_true_vs_learned(true_coefficients)
    model.plot_coefficient_comparison(true_coefficients)

