**Implementation of Lasso Regression using Homotopy method**

**Team Members:-**
1. Nikita Sharma ​    –  A20588827
2. Naga Sai Kumar Potti –   A20539661
3. Nikita Rai ​​    –   A20592025
4. Stephen Amankwa​    –   A20529876

 =====================================================================================================
 
**LASSO Homotopy Regression**
 
**1. Setup Instructions to create virtual environment :**
 
Step 1: **python3 -m venv venv**
 
Step 2: **source venv/bin/activate**  
 
**2. Installation Steps:**
 
Step 1: **pip install -r requirements.txt**
 
Step 2: Run the main model script : **python3 LassoHomotopy.py**
 
Step 3: Run the tests: **pytest test_LassoHomotopy.py**
 
**3. Usage Example:**
 
To test with the below example please go to file LassoHomotopy.py
and update the below function with your desired values :-
 
if __name__ == "__main__":

   print("Machine Learning Assignment...")
   
   np.random.seed(42)
   
   X = np.random.randn(50, 10)
   
   true_coefficients = np.array([1.5, -2.0, 0, 0, 3.0, 0, 0, -1.2, 0, 2.5])
   
   y = X @ true_coefficients + np.random.randn(50) * 0.1  # Add small Gaussian noise
   
   model = LassoHomotopy(lambda_val=0.1)
   
   model.fit(X, y)
   
   model.evaluate(X, y)
  
**4. Outputs :**
 
Learned coefficients, True coefficients, Sample predictions, MSE and R² score
 
**5. Plots:**
 
Coefficients bar plot, Actual vs predicted scatter, Residual plot, Loss convergence, LASSO predictions vs actual, True coefficient vs learned coefficient.

 =====================================================================================================

***QUESTIONS***
 
**1. What does the model you have implemented do and when should it be used?**
 
We have created a model which implements LASSO regression by using Homotopy method. It estimates linear relationships between input features and a target variable. It also promotes sparsity by shrinking less important coefficients to zero.
 
We can use this model when there are multiple features but only some features are important to us. It can also be used when we want automatic feature selection to be present in the learning process. Additionally, when there’s a need of an interpretable, robust model which avoids overfitting. It is also efficient when we need to trace the solution path as the value of λ varies. Also, when we are working with high-dimensional but sparse data.
 
 
**2. How did you test your model to determine if it is working reasonably correctly?**
 
We have implemented four specific unit test cases by using pytest to verify that the LASSO Homotopy model works correctly. These test cases validate our model's ability to enforce sparsity, handle the edge cases, and acknowledge different types of data. In the collinear data test we have created two perfectly correlated features and confirmed the same using np.isclose() function that if the model suppresses at least one of the coefficient to ensure that our LASSO model is behaving as expected. The single feature test verifies that the model can handle one dimensional input without any issue by checking if the resulting coefficient was non-zero and is shaped correctly. This guarantees our model’s functionality in both redundant and minimal input scenarios.
 
We have also tested robustness to noisy data by injecting strong gaussian noise into the target variable and verifying that the model is still returning the valid coefficients. Also, we tested extreme regularization by setting a very high lambda value and used np.allclose() function to verify that all coefficients were driven to zero. These behaviors were verified via model.fit(X, y) and validated through model.coefficients. All the tests that we have implemented assures that our model is successfully implementing sparsity, regularization strength, and numerical stability and thereby functioning properly as expected with LASSO implementation.
 
**3. What parameters have you exposed to users of your implementation in order to tune performance?**
 
We have exposed the below parameters which can be used by users to tune performance:-
 
a. lambda_val: The regularization strength (higher value symbolize more sparsity).
b. max_iter: Max number of iterations in the fitting loop (default value is 1000).
c. tol: Tolerance for convergence (default value is 1e-4).
d. non-zero coefficient
 
These parameters allow the users to tune for speed, accuracy, and regularization strength. This gives the users full control over how the model behaves.
 
**4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?**
 
We have applied LASSO Homotopy model which runs into challenges with specific input types. When the data is extremely noisy, and the ratio of signal-to-noise is low it gets difficult to distinguish related features. In the case of optimization, the datasets with highly correlated or high dimensional feature sets might result in slower convergence or instability. Also, for very small values of the regularization parameter the λ can also cause our model to overfit or produce unstable outputs. These challenges cannot be considered as flaws as the Homotopy method itself reflect the areas for practical improvement in our current implementation.
 
It also does not support Elastic Net-style L1+L2 regularization, which could be beneficial in handling highly correlated features more gracefully. 
 
If we are provided with more time, we can address the above issues by integrating adaptive λ strategies, path-wise LARS style improvements which can help include more stable features, and more numerically robust linear algebra techniques which can enhance stability and scalability. Additionally, the current Homotopy implementation assumes clean, non-pathological input and is implemented specifically for L1 regularization.
