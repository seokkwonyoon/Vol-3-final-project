I am trying to work in dynamic_ic/ to execute this plan. I want the structure to be slightly similar to momentum/. Can you help me to execute this plan?

we should pre-compute our z-scores for all of our many signals, then do a walkforward analysis where we train our model while simultaneously predicting our IC. our models should have a modular option to plug in each IC function predicting method in models/


Plan:
Phase 1: Data Collection and Standardization Gather common factor signals such as momentum (past returns) and value (price-to-book ratios). For every daily timestep, calculate the cross-sectional z-score for each stock. This is done by subtracting the mean signal value of all stocks and dividing by the standard deviation for that specific day. This ensures all signals are on the same scale regardless of market volatility. Pre-compute z-scores for all signals and 


Phase 2: Static Baseline Backtest Establish a benchmark using a fixed Information Coefficient (IC) of 0.05. Calculate the expected return for each stock as: Expected Return = Volatility * 0.05 * Z-score. Input these expected returns and a historical covariance matrix into a Mean-Variance Optimizer to determine portfolio weights. Run this backtest over your entire dataset to generate a baseline equity curve and Sharpe Ratio.


Phase 3: Rolling Window Configuration Define a rolling lookback window, for example, the most recent 60 or 120 trading days. At each rebalancing period (daily or weekly), gather all the (z-score, realized return) pairs from all stocks within that window. This creates a pooled dataset of recent market behavior.


Phase 4: Gaussian Process Training Train a Gaussian Process (GP) regression on the collected window of data. The input is the z-score and the target is the realized return. The GP will learn the specific shape of the relationship between signal strength and actual performance. By using a non-linear kernel like RBF, the model can discover if the signal is more predictive at extreme z-scores (like z > 2) compared to the average.


Phase 5: Dynamic Signal Prediction For the current timestep, take the latest stock z-scores and pass them through the trained GP. The model will provide two outputs for each stock: a predicted mean return and a predictive variance (uncertainty). This variance represents how "sure" the model is about the prediction based on recent data density. 


Phase 6: Portfolio Optimization with Uncertainty Run the Mean-Variance Optimizer using the dynamic predictions. Specifically, incorporate the GP's uncertainty into the risk term. A common method is to add the predictive variance to the diagonal of the covariance matrix. This naturally forces the optimizer to reduce the weights of stocks where the signal is currently unreliable or "noisy." 


Phase 7: Performance Attribution and Comparison Compare the results of the Dynamic GP strategy against the Static 0.05 IC strategy. Analyze whether the dynamic approach improves the Sharpe Ratio or reduces drawdowns. Visualize the "IC Function" by plotting the GP's predicted curve at different points in time to demonstrate how the model adapted to different market regimes.


Models we want to test:
1. State-Space Polynomial Kalman Filter (The "Gold Standard")
   Instead of tracking a single IC value, your Kalman state is a vector of coefficients (e.g., β0, β1, β2) for a polynomial like IC(z) = β0 + β1*z + β2*z^2.
   Why it’s best: It treats the "true" shape of the IC as a hidden state that evolves over time. At each step t, the filter uses the returns of all 3,000 assets as a high-dimensional observation to "tilt" or "bend" the curve.
   Benefit: It provides a mathematically rigorous way to handle noise (σ) and gives you an uncertainty estimate for the IC function itself.

2. Recursive Basis Function Expansion (RBF-RLS)
   You map your z-scores into a small set of "features" using fixed basis functions (like 5-7 Gaussian kernels or Splines spread across the z range). You then use Recursive Least Squares (RLS) to solve for the weights of these bases.
   Why it’s best: Unlike a polynomial which can explode at the edges, basis functions are "local." If your signal only works for extremely high z-scores, the RLS will only update the weights for the basis functions in that region.
   Benefit: It can approximate almost any function shape (highly non-linear) while remaining a simple linear update at each timestep.

3. Dynamic Binned Kalman Filters (The "Robust" Choice)
   Partition your z-scores into 10–20 equal-width or equal-count bins. Assign an independent, 1D Kalman Filter to track the IC of each bin separately.
   Why it’s best: It makes zero assumptions about the shape of the function (it’s "non-parametric"). If the 10th decile is mean-reverting but the 5th decile is trending, the bins will capture that divergence naturally.
   Benefit: Extremely easy to debug. You can literally plot the 20 filter states to "see" the IC function as a bar chart that changes every day.

4. Online Nadaraya-Watson (Kernel Smoothing)
   This is a "memory-based" approach. You maintain a rolling buffer of the last N days of data. For a new z, your predicted IC(z) is a weighted average of past returns, where weights are determined by how close the historical z was to your current z (spatial) and how recent the data is (temporal).
   Why it’s best: It is the purest way to "let the data speak." It doesn't force a functional form (like a parabola) on the IC.
   Benefit: Very responsive to "local" anomalies in the z-distribution that a global polynomial might smooth over.