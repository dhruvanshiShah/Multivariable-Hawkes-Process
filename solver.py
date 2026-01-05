import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

filename = 'BTCUSDT-trades-2025-12.csv' #replace with file name
column_names = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']

print("Loading data...")
df = pd.read_csv(filename, header=None, names=column_names, nrows=20000)

sell_df = df[df['is_buyer_maker'] == True]
t_raw = sell_df['time'].values

t_history = (t_raw - t_raw[0]) / 1000.0

t_sample = t_history[:5000]
print(f"Solver Sample: {len(t_sample)} events")
print(f"Sample Duration: {t_sample[-1]:.2f} seconds")

def hawkes_neg_log_likelihood(params, t):
    mu, alpha, beta = params
    
    if mu <= 1e-5 or alpha <= 1e-5 or beta <= 1e-5:
        return 1e10 
        
    T_max = t[-1] 
    n = len(t)
    
    integral = mu * T_max + (alpha / beta) * np.sum(1 - np.exp(-beta * (T_max - t)))
    
    sum_log_lambda = 0.0
    R = 0.0 # recursive memory term
    
    sum_log_lambda += np.log(mu)
    
    for i in range(1, n):
        delta_t = t[i] - t[i-1]
        
        R = np.exp(-beta * delta_t) * (R + alpha)
        
        current_lambda = mu + R
        sum_log_lambda += np.log(current_lambda)
        
    return -(sum_log_lambda - integral)

print("\nRunning Calibration (this may take 15-30 seconds)...")
start_time = time.time()

initial_params = [0.1, 0.5, 1.0]

res = minimize(
    hawkes_neg_log_likelihood, 
    initial_params, 
    args=(t_sample,),
    method='L-BFGS-B',
    bounds=[(1e-5, None), (1e-5, None), (1e-5, None)]
)

end_time = time.time()

print(f"Done in {end_time - start_time:.2f} seconds.")
print(f"Success: {res.success}")
mu_est, alpha_est, beta_est = res.x

print(f"\n--- RESULTS ---")
print(f"Mu (Background Rate): {mu_est:.4f} Hz")
print(f"Alpha (Excitation):   {alpha_est:.4f}")
print(f"Beta (Decay Speed):   {beta_est:.4f}")

branching_ratio = alpha_est / beta_est
print(f"Branching Ratio (n):  {branching_ratio:.4f}")

if branching_ratio > 1.0:
    print(">> REGIME: UNSTABLE (Panic/Viral Behavior detected)")
else:
    print(">> REGIME: STABLE (Mean-reverting)")