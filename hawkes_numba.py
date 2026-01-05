import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numba import jit
import time

@jit(nopython=True)
def hawkes_neg_log_likelihood_numba(params, t):
    mu = params[0]
    alpha = params[1]
    beta = params[2]
    
    if mu <= 1e-5 or alpha <= 1e-5 or beta <= 1e-5:
        return 1e10
        
    n = len(t)
    T_max = t[n-1]
    
    integral = mu * T_max + (alpha / beta) * np.sum(1 - np.exp(-beta * (T_max - t)))
    
    sum_log_lambda = 0.0
    R = 0.0
    
    sum_log_lambda += np.log(mu)
    
    for i in range(1, n):
        delta_t = t[i] - t[i-1]
        
        R = np.exp(-beta * delta_t) * (R + alpha)
        
        current_lambda = mu + R
        sum_log_lambda += np.log(current_lambda)
        
    return -(sum_log_lambda - integral)

filename = 'BTCUSDT-trades-2025-12.csv'  #replace with filename
column_names = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']

print("Loading data...")
df = pd.read_csv(filename, header=None, names=column_names, nrows=500000)

sell_df = df[df['is_buyer_maker'] == True]
t_raw = sell_df['time'].values

# Normalize time to seconds
t_history = (t_raw - t_raw[0]) / 1000.0

t_sample = t_history[:100000]
t_sample = t_sample.astype(np.float64)

print(f"Sample: {len(t_sample)} events")
print(f"Duration: {t_sample[-1]:.2f} seconds")

# --- 3. Run Optimization ---
print("\nRunning Calibration with Numba JIT...")
start_time = time.time()

# Initial guess
initial_params = np.array([0.1, 100.0, 200.0])

res = minimize(
    hawkes_neg_log_likelihood_numba, 
    initial_params, 
    args=(t_sample,),
    method='L-BFGS-B',
    bounds=[(1e-5, None), (1e-5, None), (1e-5, None)]
)

end_time = time.time()

print(f"Done in {end_time - start_time:.2f} seconds.")
print(f"Success: {res.success}")
print(f"Parameters (Mu, Alpha, Beta): {res.x}")

branching_ratio = res.x[1] / res.x[2]
print(f"Branching Ratio (n): {branching_ratio:.4f}")

if branching_ratio > 1.0:
    print(">> REGIME: UNSTABLE")
else:
    print(">> REGIME: STABLE")