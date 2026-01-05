import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

# IMPORT THE CYTHON MODULE
try:
    from hawkes import hawkes_neg_log_likelihood_cython
    print("✅ Cython module loaded successfully.")
except ImportError as e:
    print("❌ Cython module not found. Did you run 'python setup.py build_ext --inplace'?")
    exit()

# --- Load Data (Same as before) ---
filename = 'BTCUSDT-trades-2025-12.csv' 
column_names = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']
print("Loading data...")
# Read 200,000 rows this time!
df = pd.read_csv(filename, header=None, names=column_names, nrows=200000)

sell_df = df[df['is_buyer_maker'] == True]
t_raw = sell_df['time'].values
t_history = (t_raw - t_raw[0]) / 1000.0

# Sample size: 50,000 events (10x larger than before)
t_sample = t_history[:50000]
# Cython requires typed memory views, ensure it's float64 (double)
t_sample = t_sample.astype(np.float64)

print(f"Sample: {len(t_sample)} events")

# --- Run Optimization ---
print("\nRunning Calibration with Cython...")
start_time = time.time()

initial_params = [0.1, 100.0, 200.0] # Adjusted guess for high beta

res = minimize(
    hawkes_neg_log_likelihood_cython, # Using the Cython function
    initial_params, 
    args=(t_sample,),
    method='L-BFGS-B',
    bounds=[(1e-5, None), (1e-5, None), (1e-5, None)]
)

end_time = time.time()

print(f"Done in {end_time - start_time:.2f} seconds.")
print(f"Success: {res.success}")
print(f"Parameters: {res.x}")
print(f"Branching Ratio: {res.x[1]/res.x[2]:.4f}")