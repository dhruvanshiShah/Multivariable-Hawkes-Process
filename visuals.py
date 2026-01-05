import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

filename = 'BTCUSDT-trades-2025-12.csv' 
column_names = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']
df = pd.read_csv(filename, header=None, names=column_names, nrows=100000)

sell_df = df[df['is_buyer_maker'] == True]
t_raw = sell_df['time'].values
prices = sell_df['price'].values

t_history = (t_raw - t_raw[0]) / 1000.0
t_sample = t_history.astype(np.float64)

mu = 0.059
alpha = 261477.0
beta = 166407.0

@jit(nopython=True)
def compute_intensity(t, mu, alpha, beta):
    n = len(t)
    intensities = np.zeros(n)
    R = 0.0
    
    intensities[0] = mu
    
    for i in range(1, n):
        delta_t = t[i] - t[i-1]
        R = np.exp(-beta * delta_t) * (R + alpha)
        intensities[i] = mu + R
        
    return intensities

print("Computing Intensity Profile...")
lambda_values = compute_intensity(t_sample, mu, alpha, beta)

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Sell Intensity (Orders/sec)', color=color)
ax1.plot(t_sample, lambda_values, color=color, linewidth=0.5, alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('BTC Price (USDT)', color=color)
ax2.plot(t_sample, prices, color=color, linewidth=1.5)
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f'Market Microstructure: Intensity vs Price (Branching Ratio: {alpha/beta:.2f})')
fig.tight_layout()
plt.show()