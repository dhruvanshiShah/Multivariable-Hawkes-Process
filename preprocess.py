import pandas as pd
import numpy as np

filename = 'BTCUSDT-trades-2025-12.csv' 

column_names = [
    'id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match'
]

print(f"Loading {filename}...")

df = pd.read_csv(
    filename, 
    header=None, 
    names=column_names, 
    usecols=['time', 'is_buyer_maker']
)

print("File loaded successfully.")

df = df.sort_values('time')

# Separate Buys and Sells
# is_buyer_maker = True  -> Taker is Seller (Sell Trade)
# is_buyer_maker = False -> Taker is Buyer (Buy Trade)
sell_df = df[df['is_buyer_maker'] == True]
buy_df = df[df['is_buyer_maker'] == False]

t_sell_raw = sell_df['time'].values
t_buy_raw = buy_df['time'].values

t_start = df['time'].iloc[0]

t_sell = (t_sell_raw - t_start) / 1000.0
t_buy = (t_buy_raw - t_start) / 1000.0

print(f"Total Trades: {len(df)}")
print(f"Buy Trades: {len(t_buy)}")
print(f"Sell Trades: {len(t_sell)}")
print(f"Duration: {t_sell[-1]:.2f} mseconds")

t_sell_sample = t_sell[:10000] 
print("Data ready. Sample size set to 10,000.")