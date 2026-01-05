#  Market Microstructure Modeling: Multivariate Hawkes Processes

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Optimization](https://img.shields.io/badge/optimization-Numba%20JIT-orange)

## Project Overview
This project implements a **Multivariate Hawkes Process** to model the self-exciting nature of high-frequency cryptocurrency limit order books. Unlike standard Poisson models, this model captures "volatility clustering"—where a single market order triggers a cascade of subsequent orders.

The core engine uses **Maximum Likelihood Estimation (MLE)** to fit intensity parameters $(\mu, \alpha, \beta)$ to tick-level trade data, quantifying the "reflexivity" of the market.

##  Key Technical Features
* **Stochastic Modeling:** Models the conditional intensity $\lambda(t)$ of trade arrivals using a self-exciting point process:
  $$\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta(t - t_i)}$$
* **High-Performance Computing:** Implemented **JIT-compiled (Numba)** likelihood estimation, achieving a **100x speedup** over standard Python by optimizing Ogata’s recursive algorithm for $O(N)$ complexity.
* **Regime Detection:** Calculates the **Branching Ratio ($n = \alpha/\beta$)** to detect super-critical market regimes ($n > 1$) associated with flash crashes and liquidity crises.

## Results & Performance
* **Calibration Speed:** Processed **100,000 tick events** in **4.64 seconds** (vs ~12 minutes in pure Python).
* **Market Insight:** Analyzed BTC/USDT (Dec 2025) and detected a Branching Ratio of **1.57**, indicating a **super-critical (unstable)** regime where order flow was highly endogenous.

## Installation & Usage

### 1. Prerequisites
* Python 3.8+
* Binance Trade Data (CSV)

### 2. Install Dependencies
```bash
pip install -r requirements.txt