import pandas as pd

def generate_signals(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    
    """Returns a binary Series of positions (1.0 = Long, 0.0 = Cash)."""

    # Compute rolling moving averages
    fast_ma = prices.rolling(window=fast).mean()
    slow_ma = prices.rolling(window=slow).mean()

    # Long when the fast moving average is above the slow moving average
    signal = (fast_ma > slow_ma).astype(float)

    # Shift by 1 to avoid look-ahead bias
    positions = signal.shift(1)

    # No position before enough information is available
    positions = positions.fillna(0.0)

    return positions