"""Застосування математичного аналізу в аналізі ринку акцій"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def download_data(ticker: str, period: str = "1y", interval: str = "1d") -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Завантажує історичні дані по тикеру з Yahoo Finance."""
    data = yf.download(ticker, period=period, interval=interval)
    prices = data['Close'].values.flatten()
    dates = data.index
    return prices, dates


def compute_smooth_and_bands(
    prices: np.ndarray,
    window: int = 20,
    sensitivity: float = 1.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Розраховує EMA, смуги відхилення та кінематичні показники.

    Повертає:
        smooth_prices — EMA-згладжені ціни
        dynamic_std — динамічне стандартне відхилення (rolling)
        velocity — перша похідна (швидкість зміни)
        acceleration — друга похідна (прискорення)
    """
    prices_series = pd.Series(prices)
    smooth_prices = prices_series.ewm(span=window).mean().values
    dynamic_std = prices_series.rolling(window=window, min_periods=1).std().values
    velocity = np.gradient(smooth_prices)
    acceleration = np.gradient(velocity)
    return smooth_prices, dynamic_std, velocity, acceleration


def generate_signals(
    prices: np.ndarray,
    smooth_prices: np.ndarray,
    dynamic_std: np.ndarray,
    acceleration: np.ndarray,
    sensitivity: float = 1.2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Генерує сигнали BUY/SELL на основі відхилення від EMA та прискорення.

    BUY:  ціна значно нижче EMA + прискорення вгору
    SELL: ціна значно вище EMA + прискорення вниз
    """
    deviation = prices - smooth_prices
    buy_signals = (deviation < -sensitivity * dynamic_std) & (acceleration > 0)
    sell_signals = (deviation > sensitivity * dynamic_std) & (acceleration < 0)
    return buy_signals, sell_signals


def plot_analysis(
    ticker: str,
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    smooth_prices: np.ndarray,
    dynamic_std: np.ndarray,
    buy_signals: np.ndarray,
    sell_signals: np.ndarray,
    sensitivity: float = 1.2,
    window: int = 20
) -> None:
    """Будує графік з ціною, EMA-смугами та сигналами входу/виходу."""
    plt.figure(figsize=(15, 8))

    plt.plot(dates, prices, label='Ціна close', color='black', alpha=0.3, linewidth=1)
    plt.plot(dates, smooth_prices, label=f'EMA-{window} (базова функція)', color='blue', linewidth=2)

    plt.fill_between(
        dates,
        smooth_prices - sensitivity * dynamic_std,
        smooth_prices + sensitivity * dynamic_std,
        color='blue', alpha=0.1, label='Зона математичної норми'
    )

    plt.scatter(dates[buy_signals], prices[buy_signals],
                color='green', marker='^', s=120, label='BUY (точка входу)')
    plt.scatter(dates[sell_signals], prices[sell_signals],
                color='red', marker='v', s=120, label='SELL (точка виходу)')

    plt.title(f"Комплексний аналіз {ticker}: Диференціальні сигнали та статистичні межі")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def run_analysis(
    ticker: str = "TSLA",
    period: str = "1y",
    interval: str = "1d",
    window: int = 20,
    sensitivity: float = 1.2
) -> None:
    """Головна функція: запускає повний аналіз для заданого тикера."""
    prices, dates = download_data(ticker, period, interval)
    smooth_prices, dynamic_std, velocity, acceleration = compute_smooth_and_bands(prices, window, sensitivity)
    buy_signals, sell_signals = generate_signals(prices, smooth_prices, dynamic_std, acceleration, sensitivity)
    plot_analysis(ticker, dates, prices, smooth_prices, dynamic_std, buy_signals, sell_signals, sensitivity, window)


if __name__ == "__main__":
    run_analysis(ticker="TSLA", window=20, sensitivity=1.2)
