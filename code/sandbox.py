import os
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import date


def resample_returns(
        price_series: pd.DataFrame,
        train_test_split: float,
        random_state: int | None = None,
    ) -> tuple:
    """
    Split a price series into sampled training and out-of-sample returns.
    ONLY use for strategies that don't require time-depedency. 

    Args:
        price_series: Historical prices, with one column per asset.
        train_test_split: Fraction of the returns to include in the training sample.
        random_state: Optional seed used for reproducible sampling.

    Returns:
        A tuple containing cumulative, normalized training returns and the
        remaining test returns.
    """
    returns = price_series.pct_change().dropna()

    n_sample = round(train_test_split * len(returns))
    train_ = returns.sample(
        n=n_sample,
        replace=False,
        random_state=random_state,
        ignore_index=False,
    )
    test_ = returns.loc[~np.isin(returns.index, train_.index)]

    zero_row = pd.DataFrame(0, index=[0], columns=train_.columns)
    train_ = pd.concat([zero_row, train_], ignore_index=True)
    train_ = (1+train_).cumprod() * 100
    
    train_.index = returns.index[:n_sample+1]
    test_.index = returns.index[n_sample:]

    return train_, test_

class PortfolioEngine:
    def __init__(
            self, 
            stock_data: pd.DataFrame, 
            signal_data: pd.DataFrame
        ):
        """
        Initialize the portfolio engine.

        Args:
            stock_data: DataFrame containing stock prices or another primary price series.
            signal_data: DataFrame containing the trading signal column to join by index.
        """
        self.stock_data = stock_data
        self.signal_data = signal_data

        self.df = self.stock_data.join(self.signal_data, how='left')
        self.strategy_pnl = None
        self.signal_lag = None

    def strategy_pnl_creation(self, signal_lag: int = 1) -> None:
        """
        Create the strategy PnL and store it on the engine.

        Args:
            signal_lag: Number of periods by which the trading signal is shifted
                to avoid look-ahead bias. Default = 1. 
        """
        df = self.df.copy()
        df['pct_chg'] = df[df.columns[0]].pct_change()

        # Cumulative PnL calculation
        pnl = 1 + df['pct_chg'] * df['signal'].shift(signal_lag).fillna(0) # Signal is shifted to avoid lookahead bias
        pnl.iloc[0] = 1
        df['pnl'] = pnl.cumprod() * 100
        
        self.strategy_pnl = df
        self.signal_lag = signal_lag

    def pnl_drawdown(self) -> float:
        """
        Calculate the portfolio PnL drawdown series.

        Returns:
            A Series whose values represent drawdown from the running peak.
        """
        peak = self.strategy_pnl['pnl'].cummax().clip(lower=1.0)   # starting capital counts as the first peak
        return (self.strategy_pnl['pnl'] / peak - 1)
    
    def annualized_return(self) -> float:
        """
        Calculate the annualized return over the strategy data period.

        Returns:
            The annualized return as a decimal.
        """
        total_return = self.strategy_pnl['pnl'].iloc[-1] / self.strategy_pnl['pnl'].iloc[0] - 1
        num_years = (self.strategy_pnl.index[-1] - self.strategy_pnl.index[0]).days / 365.25
        return (1 + total_return) ** (1 / num_years) - 1

    def annualized_volatility(self) -> float:
        """
        Calculate the annualized volatility of the strategy PnL.

        Returns:
            The annualized volatility based on 252 trading periods.
        """
        return self.strategy_pnl['pnl'].pct_change().std() * np.sqrt(252)

    def performance_statistics(self) -> dict:
        """
        Compute a summary of the strategy's performance.

        Returns:
            A dictionary containing cumulative return, annualized return,
            annualized volatility, Sharpe ratio, Calmar ratio, drawdown, and
            hit ratio statistics.

        Raises:
            ValueError: If strategy PnL has not been calculated yet.
        """
        if self.strategy_pnl is None:
            raise ValueError("PnL has not been calculated. Please run strategy_pnl_creation() first.")

        # Calculate performance statistics
        cum_total_return = self.strategy_pnl['pnl'].iloc[-1] - 100
        annualized_return = self.annualized_return()
        annualized_volatility = self.annualized_volatility()
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
        
        max_drawdown = self.pnl_drawdown().min()
        average_drawdown = self.pnl_drawdown().mean()

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

        live_position = self.strategy_pnl[abs(self.strategy_pnl['signal']) > 0] # Filter out signal trades
        direction = live_position[
            ((live_position['pct_chg'] > 0) & (live_position['signal'].shift(self.signal_lag) > 0)) | 
            ((live_position['pct_chg'] < 0) & (live_position['signal'].shift(self.signal_lag) < 0))
        ]
        hit_ratio = len(direction) / len(live_position) if len(live_position) > 0 else np.nan

        return {
            'Cumulative Total Return': cum_total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'Average Drawdown': average_drawdown,
            'Hit Ratio': hit_ratio
        }


class YahooFinanceData:
    def __init__(
            self, 
            ticker: list[str], 
            start_date: date, 
            end_date: date
        ):
        """
        Initialize Yahoo Finance download parameters.

        Args:
            ticker: Ticker symbols to download.
            start_date: Inclusive start date for the download.
            end_date: Exclusive end date for the download.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self) -> None:
        """
        Download historical market data and store it on the instance.
    
        Raises:
            ValueError: If Yahoo Finance returns no data for the requested
                tickers and date range.
        """
        self.data = yf.download(
            self.ticker, 
            start=self.start_date.strftime('%Y-%m-%d'), 
            end=self.end_date.strftime('%Y-%m-%d')
        )
        if self.data.empty:
            raise ValueError(f"No data fetched for {self.ticker} between {self.start_date} and {self.end_date}.")
