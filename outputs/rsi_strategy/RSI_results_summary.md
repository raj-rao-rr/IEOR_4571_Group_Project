# RSI strategy: measured results (auto-generated)

Data: LQD, adjusted close, 2002-12-31 to 2026-11-18; source: Yahoo Finance via yfinance (auto_adjust=True).
Test sample: 2020-01-02 to 2026-11-18 (1,730 trading sessions).

Baseline: Wilder RSI(14), entry below 30 / above 70, exit at 50, held position shifted one trading day.
Costs: illustrative 5 basis points per one-way 100% notional change (short borrowing and financing excluded).

| Out-of-sample metric | RSI net | Buy and hold |
|---|---:|---:|
| Mean daily return | -0.0034% | 0.0041% |
| Annualized arithmetic mean return | -0.85% | 1.02% |
| Annualized volatility | 7.40% | 10.06% |
| CAGR | -1.11% | 0.52% |
| Cumulative return | -7.39% | 3.61% |
| Maximum drawdown | -16.58% | -24.95% |
| Sharpe, rf=0 | -0.114 | 0.102 |

The baseline net RSI strategy **underperformed** buy-and-hold in total compounded returns over this test sample. This retrospective observation does not establish prospective predictive value.

Training-only sensitivity grid tested RSI windows 7, 14, 21 and symmetric entry thresholds 25/75, 30/70, 35/65. The best *training* Sharpe used RSI(21) and entry thresholds 30/70; its holdout performance is in `rsi_tuned_test_metrics.csv`. Do not select the preferred strategy by looking at test results.

Limitations: daily-close approximations, illustrative costs, no borrowing/short-locate constraints or cash interest, threshold overfitting risk, and potential nonstationarity in bond ETF behavior. Data include dividends through adjusted prices. Returns are historical and not guarantees.

Figures: `rsi_vs_buyhold_test.png`, `rsi_drawdown_test.png`, `rsi_indicator.png`.
