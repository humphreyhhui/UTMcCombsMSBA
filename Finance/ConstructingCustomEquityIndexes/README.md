- Built three custom stock indexes (equal-weighted, value-weighted, price-weighted) from top 100 stocks by market cap using CRSP data (2015-2024), incorporating delisting returns to avoid survivorship bias and implementing monthly reconstitution with strict t-1 information to prevent look-ahead bias.

- Compared custom indexes against SPY, IWM, and QQQ using correlation analysis and log returns, computed HHI scores for sector diversification analysis, and tested robustness across different market conditions by resetting indexes during 2020, 2022, and 2023 periods.

- Tech Stack: pandas, numpy, matplotlib, wrds, scipy, seaborn
