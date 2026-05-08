# monthly_momentum_rebalance.py

import os
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from datetime import datetime

TOP_N = 10
MAX_SECTOR_WEIGHT = 0.50

LOOKBACK_12M_DAYS = 252
LOOKBACK_6M_DAYS = 126
SKIP_DAYS = 21

WEIGHT_6M = 0.50
WEIGHT_12M = 0.50

MIN_PRICE = 10
MIN_DOLLAR_VOLUME = 50_000_000
MIN_VALID_DAYS = 252

MAX_ALLOWED_MOMENTUM = 2.5
MAX_SINGLE_DAY_MOVE = 0.50
MAX_ANNUALIZED_VOL = 1.50

PORTFOLIO_FILE = "portfolio.csv"
HTML_FILE = "monthly_email.html"


def get_sp500_table():
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(wiki_url, headers=headers, timeout=20)
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
    except Exception as e:
        print(f"Wikipedia failed: {e}")
        print("Using fallback S&P 500 source...")

        fallback_url = (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/master/data/constituents.csv"
        )
        table = pd.read_csv(fallback_url)

    table["Ticker"] = table["Symbol"].str.replace(".", "-", regex=False)
    table["Name"] = table["Security"]
    table["Sector"] = table["GICS Sector"]

    return table[["Ticker", "Name", "Sector"]]


def download_data(tickers):
    print(f"Downloading {len(tickers)} tickers...")

    return yf.download(
        tickers=tickers,
        period="2y",
        auto_adjust=True,
        group_by="ticker",
        progress=True,
        threads=True,
    )


def get_ticker_data(data, ticker):
    try:
        df = data[ticker].copy()
        df = df.dropna()
        return df
    except Exception:
        return None


def calculate_momentum_components(close):
    close = close.dropna()

    if len(close) < MIN_VALID_DAYS:
        return None

    try:
        price_now = close.iloc[-1]
        price_6m = close.iloc[-LOOKBACK_6M_DAYS]
        price_12m = close.iloc[-LOOKBACK_12M_DAYS]
        price_skip = close.iloc[-SKIP_DAYS]

        if min(price_now, price_6m, price_12m, price_skip) <= 0:
            return None

        momentum_6m = price_now / price_6m - 1
        momentum_12m_ex_1m = price_skip / price_12m - 1

        blended_momentum = (
            WEIGHT_6M * momentum_6m
            + WEIGHT_12M * momentum_12m_ex_1m
        )

        return {
            "Momentum6M": momentum_6m,
            "Momentum12MEx1M": momentum_12m_ex_1m,
            "BlendedMomentum": blended_momentum,
        }

    except Exception:
        return None


def passes_data_quality_filters(close):
    close = close.dropna()

    if len(close) < MIN_VALID_DAYS:
        return False

    daily_returns = close.pct_change().dropna()

    if daily_returns.empty:
        return False

    if daily_returns.abs().max() > MAX_SINGLE_DAY_MOVE:
        return False

    annualized_vol = daily_returns.std() * (252 ** 0.5)

    if annualized_vol > MAX_ANNUALIZED_VOL:
        return False

    components = calculate_momentum_components(close)

    if components is None:
        return False

    blended_momentum = components["BlendedMomentum"]

    if blended_momentum <= 0:
        return False

    if blended_momentum > MAX_ALLOWED_MOMENTUM:
        return False

    return True


def select_with_sector_cap(ranked):
    max_names_per_sector = int(TOP_N * MAX_SECTOR_WEIGHT)

    selected = []
    sector_counts = {}

    for _, row in ranked.iterrows():
        sector = row["Sector"]
        current_count = sector_counts.get(sector, 0)

        if current_count < max_names_per_sector:
            selected.append(row)
            sector_counts[sector] = current_count + 1

        if len(selected) >= TOP_N:
            break

    selected_df = pd.DataFrame(selected)

    if selected_df.empty:
        return selected_df

    selected_df = selected_df.reset_index(drop=True)
    selected_df["PortfolioRank"] = selected_df.index + 1

    return selected_df


def load_current_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return pd.DataFrame(columns=["Ticker", "Name"])

    df = pd.read_csv(PORTFOLIO_FILE)

    if "Ticker" not in df.columns:
        raise ValueError("portfolio.csv must contain a Ticker column.")

    if "Name" not in df.columns:
        df["Name"] = ""

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()

    return df[["Ticker", "Name"]]


def save_new_portfolio(top_portfolio):
    new_portfolio = top_portfolio[["Ticker", "Name"]].copy()
    new_portfolio.to_csv(PORTFOLIO_FILE, index=False)


def build_changes(current_portfolio, new_portfolio):
    current_tickers = set(current_portfolio["Ticker"].tolist())
    new_tickers = set(new_portfolio["Ticker"].tolist())

    sells = sorted(list(current_tickers - new_tickers))
    buys = sorted(list(new_tickers - current_tickers))
    holds = sorted(list(current_tickers & new_tickers))

    return sells, buys, holds


def build_html_email(run_date, spy_momentum, mode, top_portfolio, current_portfolio, sells, buys, holds):
    if mode == "RISK-OFF":
        header_colour = "#b00020"
        action = "ACTION REQUIRED: CLOSE ALL POSITIONS"
        summary = "SPY blended momentum is negative or zero. The system is risk-off."
    else:
        header_colour = "#0b6b3a"
        action = "MONTHLY REBALANCE REQUIRED"
        summary = "SPY blended momentum remains positive. Rebalance to the new top 10 portfolio."

    name_map = dict(zip(top_portfolio["Ticker"], top_portfolio["Name"])) if not top_portfolio.empty else {}
    old_name_map = dict(zip(current_portfolio["Ticker"], current_portfolio["Name"])) if not current_portfolio.empty else {}

    def action_rows(tickers, label, colour):
        rows = ""
        for ticker in tickers:
            name = name_map.get(ticker, old_name_map.get(ticker, ""))
            rows += f"""
            <tr>
                <td style="color:{colour};"><strong>{label}</strong></td>
                <td><strong>{ticker}</strong></td>
                <td>{name}</td>
            </tr>
            """
        return rows

    changes_rows = ""
    changes_rows += action_rows(sells, "SELL", "#b00020")
    changes_rows += action_rows(buys, "BUY", "#0b6b3a")
    changes_rows += action_rows(holds, "HOLD", "#555555")

    if not changes_rows:
        changes_rows = """
        <tr>
            <td colspan="3"><strong>No changes required.</strong></td>
        </tr>
        """

    top_rows = ""

    if top_portfolio is not None and not top_portfolio.empty:
        for _, row in top_portfolio.iterrows():
            top_rows += f"""
            <tr>
                <td>{int(row["PortfolioRank"])}</td>
                <td><strong>{row["Ticker"]}</strong></td>
                <td>{row["Name"]}</td>
                <td>{row["Sector"]}</td>
                <td>{row["Momentum6M"]:.2%}</td>
                <td>{row["Momentum12MEx1M"]:.2%}</td>
                <td><strong>{row["BlendedMomentum"]:.2%}</strong></td>
            </tr>
            """

    top_table = ""

    if top_rows:
        top_table = f"""
        <h2>New Top 10 Momentum Portfolio</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Ticker</th>
                <th>Name</th>
                <th>Sector</th>
                <th>6M Momentum</th>
                <th>12M Ex-1M</th>
                <th>Blended Momentum</th>
            </tr>
            {top_rows}
        </table>
        """

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                color: #222222;
                line-height: 1.5;
                background: #ffffff;
            }}
            .container {{
                max-width: 950px;
                margin: auto;
                padding: 20px;
            }}
            .header {{
                background: {header_colour};
                color: #ffffff;
                padding: 20px;
                border-radius: 8px;
            }}
            .header h1 {{
                margin: 0 0 8px 0;
                font-size: 26px;
            }}
            .header h2 {{
                margin: 0;
                font-size: 20px;
            }}
            .box {{
                background: #f5f5f5;
                padding: 16px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 15px;
            }}
            th, td {{
                border: 1px solid #dddddd;
                padding: 8px;
                text-align: left;
                font-size: 14px;
            }}
            th {{
                background: #eeeeee;
            }}
            .footer {{
                margin-top: 25px;
                font-size: 13px;
                color: #666666;
            }}
        </style>
    </head>
    <body>
        <div class="container">

            <div class="header">
                <h1>Monthly Momentum Rebalance</h1>
                <h2>{action}</h2>
            </div>

            <div class="box">
                <p><strong>Run date:</strong> {run_date}</p>
                <p><strong>SPY blended momentum:</strong> {spy_momentum:.2%}</p>
                <p><strong>Mode:</strong> {mode}</p>
                <p>{summary}</p>
            </div>

            <h2>Required Actions</h2>
            <table>
                <tr>
                    <th>Action</th>
                    <th>Ticker</th>
                    <th>Name</th>
                </tr>
                {changes_rows}
            </table>

            {top_table}

            <div class="footer">
                <p>
                    portfolio.csv has been updated automatically to the latest top 10 holdings.
                    You only need to make the trades shown above.
                </p>
            </div>

        </div>
    </body>
    </html>
    """


def run_monthly_rebalance():
    run_date = datetime.now().strftime("%Y-%m-%d")

    current_portfolio = load_current_portfolio()

    sp500_table = get_sp500_table()

    tickers = sp500_table["Ticker"].tolist()
    sector_map = dict(zip(sp500_table["Ticker"], sp500_table["Sector"]))
    name_map = dict(zip(sp500_table["Ticker"], sp500_table["Name"]))

    all_tickers = sorted(list(set(tickers + ["SPY"])))

    data = download_data(all_tickers)

    spy_df = get_ticker_data(data, "SPY")

    if spy_df is None or spy_df.empty:
        raise ValueError("Could not retrieve SPY data.")

    spy_components = calculate_momentum_components(spy_df["Close"])

    if spy_components is None:
        raise ValueError("Could not calculate SPY blended momentum.")

    spy_momentum = spy_components["BlendedMomentum"]

    mode = "RISK-ON" if spy_momentum > 0 else "RISK-OFF"

    results = []

    for ticker in tickers:
        df = get_ticker_data(data, ticker)

        if df is None or df.empty:
            continue

        if "Close" not in df.columns or "Volume" not in df.columns:
            continue

        close = df["Close"].dropna()
        volume = df["Volume"].dropna()

        if len(close) < MIN_VALID_DAYS or len(volume) < MIN_VALID_DAYS:
            continue

        latest_price = close.iloc[-1]
        avg_dollar_volume = (close * volume).rolling(20).mean().iloc[-1]

        if latest_price < MIN_PRICE:
            continue

        if pd.isna(avg_dollar_volume) or avg_dollar_volume < MIN_DOLLAR_VOLUME:
            continue

        if not passes_data_quality_filters(close):
            continue

        components = calculate_momentum_components(close)

        if components is None:
            continue

        results.append({
            "Ticker": ticker,
            "Name": name_map.get(ticker, "Unknown"),
            "Sector": sector_map.get(ticker, "Unknown"),
            "Momentum6M": components["Momentum6M"],
            "Momentum12MEx1M": components["Momentum12MEx1M"],
            "BlendedMomentum": components["BlendedMomentum"],
        })

    ranked = pd.DataFrame(results)

    if ranked.empty:
        top_portfolio = pd.DataFrame(columns=[
            "PortfolioRank",
            "Ticker",
            "Name",
            "Sector",
            "Momentum6M",
            "Momentum12MEx1M",
            "BlendedMomentum",
        ])
    else:
        ranked = ranked.sort_values("BlendedMomentum", ascending=False).reset_index(drop=True)
        ranked["RawRank"] = ranked.index + 1
        top_portfolio = select_with_sector_cap(ranked)

    if mode == "RISK-OFF":
        new_portfolio = pd.DataFrame(columns=["Ticker", "Name"])
        sells = sorted(current_portfolio["Ticker"].tolist())
        buys = []
        holds = []
    else:
        new_portfolio = top_portfolio[["Ticker", "Name"]].copy()
        sells, buys, holds = build_changes(current_portfolio, new_portfolio)

    html_body = build_html_email(
        run_date=run_date,
        spy_momentum=spy_momentum,
        mode=mode,
        top_portfolio=top_portfolio,
        current_portfolio=current_portfolio,
        sells=sells,
        buys=buys,
        holds=holds,
    )

    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_body)

    new_portfolio.to_csv(PORTFOLIO_FILE, index=False)

    print("==============================")
    print("MONTHLY MOMENTUM REBALANCE")
    print("==============================")
    print(f"Run date: {run_date}")
    print(f"SPY blended momentum: {spy_momentum:.2%}")
    print(f"Mode: {mode}")
    print(f"SELL: {', '.join(sells) if sells else 'None'}")
    print(f"BUY: {', '.join(buys) if buys else 'None'}")
    print(f"HOLD: {', '.join(holds) if holds else 'None'}")
    print(f"HTML email saved to: {HTML_FILE}")
    print(f"{PORTFOLIO_FILE} updated.")


if __name__ == "__main__":
    run_monthly_rebalance()