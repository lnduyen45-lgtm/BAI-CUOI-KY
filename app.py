import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tối ưu hóa Danh mục Đầu tư", page_icon="📈", layout="wide")

st.title("📈 Ứng dụng Tối ưu hóa Danh mục Đầu tư (RSI + MACD)")
st.markdown("""
Ứng dụng này mô phỏng và kiểm định chiến lược giao dịch kết hợp chỉ báo RSI và MACD trên dữ liệu chứng khoán HOSE (2020 - 2023).
*   **Giai đoạn Huấn luyện (2020)**: Tìm và chọn ra 5 cổ phiếu tốt nhất dựa trên tỷ lệ Sharpe.
*   **Giai đoạn Kiểm định (2021-2023)**: Đánh giá chiến lược trên 5 cổ phiếu này, so sánh với các danh mục Benchmark như VN-INDEX, Mua & Giữ (Buy & Hold), và 1/N.
""")

st.sidebar.header("📂 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu CSV (VD: HOSE_2020_2023.csv)", type=["csv"])

@st.cache_data
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['ticker'] = df['ticker'].astype(str).str.upper()
    
    vnindex_df = df[df['ticker'] == 'VNINDEX'].sort_values('date').reset_index(drop=True)
    stocks_df = df[df['ticker'] != 'VNINDEX'].sort_values(['ticker', 'date']).reset_index(drop=True)
    
    stocks_df['Daily_Return'] = stocks_df.groupby('ticker')['adj_close'].pct_change()
    vnindex_df['VNINDEX_Return'] = vnindex_df['adj_close'].pct_change()
    
    return stocks_df, vnindex_df

def calc_rsi(x, period=14):
    delta = x.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_macd_line(x):
    return x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean()

@st.cache_data
def generate_signals(_stocks_df):
    stocks_df = _stocks_df.copy()
    stocks_df['RSI'] = stocks_df.groupby('ticker')['adj_close'].transform(calc_rsi)
    stocks_df['MACD'] = stocks_df.groupby('ticker')['adj_close'].transform(calc_macd_line)
    stocks_df['Signal_Line'] = stocks_df.groupby('ticker')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    
    buy_cond = (stocks_df['MACD'] > stocks_df['Signal_Line']) & (stocks_df['RSI'] > 50)
    rsi_cross_down = (stocks_df['RSI'] < 70) & (stocks_df.groupby('ticker')['RSI'].shift(1) >= 70)
    macd_cross_down = (stocks_df['MACD'] < stocks_df['Signal_Line'])
    sell_cond = macd_cross_down | rsi_cross_down
    
    stocks_df['Signal_Event'] = 0
    stocks_df.loc[buy_cond, 'Signal_Event'] = 1
    stocks_df.loc[sell_cond, 'Signal_Event'] = -1
    
    stocks_df['State'] = stocks_df['Signal_Event'].replace(0, np.nan)
    stocks_df['State'] = stocks_df.groupby('ticker')['State'].ffill().fillna(0)
    stocks_df['Position'] = np.where(stocks_df['State'] == 1, 1, 0)
    stocks_df['Signal'] = stocks_df.groupby('ticker')['Position'].shift(1).fillna(0)
    stocks_df['Strategy_Return'] = stocks_df['Signal'] * stocks_df['Daily_Return']
    
    return stocks_df

@st.cache_data
def backtest(stocks_df, vnindex_df):
    train_df = stocks_df[stocks_df['date'].dt.year == 2020]
    train_perf = train_df.groupby('ticker').agg(
        Ret=('Strategy_Return', lambda x: (1+x).prod() - 1),
        Vol=('Strategy_Return', lambda x: x.std() * np.sqrt(252))
    )
    train_perf['Sharpe'] = np.where(train_perf['Vol'] > 0, train_perf['Ret'] / train_perf['Vol'], 0)
    top_5_tickers = train_perf.sort_values('Sharpe', ascending=False).head(5).index.tolist()
    
    test_df = stocks_df[(stocks_df['date'].dt.year >= 2021)].copy()
    benchmark_1n = test_df.groupby('date')['Daily_Return'].mean().reset_index()
    benchmark_1n.rename(columns={'Daily_Return': '1N_Return'}, inplace=True)
    
    test_top5 = test_df[test_df['ticker'].isin(top_5_tickers)]
    port_strat = test_top5.groupby('date')['Strategy_Return'].mean().reset_index()
    port_bh = test_top5.groupby('date')['Daily_Return'].mean().reset_index()
    port_bh.rename(columns={'Daily_Return': 'BH_Return'}, inplace=True)
    
    portfolio = pd.merge(port_strat, port_bh, on='date', how='left')
    portfolio = pd.merge(portfolio, benchmark_1n, on='date', how='left')
    
    vn_benchmark = vnindex_df[vnindex_df['date'].dt.year >= 2021][['date', 'VNINDEX_Return']]
    portfolio = pd.merge(portfolio, vn_benchmark, on='date', how='left').set_index('date')
    portfolio.fillna(0, inplace=True)
    
    avg_signal = test_top5.groupby('date')['Signal'].mean()
    
    return top_5_tickers, portfolio, avg_signal

def calc_trade_metrics(signals, returns):
    trades = []
    current_return = 0
    in_trade = False
    for s, r in zip(signals, returns):
        if s > 0:
            in_trade = True
            current_return = (1 + current_return) * (1 + r) - 1
        elif s == 0 and in_trade:
            trades.append(current_return)
            current_return = 0
            in_trade = False
    if in_trade: trades.append(current_return)

    trades = np.array(trades)
    if len(trades) > 0:
        win_rate = len(trades[trades > 0]) / len(trades)
        best_trade = trades.max()
        worst_trade = trades.min()
    else:
        win_rate = best_trade = worst_trade = 0
    return win_rate, best_trade, worst_trade

def portfolio_metrics(returns_series, signals_series=None):
    cum_ret = (1 + returns_series).cumprod()
    total_ret = cum_ret.iloc[-1] - 1 if not cum_ret.empty else 0
    vol = returns_series.std() * np.sqrt(252)
    sharpe = total_ret / vol if vol > 0 else 0
    roll_max = cum_ret.cummax()
    max_dd = (cum_ret / roll_max - 1).min() if not cum_ret.empty else 0
    downside_vol = returns_series[returns_series < 0].std() * np.sqrt(252)
    sortino = total_ret / downside_vol if downside_vol > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd < 0 else 0
    exposure = len(returns_series[returns_series != 0]) / len(returns_series) if len(returns_series) > 0 else 0

    metrics = {
        'Return (%)': total_ret * 100,
        'Volatility (%)': vol * 100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_dd * 100,
        'Calmar Ratio': calmar,
        'Exposure Time (%)': exposure * 100
    }

    if signals_series is not None:
        win, best, worst = calc_trade_metrics(signals_series, returns_series)
        metrics['Win Rate (%)'] = win * 100
        metrics['Best Trade (%)'] = best * 100
        metrics['Worst Trade (%)'] = worst * 100
    else:
        metrics['Win Rate (%)'] = np.nan
        metrics['Best Trade (%)'] = np.nan
        metrics['Worst Trade (%)'] = np.nan

    return pd.Series(metrics)

if uploaded_file is None:
    st.info("👋 Xin chào! Để bắt đầu, vui lòng tải lên file dữ liệu CSV (HOSE_2020_2023.csv) từ thanh công cụ bên trái (Sidebar).")
else:
    with st.spinner("Đang xử lý dữ liệu, vui lòng đợi..."):
        try:
            raw_stocks, raw_vnindex = load_and_preprocess_data(uploaded_file)
            processed_stocks = generate_signals(raw_stocks)
            top_5, portfolio, avg_signal = backtest(processed_stocks, raw_vnindex)
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình đọc hoặc xử lý dữ liệu: {e}")
            st.stop()

    st.success("Xử lý dữ liệu hoàn tất!")

    # Hiển thị Top 5
    st.subheader("🏆 Top 5 Cổ Phiếu Dựa Trên Huấn Luyện (2020)")
    st.info(f"Các mã được chọn: **{', '.join(top_5)}**")

    # Biểu đồ hiệu suất
    st.subheader("📈 Hiệu suất Tích lũy (Cumulative Returns) 2021-2023")

    portfolio['Cum_Strategy'] = (1 + portfolio['Strategy_Return']).cumprod()
    portfolio['Cum_BH'] = (1 + portfolio['BH_Return']).cumprod()
    portfolio['Cum_VNINDEX'] = (1 + portfolio['VNINDEX_Return']).cumprod()
    portfolio['Cum_1N'] = (1 + portfolio['1N_Return']).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['Cum_Strategy'], mode='lines', name='Chiến lược (RSI+MACD)'))
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['Cum_BH'], mode='lines', name='Mua & Giữ (Top 5)'))
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['Cum_VNINDEX'], mode='lines', name='VN-INDEX'))
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['Cum_1N'], mode='lines', name='1/N HOSE'))

    fig.update_layout(
        title="So sánh Lợi nhuận Tích lũy Chiến lược vs Các Benchmark",
        xaxis_title="Thời gian",
        yaxis_title="Lợi nhuận tích lũy",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bảng 1: Phân tích hiệu quả toàn giai đoạn
    st.subheader("📊 BẢNG 1: Phân tích Hiệu quả và Rủi ro Danh mục (2021-2023)")
    df_metrics = pd.DataFrame({
        '1. Tối ưu RSI+MACD': portfolio_metrics(portfolio['Strategy_Return'], avg_signal),
        '2. Buy & Hold (Top 5)': portfolio_metrics(portfolio['BH_Return']),
        '3. Benchmark VNINDEX': portfolio_metrics(portfolio['VNINDEX_Return']),
        '4. Benchmark 1/N HOSE': portfolio_metrics(portfolio['1N_Return'])
    }).T

    st.dataframe(df_metrics.round(3).style.highlight_max(subset=['Return (%)', 'Sharpe Ratio', 'Win Rate (%)'], color='lightgreen'))

    # Bảng 2: Kiểm định đa giai đoạn (Regime Testing)
    st.subheader("🔍 BẢNG 2: Kiểm định Đa Giai đoạn (Regime Testing)")
    regimes = {
        '1. Giai đoạn tăng trưởng (Bull: 2021 - T3/2022)': ('2021-01-01', '2022-03-31'),
        '2. Giai đoạn suy giảm (Bear: T4/2022 - T11/2022)': ('2022-04-01', '2022-11-15'),
        '3. Giai đoạn phục hồi (Sideway: T12/2022 - 2023)': ('2022-11-16', '2023-12-31')
    }

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for i, (name, (start, end)) in enumerate(regimes.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            port_regime = portfolio.loc[start:end]
            if not port_regime.empty:
                regime_metrics = pd.DataFrame({
                    'Tối ưu RSI+MACD': portfolio_metrics(port_regime['Strategy_Return']),
                    'Buy & Hold Top 5': portfolio_metrics(port_regime['BH_Return']),
                    'VNINDEX': portfolio_metrics(port_regime['VNINDEX_Return'])
                }).loc[['Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio']]
                st.dataframe(regime_metrics.round(3))

    # Bảng 3: Walk-Forward Out-of-sample
    st.subheader("🚀 BẢNG 3: Kiểm định Độ tin cậy Walk-Forward (Out-of-Sample)")
    st.markdown("Đánh giá hiệu suất trượt cửa sổ từng năm độc lập trên danh mục Top 5:")

    walk_forward_results = []
    for year in [2021, 2022, 2023]:
        year_data = portfolio[portfolio.index.year == year]
        if not year_data.empty:
            strat_ret = (1 + year_data['Strategy_Return']).prod() - 1
            bh_ret = (1 + year_data['BH_Return']).prod() - 1
            vn_ret = (1 + year_data['VNINDEX_Return']).prod() - 1
            status = "✅ VƯỢT TRỘI" if strat_ret > vn_ret else "❌ THẤP HƠN"
            
            walk_forward_results.append({
                "Năm": year,
                "Lợi nhuận Chiến lược (%)": round(strat_ret * 100, 2),
                "Lợi nhuận Mua & Giữ (%)": round(bh_ret * 100, 2),
                "Lợi nhuận VN-INDEX (%)": round(vn_ret * 100, 2),
                "Đánh giá vs VN-INDEX": status
            })

    if walk_forward_results:
        st.table(pd.DataFrame(walk_forward_results))
