import pandas as pd
import math
import numpy as np
import yfinance as yf
import streamlit as st
# Read lot sizes and margin required from CSV file
lot_size_df = pd.read_csv('./lot_sizes.csv')
# Input values from Streamlit UI
stockA = st.text_input('Enter Stock A symbol (e.g., HDFC.NS):')
# lotA = st.number_input('Enter lot size for this stock:')

stockB = st.text_input('Enter Stock B symbol (e.g., HDFCBANK.NS):')
# lotB = st.number_input('Enter lot size for this stock:')

start_date = st.date_input('Enter Start Date:')
end_date = st.date_input('Enter End Date:')
window_size = st.select_slider('Select Window Size:', options=[60, 100, 250])
entry_zscore = st.slider('Entry Z-Score', min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
exit_zscore = st.slider('Exit Z-Score', min_value=-5.0, max_value=5.0, value=0.4, step=0.1)
# margin_required = st.number_input('Enter Margin required for this trade:')
lotA = lot_size_df.loc[lot_size_df['StockA'] == stockA, 'LotA'].values[0]
lotB = lot_size_df.loc[lot_size_df['StockB'] == stockB, 'LotB'].values[0]
margin_required = lot_size_df.loc[lot_size_df['StockA'] == stockA, 'Margin'].values[0]
st.write('Lot Size for Stock A:', lotA)
st.write('Lot Size for Stock B:', lotB)
st.write('Marzin Required:', margin_required)

# Download stock data
data = yf.download(stockA, start=start_date, end=end_date)['Adj Close']
data2 = yf.download(stockB, start=start_date, end=end_date)['Adj Close']
data_new = pd.concat([data, data2], axis=1, keys=['stockA', 'stockB'])
data_new['ratio'] = np.log10(data_new['stockA'] / data_new['stockB'])
data_new['average'] = data_new['ratio'].rolling(int(window_size)).mean()
data_new['stdev'] = data_new['ratio'].rolling(int(window_size)).std()
data_new['Zscore'] = (data_new['ratio'] - data_new['average']) / data_new['stdev']
df = data_new.dropna()
df['entry'] = np.where(np.abs(df['Zscore']) >= entry_zscore, 1, 0)
df['exit'] = np.where(np.abs(df['Zscore']) <= exit_zscore, 1, 0)
df['ApnL'] = 0
df['BpnL'] = 0

# Create a list to store the dataframes and trade metrics
result_dfs = []
trade_metrics = []

def calculate(entry_index, current_index, entry_zscore,lotA,lotB,margin_required):
    capital = margin_required
    entryA = df.iloc[entry_index]['stockA']
    entryB = df.iloc[entry_index]['stockB']
    currentA = df.iloc[current_index]['stockA']
    currentB = df.iloc[current_index]['stockB']
    qtyA = lotA
    qtyB = lotB

    if entry_zscore == 2:
        apnl = round((entryA - currentA) * qtyA, 2)
        bpnl = round((currentB - entryB) * qtyB, 2)
        pnl = apnl + bpnl
    else:
        apnl = round((currentA - entryA) * qtyA, 2)
        bpnl = round((entryB - currentB) * qtyB, 2)
        pnl = apnl + bpnl

    # Calculate daily return for stock A and B
    if current_index == entry_index or entry_index == 0:
        daily_return_A = 0.0
        daily_return_B = 0.0
    else:
        daily_return_A = round((apnl / capital) * 100,2)
        daily_return_B = round((bpnl / capital) * 100,2)

    return apnl, bpnl, pnl, daily_return_A, daily_return_B

hold = 0
df['pnl'] = 0
total_trades = 0

for i, b in enumerate(df.iterrows()):
    if hold == 0 and b[1]['entry'] == 1:
        total_trades += 1
        entry_index = i
        hold = 1
        if df.iloc[entry_index]['Zscore'] >= 2:
            zscore = 2
        else:
            zscore = -2
    if hold == 1 and b[1]['exit'] == 1:
        exit_index = i
        exit_date = df.index[i]
        days_between = (exit_index - entry_index)
        df.loc[exit_date, 'holding_days'] = days_between
        apnl, bpnl, pnl, stockA_return, stockB_return = calculate(entry_index, i, zscore,lotA,lotB,margin_required)
        df.loc[df.index[i], ['ApnL']], df.loc[df.index[i], ['BpnL']] = apnl, bpnl
        df.loc[df.index[i], ['pnl']] = pnl
        df.loc[df.index[i], ['StockA_return']] = round(stockA_return,2)
        df.loc[df.index[i], ['StockB_return']] = round(stockB_return,2)
        hold = 0
        result_dfs.append(df.iloc[entry_index:exit_index+1])
        trade_metrics.append({
            'Trade': len(result_dfs),
            'Total Profit': df.iloc[entry_index:exit_index+1]['pnl'].sum(),
            'Maximum Loss': df.iloc[entry_index:exit_index+1]['pnl'].min(),
            'Maximum Profit': df.iloc[entry_index:exit_index+1]['pnl'].max(),
            'Average Loss': df.iloc[entry_index:exit_index+1]['pnl'].where(df['pnl'] < 0).mean(),
            'Average Profit': df.iloc[entry_index:exit_index+1]['pnl'].where(df['pnl'] > 0).mean(),
            'Holding Days': days_between
        })
    if hold == 1:
        apnl, bpnl, pnl, stockA_return, stockB_return = calculate(entry_index, i, zscore,lotA,lotB,margin_required)
        df.loc[df.index[i], ['ApnL']], df.loc[df.index[i], ['BpnL']] = apnl, bpnl
        df.loc[df.index[i], ['pnl']] = pnl
        df.loc[df.index[i], ['StockA_return']] = round(stockA_return,2)
        df.loc[df.index[i], ['StockB_return']] = round(stockB_return,2)
# display_df = result_dfs.drop(['average','stdev','entry','exit','holding_days'], axis=1)
# Display the result dataframes and trade metrics
for i, result_df in enumerate(result_dfs):
    st.subheader(f'Trade {i+1}')
    st.write(result_df.drop(['average','stdev','entry','exit','holding_days'], axis=1))

st.subheader('Trade Metrics')
trade_metrics_df = pd.DataFrame(trade_metrics)
st.write(trade_metrics_df)
