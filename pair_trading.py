import pandas as pd
import math
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date,datetime,timedelta
import plotly.express as px
import requests
# from github_api import *

st.set_page_config(layout="wide")

# Read lot sizes and margin required from CSV file
lot_size_df = pd.read_csv('./lots.csv')
stock = tuple(lot_size_df['SYMBOL'].tolist())
print(stock)
# Input values from Streamlit UI
stockA = st.selectbox('Enter Stock A symbol (e.g., HDFC.NS):',stock)
# lotA = st.number_input('Enter lot size for this stock:')

stockB = st.selectbox('Enter Stock B symbol (e.g., HDFCBANK.NS):',stock)
# lotB = st.number_input('Enter lot size for this stock:')
default_date = date(2010, 1, 1)
default_end_date = date.today() + timedelta(days=1)

# Calculate the default start date as 10 years ago from the default end date
# default_start_date = default_end_date - timedelta(days=365 * 10)

start_date = st.date_input('Enter Start Date:',value=default_date)
current_date = datetime.now()

end_date_limit = current_date - timedelta(days=365 * 20)
end_date = st.date_input('Enter End Date:', min_value=current_date - timedelta(days=365 * 20), max_value=default_end_date)

default_window_size = 100  # Default value for window size select slider
default_entry_zscore = 3.0  # Default value for entry z-score slider
default_exit_zscore = 2.0 

window_size = st.select_slider('Select Window Size:', options=[10,30,60,100,250], value=default_window_size)



entry_zscore = st.slider('Entry Z-Score', min_value=-5.0, max_value=5.0, value=default_entry_zscore, step=0.1)
exit_zscore = st.slider('Exit Z-Score', min_value=-5.0, max_value=5.0, value=default_exit_zscore, step=0.1)
# margin_required = st.number_input('Enter Margin required for this trade:')
# lotA = lot_size_df.loc[lot_size_df['SYMBOL'] == stockA, 'LOT'].values[0]
# lotB = lot_size_df.loc[lot_size_df['SYMBOL'] == stockB, 'LOT'].values[0]
# margin_requiredA = lot_size_df.loc[lot_size_df['SYMBOL'] == stockA, 'Margin'].values[0]
# margin_requiredB = lot_size_df.loc[lot_size_df['SYMBOL'] == stockB, 'Margin'].values[0]
# margin_required = margin_requiredA+margin_requiredB
margin_requiredA=500000
margin_requiredB=500000
margin_required = 500000
# st.write('Lot Size for Stock A:', lotA)
# st.write('Lot Size for Stock B:', lotB)
st.write('Marzin Required:', margin_required)

# Download stock data
data = yf.download(stockA.strip(), start=start_date, end=end_date)['Adj Close']
data2 = yf.download(stockB.strip(), start=start_date, end=end_date)['Adj Close']
data_new = pd.concat([data, data2], axis=1, keys=['stockA', 'stockB'])
data_new['ratio'] = np.log10(data_new['stockA'] / data_new['stockB'])
data_new['average'] = data_new['ratio'].rolling(int(window_size)).mean()
data_new['stdev'] = data_new['ratio'].rolling(int(window_size)).std()
data_new['Zscore'] = (data_new['ratio'] - data_new['average']) / data_new['stdev']
df = data_new.dropna()
if entry_zscore > 0 :
    df['entry'] = np.where((df['Zscore'] >= entry_zscore), 1, 0)
    df['exit'] = np.where(df['Zscore'] <= exit_zscore, 1, 0)
else:
    df['entry'] = np.where((df['Zscore'] <= entry_zscore), 1, 0)
    df['exit'] = np.where((df['Zscore']) >= exit_zscore, 1, 0)


df['ApnL'] = 0
df['BpnL'] = 0
df['Total_%'] = 0


tolerance = 0.1

# Calculate the maximum Z-score and its count, including values within the tolerance range
max_zscore = df['Zscore'].max()
max_negative_zscore = df['Zscore'].min()

max_zscore_count_positive = df[(df['Zscore'] >= max_zscore - tolerance) & (df['Zscore'] <= max_zscore + tolerance)].shape[0]
max_negative_zscore_count = df[(df['Zscore'] >= max_negative_zscore - tolerance) & (df['Zscore'] <= max_negative_zscore + tolerance)].shape[0]


def zscore1(series,window):
    return (series - series.rolling(window=window).mean()) / series.rolling(window=window).std()


# def zscore(data,data2,window_size):
#     print(data)
#     data_new1 = pd.DataFrame()
#     data_new1['ratio'] = np.log10(data / data2)
#     data_new1['average'] = data_new1['ratio'].rolling(int(window_size)).mean()
#     data_new1['stdev'] = data_new1['ratio'].rolling(int(window_size)).std()
#     data_new1['Zscore'] = (data_new1['ratio'] - data_new1['average']) / data_new1['stdev']
#     last_z_score = data_new1['Zscore'].iloc[-1]
#     return last_z_score

def correlation(df1,df2):
    corr_250d = (df1.tail(250)).corr(df2.tail(250))
    corr_100d = (df1.tail(100)).corr(df2.tail(100))
    corr_10d = (df1.tail(10)).corr(df2.tail(10))

    return corr_250d,corr_100d,corr_10d


col1, col2, col3 = st.columns(3)

with col1:
    live_zscore_60days = zscore1(data_new['ratio'],60)
    last_zscore_60 = live_zscore_60days[-1]
    st.write('Live Zscore 60days', last_zscore_60)

with col2:
    live_zscore_100days = zscore1(data_new['ratio'],100)
    last_zscore_100 = live_zscore_100days[-1]
    st.write('Live Zscore 100days', last_zscore_100)

with col3:
    live_zscore_250days = zscore1(data_new['ratio'],250)
    last_zscore_250 = live_zscore_250days[-1]
    st.write('Live Zscore 250days', last_zscore_250)



# Calculate correlations

corr_250d, corr_100d, corr_10d = correlation(data, data2)  # Replace 'col1' and 'col2' with your column names
col1, col2, col3 = st.columns(3)

with col1:
    st.write('Correlation (250 days):', corr_250d)
with col2:
    st.write('Correlation (100 days):', corr_100d)
with col3:
    st.write('Correlation (10 days):', corr_10d)

col1, col2 = st.columns(2)  # You can also use st.columns(2) in newer versions of Streamlit


# with col3:
#     trade_stats_table = st.table(trade_stats_df.style.hide_index())

# Display the max Z-score and its count on positive and negative sides
with col1:
    st.write('Maximum Positive Z-Score:', max_zscore)
    st.write('Maximum Negative Z-Score:', max_negative_zscore)
with col2:
    st.write('Number of times Z-Score reached close to the maximum value (Positive side):', max_zscore_count_positive)
    st.write('Number of times Z-Score reached close to the maximum negative value:', max_negative_zscore_count)





# Create a list to store the dataframes and trade metrics
result_dfs = []
trade_metrics = []

def calculate(entry_index, current_index, entry_zscore,margin_required):
    capitalA = capitalB = capital = margin_required
    # capitalB = margin_requiredB
    # capital = margin_required

    entryA = df.iloc[entry_index]['stockA']
    entryB = df.iloc[entry_index]['stockB']
    currentA = df.iloc[current_index]['stockA']
    currentB = df.iloc[current_index]['stockB']
    qtyA = margin_requiredA//entryA
    qtyB = margin_requiredB//entryB
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
        total_return = 0.0
    else:
        daily_return_A = round((apnl / capitalA) * 100,2)
        daily_return_B = round((bpnl / capitalB) * 100,2)
        total_return = round((pnl/capital)*100,2)
    return apnl, bpnl, pnl, daily_return_A, daily_return_B, total_return,qtyA,qtyB

hold = 0
df['pnl'] = 0
total_trades = 0

for i, b in enumerate(df.iterrows()):
    if hold == 0 and b[1]['entry'] == 1:
        total_trades += 1
        entry_index = i
        entry_date = df.index[i].date()  # Add entry date
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
        apnl, bpnl, pnl, stockA_return, stockB_return,total_return,qtyA,qtyB= calculate(entry_index, i, zscore,margin_required)
        df.loc[df.index[i], ['ApnL']], df.loc[df.index[i], ['BpnL']] = apnl, bpnl
        df.loc[df.index[i], ['pnl']] = pnl
        df.loc[df.index[i], ['StockA_return']] = round(stockA_return,2)
        df.loc[df.index[i], ['StockB_return']] = round(stockB_return,2)
        df.loc[df.index[i], ['Total_%']] = round(total_return,2)
        hold = 0
        result_dfs.append(df.iloc[entry_index:exit_index+1])
        total_profit = "{:.2f}".format(float(apnl) + float(bpnl))
        total_profit_percentage = "{:.2f}".format(((apnl + bpnl)/margin_required)*100)
        avg_loss_percentage = round(df.iloc[entry_index:exit_index + 1]['pnl'].where(df['pnl'] < 0).mean() / float(margin_required) * 100, 2)
        avg_profit_percentage = round(df.iloc[entry_index:exit_index + 1]['pnl'].where(df['pnl'] > 0).mean() / float(margin_required) * 100, 2)
        max_loss_percantage = round(df.iloc[entry_index:exit_index+1]['pnl'].min()/float(margin_required) * 100, 2,),
        maxl_profit_percantage =round(df.iloc[entry_index:exit_index+1]['pnl'].max()/float(margin_required) * 100, 2,),
        trade_metrics.append({
            'Trade': len(trade_metrics) + 1,
            'Entry Date': entry_date,  # Add entry date
            'Holding Days': days_between,
            'Total Profit': total_profit,
            'Total Profit_%': total_profit_percentage,
            'Average Loss_%': avg_loss_percentage,
            'Average Profit_%': avg_profit_percentage,
            'Max Loss_%':max_loss_percantage,
            'Max Profit_%':maxl_profit_percantage,
            'Average Loss': df.iloc[entry_index:exit_index+1]['pnl'].where(df['pnl'] < 0).mean(),
            'Average Profit': df.iloc[entry_index:exit_index+1]['pnl'].where(df['pnl'] > 0).mean(),
            'Maximum Loss': df.iloc[entry_index:exit_index+1]['pnl'].min(),
            'Maximum Profit': df.iloc[entry_index:exit_index+1]['pnl'].max(),
        })
    if hold == 1:
        apnl, bpnl, pnl, stockA_return, stockB_return, total_return,qtyA,qtyB= calculate(entry_index, i, zscore,margin_required)
        df.loc[df.index[i], ['ApnL']], df.loc[df.index[i], ['BpnL']] = apnl, bpnl
        df.loc[df.index[i], ['pnl']] = pnl
        df.loc[df.index[i], ['StockA_return']] = round(stockA_return,2)
        df.loc[df.index[i], ['StockB_return']] = round(stockB_return,2)
        df.loc[df.index[i], ['Total_%']] = round(total_return,2)
# display_df = result_dfs.drop(['average','stdev','entry','exit','holding_days'], axis=1)
# Display the result dataframes and trade metrics

with st.container():
    st.subheader('Trade Metrics')
    trade_metrics_df = pd.DataFrame(trade_metrics).set_index('Trade')
    st.write(trade_metrics_df)



# Create variables to store trade statistics
total_trades = len(trade_metrics)  # Total number of trades
winning_trades = sum(1 for trade in trade_metrics if float(trade['Total Profit']) > 0)  # Count of winning trades
losing_trades = sum(1 for trade in trade_metrics if float(trade['Total Profit']) < 0)  # Count of losing trades
winning_accuracy = (winning_trades / total_trades) * 100 if total_trades > 0 else 0  # Winning accuracy percentage

new_trade_stats = {
    'Total Trades': [total_trades],  # Update with your total_trades value
    'Winning Trades': [winning_trades],  # Update with your winning_trades value
    'Losing Trades': [losing_trades],  # Update with your losing_trades value
    'Winning Accuracy (%)': [winning_accuracy],  # Update with your winning_accuracy value
}

trade_stats_df = pd.DataFrame(new_trade_stats)
# Update the values in the trade statistics table
col1,col2 = st.columns(2)
with col1:
    st.subheader('Trade Statistics')
    st.table(trade_stats_df)

# # Assuming you have two sets of data: data_new_2000 and data_new_5000
# data_new_2000 = data_new['ratio'].tail(1250)
# data_new_5000 = data_new['ratio'].tail(3750)

# # Create a Matplotlib figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Plot the first series (last 2000 values) on the primary y-axis
# ax1.plot(data_new_2000.index, data_new_2000, label='Last 5years', color='blue')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Last 5years', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')

# # Create a secondary y-axis for the second series
# ax2 = ax1.twinx()

# # Plot the second series (last 5000 values) on the secondary y-axis
# ax2.plot(data_new_5000.index, data_new_5000, label='Last 15years ', color='red')
# ax2.set_ylabel('Last 15years', color='red')
# ax2.tick_params(axis='y', labelcolor='red')

# # Customize the chart
# plt.title('Comparison of Last 5 and 15 years ')

# # Display the Matplotlib chart using Streamlit
# st.pyplot(fig)



# Assuming you have two sets of data: data_new_2000 and data_new_5000
data_new_2000 = data_new['ratio'].tail(1250)
data_new_5000 = data_new['ratio'].tail(3750)

# Create a Plotly figure
fig = px.line(data_new_2000, x=data_new_2000.index, y='ratio', labels={'ratio': 'Last 5years Values'})
fig.add_scatter(x=data_new_5000.index, y=data_new_5000, mode='lines', name='Last 15 years Values')

# Customize the chart
fig.update_layout(title='Comparison of Last 2000 and Last 5000 Values')
fig.update_layout(width=1200, height=400)

# Display the Plotly chart using Streamlit
st.plotly_chart(fig)



# st.line_chart(data_new['ratio'].tail(2000))

    # Display the result dataframes and trade metrics
# Plotting the 'ratio' and 'Zscore' columns
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Zscore'], label='Z-score', color='r')

# Drawing a line for the entry Z-score
entry_zscore_line = df[df['entry'] == 1]['Zscore']
exit_zscore_line = df[df['exit'] == 1]['Zscore']

plt.axhline(y=entry_zscore_line.values[0], color='g', linestyle='--', label='Entry Z-score')
plt.axhline(y=exit_zscore_line.values[0], color='orange', linestyle='--', label='Exit Z-score')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Z-score with Entry Z-score')
plt.legend()
plt.grid(True)
st.pyplot(plt)




# Filter and print trades with positive entry Z-score
positive_entry_trades = [result_df.drop(['average', 'stdev', 'entry', 'exit', 'holding_days'], axis=1) for result_df in result_dfs if result_df.iloc[0]['Zscore'] > 0]

# Print trade details for positive entry Z-score
st.subheader('Trades with Positive Entry Z-Score')
if not positive_entry_trades:
    st.write('No trades with positive entry Z-Score')
else:
    for i, trade_df in enumerate(positive_entry_trades):
        with st.container():
            st.subheader(f'Trade {i+1}')
            st.write(trade_df)


# Filter and print trades with negative entry Z-score
negative_entry_trades = [result_df.drop(['average', 'stdev', 'entry', 'exit', 'holding_days'], axis=1) for result_df in result_dfs if result_df.iloc[0]['Zscore'] < 0]

# Print trade details for negative entry Z-score
st.subheader('Trades with Negative Entry Z-Score')
if not negative_entry_trades:
    st.write('No trades with negative entry Z-Score')
else:
    for i, trade_df in enumerate(negative_entry_trades):
        with st.container():
            st.subheader(f'Trade {i+1}')
            st.write(trade_df)
