###================================================================Data Cleansing, Preparation and VIX Index Calculation================================================================###



##----------------------------------------------------------------------------------------------------------------------------------------------------------------##
# Import necessary modules
import pandas as pd
import numpy as np
import datetime
import pytz
import requests

pd.options.mode.chained_assignment = None  # default='warn'
now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))

# Compute the days until expire and put those data in a new column in the table
def dateConvert(x):
    x = str(x)
    y = datetime.datetime.strptime(x, '%Y%m%d')
    modified = pytz.timezone('Asia/Shanghai').localize(y)  # Take time zone into consideration
    return modified

def getDays(x) -> int:
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))  # # Take time zone into consideration
    day = (x - now).days
    return day

def initialProcess(now, fileName):
    # Read csv file created by the main.py python script
    originalTable = pd.read_csv(fileName, 
                                header = None,
                                usecols = [i for i in range(0,11)])

    # Set the title names for the table
    originalTable.columns = ['Expiration Date', 'Download Time', 'Contract ID', 
                             'Bid Volume', 'Bid', 'Latest Price', 'Ask', 
                             'Ask Volume', 'Position', 'Change %', 'Strike']
    tempTable = originalTable[['Expiration Date','Contract ID', 'Strike', 'Bid', 'Ask']].copy()

    # Data cleansing and table transformation
    tempTable = tempTable[np.isfinite(originalTable['Expiration Date'])]
    tempTable['Expiration Date'] = tempTable['Expiration Date'].astype(int)
    tempTable = tempTable.sort_values('Expiration Date')
    tempTable = tempTable.groupby('Expiration Date').apply(lambda x: x.sort_values(['Strike'])).reset_index(drop=True)

    temp = tempTable["Expiration Date"].apply(dateConvert)
    Days = temp.apply(getDays)
    tempTable["Days"] = Days
    tempTable['Date'] = now
    tempTable['Date'] = pd.to_datetime(tempTable['Date'])

    # Select the part of the table with all positive days until maturity
    return tempTable.loc[tempTable["Days"] > 0]

processedTable = initialProcess(now, "sing_stock_data.csv")

##----------------------------------------------------------------------------------------------------------------------------------------------------------------##



##----------------------------------------------------------------------------------------------------------------------------------------------------------------##
## Classifiy the call bid, call ask, put bid and put ask from processed table
def getDates() -> list:
    
    # Get available months for current options
    r = requests.get("http://stock.finance.sina.com.cn/futures/api/openapi.php/StockOptionService.getStockName")
    textData = r.json()
    aSet = set(textData['result']['data']['contractMonth'])  # get a set of available options months
    monthList = list(aSet)
    processedList = list(map(lambda x: "".join(x.split('-'))[2:], [i for i in monthList]))  
    
    return processedList

def contractID(API, month):
    
    rr = requests.get(API + month)
    stepOne = rr.text.split('"')[1].split(',')
    stepTwo = [con for con in stepOne if con != '']
    
    return stepTwo

def getContractList(optionType):
    
    monthList = getDates()
    
    # Retrieve valid conract ID for call and put options
    callAPI = "http://hq.sinajs.cn/list=OP_UP_510050"
    putAPI = "http://hq.sinajs.cn/list=OP_DOWN_510050"
    
    if optionType == "call":
        API = callAPI
        theList = [contractID(API, month) for month in monthList]
        
    if optionType == "put":
        API = putAPI
        theList = [contractID(API, month) for month in monthList]

    return theList

callList = getContractList("call")
putList = getContractList("put")

# Get the sets of call and put options
def getSet(aList):
    emptySet = set()
    for con in aList:
        newCon = set(con)
        emptySet = emptySet | newCon
        
    return emptySet

callSet = getSet(callList)
putSet = getSet(putList)

# Select the row the meets the condition that the ID belongs to one specific set
callTable = processedTable.loc[processedTable["Contract ID"].isin(callSet)]
callTable.columns = ["Expiration Date", "Contract ID", "Strike", "Call Bid", "Call Ask", "Days", "Date"]

putTable = processedTable.loc[processedTable["Contract ID"].isin(putSet)]
putTable.columns = ["Expiration Date", "Contract ID", "Strike", "Put Bid", "Put Ask", "Days", "Date"]
##----------------------------------------------------------------------------------------------------------------------------------------------------------------##



##----------------------------------------------------------------------------------------------------------------------------------------------------------------##
## Starting from here, some modifications are made to be in line with that author's code for VIX index calculation
# Yields table
daysList = list(set(callTable["Days"]))
daysList.sort()
riskFree = 0.38

data = {'Date': [now] * len(daysList), 'Days': daysList, 'Rate':[riskFree] * len(daysList)}

yields = pd.DataFrame(data)

#----------------------------------------------------------------------------------------------------------------------------#
# Options tables (Cleaning and indexing)
# Get the average number for bid and ask according to different strike prices
tempCall = callTable.groupby(["Expiration Date", "Strike"], as_index = False).agg({"Call Bid": "mean", "Call Ask": "mean", "Days": "first", "Date": "first"})
tempPut = putTable.groupby(["Expiration Date", "Strike"], as_index = False).agg({"Put Bid": "mean", "Put Ask": "mean", "Days": "first", "Date": "first"})

calls = tempCall[['Strike', 'Call Bid', 'Call Ask', 'Days', 'Date']].copy()
puts = tempPut[['Strike', 'Put Bid', 'Put Ask', 'Days', 'Date']].copy()

# VIX is computed for the date of option quotations, we do not really need Expiration
calls = calls.set_index(['Date', 'Days', 'Strike'])
calls = calls[['Call Bid', 'Call Ask']].rename(columns = {'Call Bid': 'Bid', 'Call Ask': 'Ask'})
puts = puts.set_index(['Date', 'Days', 'Strike'])
puts = puts[['Put Bid', 'Put Ask']].rename(columns = {'Put Bid': 'Bid', 'Put Ask': 'Ask'})

# Add a column indicating the type of the option
calls['CP'], puts['CP'] = 'C', 'P'

# Merge calls and puts
options = pd.concat([calls, puts])

# Reindex and sort
options = options.reset_index().set_index(['Date','Days','CP','Strike']).sort_index()

#----------------------------------------------------------------------------------------------------------------------------#
# Compute bid/ask average (to filter out in-the-money options)
options['Premium'] = (options['Bid'] + options['Ask']) / 2
options2 = options[options['Bid'] > 0]['Premium'].unstack('CP')

#----------------------------------------------------------------------------------------------------------------------------#
# Find the absolute difference
options2['CPdiff'] = (options2['C'] - options2['P']).abs()
# Mark the minimum for each date/term
options2['min'] = options2['CPdiff'].groupby(level = ['Date','Days']).transform(lambda x: x == x.min())

#----------------------------------------------------------------------------------------------------------------------------#
# Compute forward price
# Leave only at-the-money optons
df = options2[options2['min'] == 1].reset_index()
# Merge with risk-free rate
df = pd.merge(df, yields, how = 'left')

# Compute the implied forward
df['Forward'] = df['CPdiff'] * np.exp(df['Rate'] * df['Days'] / 36500)
df['Forward'] += df['Strike']
forward = df.set_index(['Date','Days'])[['Forward']]

#----------------------------------------------------------------------------------------------------------------------------#
# Compute at-the-money strike
# Merge options with implied forward price
left = options2.reset_index().set_index(['Date','Days'])
df = pd.merge(left, forward, left_index = True, right_index = True)
# Compute at-the-money strike
mid_strike = df[df['Strike'] < df['Forward']]['Strike'].groupby(level = ['Date','Days']).max()
mid_strike = pd.DataFrame({'Mid Strike' : mid_strike})

# Go back to original data and reindex it
left = options.reset_index().set_index(['Date','Days']).drop('Premium', axis = 1)
# Merge with at-the-money strike
df = pd.merge(left, mid_strike, left_index = True, right_index = True)
# Separate out-of-the-money calls and puts
P = (df['Strike'] <= df['Mid Strike']) & (df['CP'] == 'P')
C = (df['Strike'] >= df['Mid Strike']) & (df['CP'] == 'C')
puts, calls = df[P], df[C]

#----------------------------------------------------------------------------------------------------------------------------#
# Remove all quotes after two consequtive zero bids
# Indicator of zero bid
calls['zero_bid'] = (calls['Bid'] == 0).astype(int)
# Accumulate number of zero bids starting at-the-money
calls['zero_bid_accum'] = calls.groupby(level = ['Date','Days'])['zero_bid'].cumsum()

# Sort puts in reverse order inside date/term
puts = puts.sort_values(['Strike'], ascending=False).groupby(level = ['Date', 'Days']).head()

# Indicator of zero bid
puts['zero_bid'] = (puts['Bid'] == 0).astype(int)
# Accumulate number of zero bids starting at-the-money
puts['zero_bid_accum'] = puts.groupby(level = ['Date','Days'])['zero_bid'].cumsum()

# Merge puts and cals
options3 = pd.concat([calls, puts]).reset_index()
# Throw away bad stuff
options3 = options3[(options3['zero_bid_accum'] < 2) & (options3['Bid'] > 0)]

# Compute option premium as bid/ask average
options3['Premium'] = (options3['Bid'] + options3['Ask']) / 2
options3 = options3.set_index(['Date','Days','CP','Strike'])['Premium'].unstack('CP')

#----------------------------------------------------------------------------------------------------------------------------#
# Compute out of the money option price
# Merge wth at-the-money strike price
left = options3.reset_index().set_index(['Date','Days'])
df = pd.merge(left, mid_strike, left_index = True, right_index = True)

# Conditions to separate out-of-the-money puts and calls
condition1 = df['Strike'] < df['Mid Strike']
condition2 = df['Strike'] > df['Mid Strike']
# At-the-money we have two quotes, so take the average
df['Premium'] = (df['P'] + df['C']) / 2
# Remove in-the-money options
df['Premium'].loc[condition1] = df['P'].loc[condition1]
df['Premium'].loc[condition2] = df['C'].loc[condition2]

options4 = df[['Strike','Mid Strike','Premium']].copy()

#----------------------------------------------------------------------------------------------------------------------------#
# Compute difference between adjoining strikes
def f(group):
    new = group.copy()
    new.iloc[1:-1] = np.array((group.iloc[2:] - group.iloc[:-2]) / 2)
    new.iloc[0] = group.iloc[1] - group.iloc[0]
    new.iloc[-1] = group.iloc[-1] - group.iloc[-2]
    return new

options4['dK'] = options4.groupby(level = ['Date','Days'])['Strike'].apply(f)

#----------------------------------------------------------------------------------------------------------------------------#
# compute contribution of each strike
# Merge with risk-free rate
yields = yields.set_index(['Date','Days'])

contrib = pd.merge(options4, yields, left_index = True, right_index = True).reset_index()

contrib['sigma2'] = contrib['dK'] / contrib['Strike'] ** 2
contrib['sigma2'] *= contrib['Premium'] * np.exp(contrib['Rate'] * contrib['Days'] / 36500)

#----------------------------------------------------------------------------------------------------------------------------#
# compute each period index
# Sum up contributions from all strikes
sigma2 = contrib.groupby(['Date','Days'])[['sigma2']].sum() * 2

# Merge at-the-money strike and implied forward
sigma2['Mid Strike'] = mid_strike
sigma2['Forward'] = forward

# Compute variance for each term
sigma2['sigma2'] -= (sigma2['Forward'] / sigma2['Mid Strike'] - 1) ** 2
sigma2['sigma2'] /= sigma2.index.get_level_values(1).astype(float) / 365
sigma2 = sigma2[['sigma2']]

#----------------------------------------------------------------------------------------------------------------------------#
# compute interpolated index
def f(group):
    days = np.array(group['Days'])
    sigma2 = np.array(group['sigma2'])
    
    if days.min() <= 30:
        T1 = days[days <= 30].max()
    else:
        T1 = days.min()
    
    T2 = days[days > T1].min()
    
    sigma_T1 = sigma2[days == T1][0]
    sigma_T2 = sigma2[days == T2][0]
    
    return pd.DataFrame([{'T1' : T1, 'T2' : T2, 'sigma2_T1' : sigma_T1, 'sigma2_T2' : sigma_T2}])
    
two_sigmas = sigma2.reset_index().groupby('Date').apply(f).groupby(level = 'Date').first()

#----------------------------------------------------------------------------------------------------------------------------#
# Interpolate VIX
df = two_sigmas.copy()

for t in ['T1','T2']:
    # Convert to fraction of the year
    df['days_' + t] = df[t].astype(float) / 365
    # Convert to miutes
    df[t] = (df[t] - 1) * 1440. + 510 + 930

df['sigma2_T1'] = df['sigma2_T1'] * df['days_T1'] * (df['T2'] - 30. * 1440.)
df['sigma2_T2'] = df['sigma2_T2'] * df['days_T2'] * (30. * 1440. - df['T1'])
df['VIX'] = ((df['sigma2_T1'] + df['sigma2_T2']) / (df['T2'] - df['T1']) * 365. / 30.) ** .5 * 100

VIX = df[['VIX']]

print(VIX)
