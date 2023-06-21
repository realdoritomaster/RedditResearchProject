import json
import time
from functools import reduce

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from calendar import monthrange
import numpy as np

from scipy import interpolate
from scipy import ndimage

import yfinance as yf
import pendulum

def readJson(url):
    f = open(url, "r")
    data = json.load(f)

    # print(json.dumps(data))

    f.close()
    return data

class RedditVisualisation:
    def __init__(self, term):
        self.term = term

    def getData(self, interval, term):
        data = readJson("data1.0.json")
        filtered = []

        subredditVolumeTimeline = []
        volume = []
        timeline = 0

        dataChunks = data[0]["data"]
        dataList = data[0]["data"][0]
        # print(json.dumps(dataList))
        # print("length of data: " + str(range(len(dataList))))
        print("Collecting volume of " + term)
        for j in range(len(dataChunks)):
            dataList = data[0]["data"][j]

            for i in range(len(dataList)):
                # author = json.dumps(dataList[i]["author"])
                body = json.dumps(dataList[i]["body"])
                # subreddit = json.dumps(dataList[i]["subreddit"])
                # score = json.dumps(dataList[i]["score"])
                # awardsRecieved = json.dumps(dataList[i]["total_awards_received"])
                # date = json.dumps(dataList[i]["utc_datetime_str"])
                utc = json.dumps(dataList[i]["created_utc"])
                #
                utc_converted = pd.to_datetime(utc, unit='s')
                year = utc_converted.strftime('%Y')
                month = utc_converted.strftime('%m')
                day =  utc_converted.strftime('%d')
                # print(term)
                if body.lower().find(term.lower()) != -1:
                    if interval == "day":
                        volume.append({"date": {"year": year, "month":month, "day":day}})
                    elif interval == "month":
                        volume.append({"date": {"year": year, "month":month}})
                # print()
                # print("#: " + str(i + (j*250)))
                # print("Body: " + body)
                # print("Subreddit: " + subreddit)
                # print("Score: " + score)
                # print("Awards Recieved: " + awardsRecieved)
                # print("Date: " + date)
                # print("Year: " + year)
                # print("Month: " + month)
        # print(googleVolume)
        # print(googleVolume[0]["date"])

        min_year = int(volume[len(volume)-1]["date"]["year"])
        max_year = int(volume[0]["date"]["year"])
        if interval == "day":
            timeline = [{"year": min_year + i, "month": j+1, "day": l+1, "count": 0} for i in range(max_year - min_year) for j in range(12) for l in range(0, monthrange(min_year + i, j+1)[1])]
        elif interval =="month":
            timeline = [{"year": min_year + i, "month": j+1, "count": 0} for i in range(max_year - min_year) for j in range(12) ]

        return {"x": np.array(timeline), "y": np.array(volume)}

    def combineData(self, data):
        newDict = {"x": [i for i in data[0]["x"]], "y": []}
        # print([i for i in data[0]["y"]])
        # print(data[0]["y"])
        for i in data:
            # print(range(len(newDict["y"])))
            for j in range(len(i["y"])):
                # print(i["y"][j])
                newDict['y'].append(i["y"][j])
        # print(newDict)
        # print(np.array(newDict))
        return {"x": np.array(newDict["x"]), "y": np.array(newDict["y"])}

    def filterByMonth(self, interval):
        terms = self.term

        # terms = terms[0]
        # df = self.getData(interval, terms)
        d = []
        for i in terms:
            d.append(self.getData(interval, i))

        df = self.combineData(d)
        # print(df)

        timeline = df["x"]
        volume = df["y"]

        totalVolume = 0
        for i in range(len(volume)):
            for j in range(len(timeline)):
                if (int(volume[i]["date"]["year"]) == timeline[j]["year"]) and (int(volume[i]["date"]["month"]) == timeline[j]["month"]):
                    timeline[j]["count"] = timeline[j]["count"] + 1
                    totalVolume += 1

        filtered_timeline = np.array([i for i in timeline if not (i['count'] == 0)])
        # filtered_timeline = timeline

        filtered_timeline = np.delete(filtered_timeline, 0) # Very Important as even though there is data for the first month, data wasn't collected for the whole month, therefore it is misleading and must be removed
        # print(filtered_timeline)
        # print(totalVolume)

        return pd.DataFrame({'month': np.array([datetime.strptime(str(filtered_timeline[i]["month"]) +"/"+ str(filtered_timeline[i]["year"]), '%m/%Y') for i in range(len(filtered_timeline))]),
                                  'volume': np.array([filtered_timeline[i]["count"] for i in range(len(filtered_timeline))])})

    def filterByDay(self, interval):
        df = self.getData(interval)
        timeline = df["x"]
        volume = df["y"]

        for i in range(len(volume)):
            for j in range(len(timeline)):
                if (int(volume[i]["date"]["year"]) == timeline[j]["year"]) and (int(volume[i]["date"]["month"]) == timeline[j]["month"]) and (int(volume[i]["date"]["day"]) == timeline[j]["day"]):
                    timeline[j]["count"] = timeline[j]["count"] + 1

        # filtered_timeline = np.array([i for i in timeline if not (i['count'] == 0)])
        filtered_timeline = timeline
        print(filtered_timeline)

        return pd.DataFrame({'month': np.array([datetime.strptime(str(filtered_timeline[i]["month"]) +"/"+ str(filtered_timeline[i]["day"]) + "/" + str(filtered_timeline[i]["year"]), '%m/%d/%Y') for i in range(len(filtered_timeline))]),
                                  'volume': np.array([filtered_timeline[i]["count"] for i in range(len(filtered_timeline))])})

    def volumeScatter(self, filter):
        dataframe = filter
        # print("dataframe: " + dataframe)
        x = dataframe.month
        y = dataframe.volume
        x_ints = np.array([i for i in range(len(x))])
        # print(x_ints)

        # Plotting the time series of given dataframe
        fig, ax = plt.subplots()
        ax.scatter(x, y, marker='o')

        s = np.polyfit(x_ints, y, 1)
        hy = s[0]*x_ints+s[1]
        r = np.corrcoef(x_ints, y)[0,1]
        ax.plot(x, hy, color="red")
        ax.text(0.75,0.9,'Slope: {:.3f}'.format(s[0]), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, weight = 'bold')
        ax.text(0.75,0.8,'R-squared: {:.3f}'.format(r), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, weight = 'bold')
        # Giving title to the chart using plt.title
        plt.title(f'{self.term} Volume by Date')

        # rotating the x-axis tick labels at 30degree
        # towards right
        plt.xticks(rotation=30, ha='right')

        # Providing x and y label to the chart
        plt.xlabel('Date')
        plt.ylabel('Volume')
        # plt.show()

    def volumeOverTime(self):
        dataframe = self.filterByMonth()

        # Plotting the time series of given dataframe
        plt.style.use('dark_background')
        plt.plot(dataframe.month, dataframe.volume)
        #
        # Giving title to the chart using plt.title
        plt.title(f'{self.term} Volume by Date')

        # rotating the x-axis tick labels at 30degree
        # towards right
        plt.xticks(rotation=30, ha='right')

        # Providing x and y label to the chart
        plt.xlabel('Date')
        plt.ylabel('Volume')
        # plt.show()

class StockVisualisation:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end

    def getData(self, type):
        # stock = yf.Ticker(self.ticker)
        price_history = yf.download(tickers=self.ticker, start=self.start, end=self.end, # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                                       interval='1mo') # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # print(price_history['Close'][0])

        price_series = []
        x = []

        if (type == "open"):
            price_series = price_history['Open']
            # print([dt for dt in list(price_history.index)])
            x = [dt for dt in list(price_history.index)]
        elif (type == "change"):
            openPrice = price_history['Open']
            closePrice = price_history['Close']
            l = len(openPrice)

            price_series = [abs(openPrice[i] - closePrice[i]) for i in range(l)]
            print(price_series)
            x = [dt for dt in list(price_history.index)]
        elif (type == "Percent Change"):
            openPrice = price_history['Open']
            closePrice = price_history['Close']
            l = len(openPrice)

            price_series = [abs((closePrice[i] - openPrice[i])/openPrice[i])*100 for i in range(l)]
            print(price_series)
            x = [dt for dt in list(price_history.index)]
            # print(x)
        # maxP = max(price_series)
        # minP = min(price_series)
        # normalize = lambda i : ((i-minP)/(maxP-minP))
        # print(len(price_series))
        return pd.DataFrame({'x': x, 'y': [i for i in price_series]})

    def yfinanceTimeline(self, type):
        df = self.getData(type)

        x_ints = [i for i in range(len(df.x))]

        x_smooth = df.x
        y_smooth = df.y
        # spl = interpolate.UnivariateSpline(x_int, price_series, k=3)
        sigma = 3
        x_g1d = ndimage.gaussian_filter1d(x_ints, sigma)
        y_g1d = ndimage.gaussian_filter1d(y_smooth, sigma)


        # plt.style.use('dark_background')
        fig, ax = plt.subplots()
        ax.plot(df.x, df.y, linewidth=2, label="Original Data")
        ax.plot(df.x, y_g1d,'red', linestyle="dashed", linewidth=1, label="Gaussian Smoothing, sigma=" + str(sigma))

        plt.title(self.ticker + ' stock')
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

        plt.legend()
        plt.grid()

        plt.xlabel('Date')
        plt.ylabel(type + ' Price ($)')
        # plt.show()

class DataVis:
    def __init__(self, ticker, term):
        self.ticker = ticker
        self.term = term
        self.file_root = "Results/"

        self.redditVis = RedditVisualisation(term)
        self.stockVis = StockVisualisation(ticker, start="2020-5-1", end="2022-12-31")

    def changeFileRoot(self, root):
        self.file_root = root

    def changeTicker(self, ticker):
        self.ticker = ticker
        self.stockVis.ticker = ticker

    def changeTerm(self, term):
        self.term = term
        self.redditVis.term = term

    def volumeHistory(self, filter):
        if filter == "month":
            filter = self.redditVis.filterByMonth("month")
        elif filter == "day":
            self.redditVis.filterByDay("day")

        print(filter)
        self.redditVis.volumeScatter(filter)

    def stockHistory(self, type):
        self.stockVis.yfinanceTimeline(type)

    def compare(self, stock_price_type):
        df1 = self.redditVis.filterByMonth("month")
        x1 = df1.month
        y1 = df1.volume

        df2 = self.stockVis.getData(stock_price_type)
        x2 = df2.x
        y2 = df2.y

        # x1_ints = np.array([i for i in range(len(x1))])

        # Plotting the time series of given dataframe
        # fitY = y2[::y1]

        fig, ax = plt.subplots()
        ax.scatter(y1, y2, marker='o')

        x_axis = y1

        s = np.polyfit(x_axis, y2, 1)
        hy = s[0]*x_axis+s[1]
        r = np.corrcoef(x_axis, y2)[0,1]

        ax.plot(x_axis, hy, color="red")

        ax.text(0.75,0.9,'Slope: {:.3f}'.format(s[0]), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, weight = 'bold')
        ax.text(0.75,0.8,'R-value: {:.3f}'.format(r), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes, weight = 'bold')
        # Giving title to the chart using plt.title
        plt.title(f'{self.term} Volume vs {self.ticker} Price')

        # rotating the x-axis tick labels at 30degree
        # towards right
        plt.xticks(rotation=30, ha='right')

        # Providing x and y label to the chart
        plt.xlabel('Volume')
        plt.ylabel(stock_price_type + ' Price')

        # plt.show()

    def showPlot(self):
        plt.show()

    def savePlot(self, file_type=".png", suffix=""):
        plt.gcf().set_size_inches(10.5, 8.5)
        # plt.figure().set_figheight(30)
        plt.savefig(f'{self.file_root}{self.term}_V_{self.ticker}{suffix}{file_type}')
        plt.close()

ticker = 'TSLA'
term = ['Tesla']
# term = ['Tesla', 'Microsoft', 'Google', 'Apple']

vis = DataVis(ticker, term)

defaultData=[
{"ticker": 'TSLA', "term": ['Tesla']},
{"ticker": 'GOOGL', "term": ['Google']},
{"ticker": 'AAPL', "term": ['Apple']},
{"ticker": 'MSFT', "term": ['Microsoft']},
{"ticker": 'NDAQ', "term": ['Nasdaq']},
]

other=[
# {"ticker": 'CORN', "term": ['Tesla', 'Google', 'Apple', 'Microsoft', 'Nasdaq']}
{"ticker": 'SPY', "term": ['Nasdaq']},
{"ticker": 'SPY', "term": ['Apple']}
]

def comparePlots(data):
    for i in data:
        vis.changeTicker(i["ticker"])
        vis.changeTerm(i["term"])
        # vis.compare(stock_price_type="open") # change, open
        # vis.savePlot(suffix="-priceOpen")
        # vis.compare(stock_price_type="change")
        # vis.savePlot(suffix="-priceChange")
        vis.compare(stock_price_type="Percent Change")
        vis.savePlot(suffix="-pricePercentChange")

def stockHistoryPlots(data):
    for i in data:
        vis.changeTicker(i["ticker"])
        vis.changeTerm(i["term"])
        vis.stockHistory("open")
        vis.savePlot(suffix="-priceOpen")
        vis.stockHistory("change")
        vis.savePlot(suffix="-priceChange")

def volumeHistoryPlots(data):
    for i in data:
        vis.changeTicker(i["ticker"])
        vis.changeTerm(i["term"])
        vis.volumeHistory("month")
        vis.savePlot()

# Create all default plots and data
vis.changeFileRoot("Results/Compare/")
comparePlots(defaultData)

# vis.changeFileRoot("Results/StockHistory/")
# stockHistoryPlots(defaultData)
#
# vis.changeFileRoot("Results/VolumeHistory/")
# volumeHistoryPlots(defaultData)

# Create other plots
# vis.changeFileRoot("Results/Other/")
# comparePlots(other)
# # stockHistoryPlots(other)
# volumeHistoryPlots(other)

# vis.changeTicker("MSFT")
# vis.changeTerm(['Microsoft'])
# vis.compare(stock_price_type="Percent Change")
# vis.showPlot()
# vis.savePlot(suffix="-pricePercentChange")

# vis.volumeHistory("month")
# vis.stockHistory("change") # change, open
