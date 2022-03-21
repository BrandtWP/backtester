import pyomo.environ as pyo
from sys import stdout
import math
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
import csv
import numpy as np

logging.getLogger('pyomo').setLevel(logging.CRITICAL)


class backtester:
    def __init__(self, priceFrame, debug = False, dateFormat=None):
        self.prices = priceFrame
        if self.prices.index.dtype != np.dtype('<M8[ns]'):
            try:
                self.prices.index = pd.to_datetime(self.prices.index, format=None)
            except:
                raise TypeError('The price dataframe index cannot be converted to datetime. Set the datetime format using the optional "dateFormat" argument')
        self.returns = (self.prices / self.prices.shift(1)).fillna(1)
        
        self.debug = debug
        
    
    def backtest(self,
                 objective,
                 signalFrame=None,
                 startDate=None,
                 cash=100,
                 tradingFee = 0, 
                 maxLeverage = 1, 
                 maximumSingleAllocation = 1,
                 minimumSingleAllocation = 0,
                 optimizer = 'mindtpy', 
                 solverArgs = dict(), 
                 returnPositions = False,
                 log = False,
                 logDir = 'logs',
                 backTestName='backtest',
                 rebalancePeriod = 1,
                 generateCovariances=None,
                 benchmark = None,
                 **kwargs):
        
        if generateCovariances == None:
            generateCovariances = self.sampleCovariance
        
        if startDate is None:
            startDate = self.prices.index[0]
        
        if signalFrame is None:
            signalFrame = pd.DataFrame(0, index=self.prices.index, columns=self.prices.columns)
        
        if not set(signalFrame.columns).issubset(set(self.prices.columns)):
            raise Exception("signalFrame contains symbols not in priceFrame. Namely: %s" % (set(signalFrame.columns) - set(self.prices.columns)))
            
        signalFrame = signalFrame.sort_index(ascending=True)
        prices = self.prices[signalFrame.columns].sort_index(ascending=True)
        returns = self.returns[signalFrame.columns].sort_index(ascending=True)
        
        if prices.shape != signalFrame.shape:
            raise Exception("Incompatible signal frame size. Expecting %s, received %s" % (prices.shape, signalFrame.shape))
            
        if self.debug: print("Creating portfolio model")
        portfolioModel = self.createPortfolioModel(signalFrame.columns, maxLeverage, objective, minimumSingleAllocation, maximumSingleAllocation, cash)
        if self.debug: print("Model created")
        
        opt = None
        if 'executable' in kwargs.keys():
            opt = pyo.SolverFactory(optimizer, executable=kwargs['executable'])
        else:
            opt = pyo.SolverFactory(optimizer)
        
        positions = [[0]* len(signalFrame.columns)]
        portfolioValues = []
        
        rebalanceDates = signalFrame.loc[startDate:].iloc[0::rebalancePeriod].index
        measurementDates = prices.loc[startDate:].index
        
        loopLength = len(rebalanceDates)
        toolbar_width = 100
        
        logWriter = None
        logFile = None
        
        if log:
            if not os.path.exists(logDir):
                os.mkdir(logDir)
                
            logFile = open(logDir + '/' + backTestName + '.log', 'w', encoding='utf-8')
            logWriter = csv.writer(logFile)
            headerRow = ["date"]
            headerRow.extend(signalFrame.columns.tolist())
            headerRow.append("cash")
            
            logWriter.writerow(headerRow)
        
        if self.debug: print("Starting Iteration")
            
        j = 0
        
        for date in measurementDates:
            
            priceRow = prices.loc[date]
            signalRow = signalFrame.loc[date]
                        
            cash = sum([positions[-1][i] * priceRow[i] for i in range(0, len(priceRow)-1)]) + cash
            
            portfolioValues.append(cash)
            
            if date in rebalanceDates:
            
                progress = math.floor((j + 1) / loopLength * toolbar_width)
                stdout.write("\rPortfolio value: $%i   Progress: [%s%s] %s%% %s\t" % (cash, "-"*progress, " "*(toolbar_width-progress), int(100 * (j+1) / loopLength), date))
                stdout.flush()

                instance = self.optimizePortfolio(portfolioModel, opt, priceRow, signalRow, generateCovariances(date, rebalancePeriod), cash, solverArgs)

                positions.append([v.value for v in instance.shares._data.values()])
                
                j = j+1
            
            if log:
                logRow = [date]
                logRow.extend(positions[-1])
                logRow.append(cash)
                
                logWriter.writerow(logRow)
                logFile.flush()
            
            cash = cash - sum([positions[-1][i] * priceRow[i] for i in range(0, len(priceRow)-1)])
            
        self.printSummary(pd.Series(portfolioValues, index=measurementDates), benchmark.loc[measurementDates], frequency=1)
            
        
    def createPortfolioModel(self, assets, maxLeverage, portfolioEvaluator, minimumSingleAllocation, maximumSingleAllocation, cash):
          
        model = pyo.AbstractModel()
        
        model.assets = pyo.Set(initialize=assets.to_list())
        
        model.covariances = pyo.Param(model.assets, model.assets, initialize=0)
        model.prices = pyo.Param(model.assets, domain=pyo.PositiveReals, initialize = 1)
        model.signals = pyo.Param(model.assets, domain=pyo.Reals)
        model.cash = pyo.Param(initialize=cash)

        def singleAllocationRule(model, i):
            maximumNumberOfShares = pyo.value(model.cash) / pyo.value(model.prices[i])
            return (minimumSingleAllocation * maximumNumberOfShares, maximumSingleAllocation * maximumNumberOfShares)
        
        model.shares = pyo.Var(model.assets, domain=pyo.Reals, initialize=0, bounds=singleAllocationRule)
        
        
        if minimumSingleAllocation >= 0:
            
            def limitLongPosition(model):
                return sum(model.shares[i]*model.prices[i] for i in model.assets) == maxLeverage * model.cash
            
            model.maxLeverageConstraint = pyo.Constraint(rule=limitLongPosition)
        else:
            model.t = pyo.Var(model.assets, domain=pyo.NonNegativeReals, initialize=0)
            
            def limitLongShortPosition(model):
                return sum(model.t[i] for i in model.assets) == maxLeverage * model.cash
            
            def positiveMinimum(model, i):
                return model.t[i] >= model.shares[i] * model.prices[i]
            
            def negativeMinimum(model, i):
                return model.t[i] >= -model.shares[i] * model.prices[i]
            
            
            model.maxLeverageConstraint = pyo.Constraint(rule=limitLongShortPosition)
            model.auxillaryAbsoluteValueConstraint1 = pyo.Constraint(model.assets, rule=positiveMinimum)
            model.auxillaryAbsoluteValueConstraint2 = pyo.Constraint(model.assets, rule=negativeMinimum)

        
        
        
        model.obj = pyo.Objective(rule = portfolioEvaluator, sense=pyo.minimize)
        
        return model
    
    def sampleCovariance(self, date, rebalancePeriod):
        return np.log(self.returns.loc[:date]).cov() * (rebalancePeriod ** 0.5)
        
    
    def optimizePortfolio(self, model, opt, prices, signals, covariances, cash, solverArgs=dict()):
        
        data = { None: {
            'prices': prices.to_dict(),
            'signals': signals.to_dict(),
            'covariances': self.to_pyomo_dict(covariances),
            'cash': {None: cash}
        }
        }
        
        if self.debug: print("Building Instance")
        instance = model.create_instance(data)

#         bestValueAsset = (signals / prices).nlargest(1).index[0]
#         instance.shares[bestValueAsset] = cash / prices[bestValueAsset]
        
        if self.debug: print("Solving model instance")
        opt.solve(instance, **solverArgs)
        return instance
    
    def printSummary(self, portfolio, benchmark, frequency=1):
        print('\n')
        print("Total Return")
        print("\t From your porfolio: ", "{:.2%}".format(portfolio[-1] / portfolio[0] - 1))
        print("\t From the benchmark: ", "{:.2%}".format(benchmark[-1] / benchmark[0] - 1))
        print("\n")
        print("Average Annualized Return")
        print("\t From your porfolio: ", "{:.2%}".format(metrics.meanAnnualReturn(portfolio, frequency=frequency)))
        print("\t From the benchmark: ", "{:.2%}".format(metrics.meanAnnualReturn(benchmark, frequency=frequency)))
        print("\n")
        print("Maximum Drawdown")
        print("\t From your porfolio: ", "{:.2%}".format(metrics.maxDrawDown(portfolio)))
        print("\t From the benchmark: ", "{:.2%}".format(metrics.maxDrawDown(benchmark)))
        print("\n")
        print("Information Ratio: ", "{:.2f}".format(metrics.informationRatio(portfolio, benchmark, frequency=frequency)))
        print("Sortino Ratio: ", "{:.2f}".format(metrics.sortinoRatio(portfolio, benchmark, frequency=frequency)))
        
        pd.DataFrame({'Portfolio': portfolio / portfolio[0], 'Benchmark': benchmark / benchmark[0]}).plot()
    
    def to_pyomo_dict(self, dictionary):
        pyomo_dict = {}
        
        for key in dictionary.keys():
            pyomo_dict.update({(key, nested_key): value for (nested_key, value) in dictionary[key].items()})

        return pyomo_dict
    

class metrics:
    def meanAnnualReturn(series, frequency = 1):
        return (series[-1] / series[0]) ** ((252 / frequency) / len(series)) - 1
    
    def maxDrawDown(series):
        return np.min(series / series.expanding(min_periods=1).max() - 1)
    
    def informationRatio(series, benchmark, frequency=1):
        returns = series.pct_change()
        benchmarkReturns = benchmark.loc[series.index].pct_change()
        return (metrics.meanAnnualReturn(series, frequency=frequency) - metrics.meanAnnualReturn(benchmark, frequency=frequency)) / (np.sqrt(np.var(returns - benchmarkReturns)) * np.sqrt(252/frequency))
    
    def sortinoRatio(series, benchmark, frequency=1):
        returnDiff = series.pct_change() - benchmark.loc[series.index].pct_change()
        downsideRisk = np.sqrt(np.mean(np.square(returnDiff.loc[returnDiff < 0])) * 252 / frequency)
        
        return (metrics.meanAnnualReturn(series, frequency=frequency) - metrics.meanAnnualReturn(benchmark, frequency=frequency)) / downsideRisk
        
        
        
        
        
    