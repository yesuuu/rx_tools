import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model

from ruoxin_util import RxModeling, RxFundamental


class STWorkFlow():

    class Procedure():

        @staticmethod
        def changeX(x, changeArgs):
            funcDict = {
                'win': RxModeling.PdTools.winsorize,
                # 'winAll': STWorkFlow.Utils.changeXWrapper.winAll,
                'clip': lambda x, **kwargs: x.clip(**kwargs),
                'zscore': RxModeling.PdTools.getZscore,
                'finite': lambda x: x[np.isfinite(x)],
                'stack': lambda x: x.stack().values,
                'log': lambda x: np.log(x),

                'toSeasonal': RxFundamental.StatementFunc.toSeasonal,
                'toYearly': RxFundamental.StatementFunc.toYearly,

                'toDaily': STWorkFlow.Utils.changeXWrapper.rpDataToDailyDataWrapper,
                'toForecast': STWorkFlow.Utils.changeXWrapper.toForecastWrapper,
                'ffillDecay': RxModeling.PdTools.ffillDecay,
                'ffillDecayMulti': RxModeling.PdTools.ffillDecayMulti,

                'toPortFix': STWorkFlow.Utils.changeXWrapper.aToPortFix,
                'toPortFloat': STWorkFlow.Utils.changeXWrapper.aToPortFloat,

                'inSample': STWorkFlow.Utils.changeXWrapper.inSample,
                'toYHat': lambda x, normalizer: x * normalizer,

                'deIndusMean': STWorkFlow.Utils.changeXWrapper.deIndusMean,
                'deIndusStd': STWorkFlow.Utils.changeXWrapper.deIndusStd,

            }
            for change in changeArgs:
                assert len(change) == 2
                x = funcDict.get(change[0], change[0])(x, **change[1])
            return x

        @staticmethod
        def rpDistPrint(x, changeXDict=None, percentiles=(0.1, 1, 99, 99.9), info=None, save=None):
            if save:
                logObject = RxModeling.Log(file_name=save)
                logObject.start()

            if percentiles is not None:
                for p in percentiles:
                    print 'percentile', p, np.nanpercentile(x.stack(), p)

            summaryDict = {}
            summaryDict['x'] = RxModeling.X.calc_basic_statistics(x, info)
            if changeXDict:
                for xName, changeXArgs in changeXDict.iteritems():
                    summaryDict[xName] = RxModeling.X.calc_basic_statistics(
                        STWorkFlow.Procedure.changeX(x, changeXArgs), info)
            summaryDf = pd.DataFrame(summaryDict)
            print summaryDf

            if save:
                logObject.close()

            return summaryDict

        @staticmethod
        def xAdd(xList, xNames, y, singleEvaluateFunc, updateYBenchFunc, valueThresFunc,
                 sortValuesFunc=None, extraFuncs=None):

            singleValue = [singleEvaluateFunc(x, y) for x in xList]

            if sortValuesFunc:
                singleValue, xList, xNames = sortValuesFunc(singleValue, xList, xNames)

            summaryDfColumns = ['singleValue', 'marginValue', 'isAdd']
            if extraFuncs:
                summaryDfColumns.extend(sorted(extraFuncs.keys()))
            summaryDf = pd.DataFrame(index=xNames, columns=summaryDfColumns)
            summaryDf['singleValue'] = singleValue

            yBench, xNamesSelect, xListSelect = y, [], []
            for i, xName, x in zip(range(len(xNames)), xNames, xList):
                print '%d of %d, evaluating %s' % (i + 1, len(xNames), xName)
                marginValue = singleEvaluateFunc(x, yBench)
                if valueThresFunc(marginValue):
                    xNamesSelect.append(xName)
                    xListSelect.append(x)
                    yBenchNew = updateYBenchFunc(yBench, x)
                    isAdd = 1.
                else:
                    yBenchNew = yBench
                    isAdd = False
                summaryDf.loc[xName, 'marginValue'] = marginValue
                summaryDf.loc[xName, 'isAdd'] = isAdd

                stateDict = {
                    'xNamesSelect': xNamesSelect[:],
                    'xListSelect': xListSelect,
                    'yBench': yBench,
                    'yBenchNew': yBenchNew,
                    'y': y,
                    'isAdd': isAdd,
                }
                if extraFuncs:
                    # print 'isAdd', stateDict['isAdd']
                    for funcName in sorted(extraFuncs.keys()):
                        val = extraFuncs[funcName](stateDict)
                        # print 'val', val
                        summaryDf.loc[xName, funcName] = val

                yBench = yBenchNew

            return xNamesSelect, summaryDf

    class Utils():

        class Basic():

            @staticmethod
            def stack(xList, xNames, dropNa=None):
                df = pd.Panel({n: v for n, v in zip(xNames, xList)}).to_frame()
                if dropNa:
                    df = df.dropna(how=dropNa)
                return df

            @staticmethod
            def XmultiPlot(x, figNames=None, figConfig=None, figsize=None, columnNum=1, saveFig=None):
                figure = plt.figure(figsize=figsize)
                for i, fig in enumerate(figNames):
                    print 'plotting %s' % (fig,)
                    figDict = figConfig.get(fig, {})
                    ax = figure.add_subplot((len(figNames) + columnNum - 1) / columnNum, columnNum, i + 1)
                    xTmp = STWorkFlow.Procedure.changeX(x, figDict.get('changeX', []))
                    plotFunc = figDict['func'](ax)
                    plotKwargs = figDict.get('args', {})
                    plotFunc(xTmp, **plotKwargs)
                    ax.set_title(fig)
                if saveFig:
                    plt.savefig(saveFig)

            @staticmethod
            def spreadChange(xList, xNames, changeXs, changeNames):
                newXList, newXNames = [], []
                for xN, x in zip(xNames, xList):
                    newXList.extend([STWorkFlow.Procedure.changeX(x, cX) for cX in changeXs])
                    newXNames.extend([xN + cN for cN in changeNames])
                return newXList, newXNames

        class SelectUtils():

            @staticmethod
            def getIRWrapper(fe, portVarType=2, portDeMean=True):
                def getIR(x, y):
                    assert x.shape == y.shape
                    port = RxFundamental.AlphaFunc.convertPortfolioFloat(fe, x, scale=100, varType=portVarType,
                                                                         demean=portDeMean)
                    pnl = port.mul(y).shift(1).sum(axis=1)
                    ir = pnl.mean() / pnl.std() * np.sqrt(255)
                    return ir

                return getIR

            @staticmethod
            def getUpdateYBenchRegression(weight=None):
                def updateYBenchRegression(y, x):
                    df = STWorkFlow.Utils.Basic.stack([y, x], ['y', 'x'], dropNa='any')
                    weightArr = weight.stack().loc[df.index].values if weight is not None else None
                    lr = linear_model.LinearRegression(fit_intercept=True)
                    lr.fit(df['x'].values.reshape(-1, 1), df['y'].values, sample_weight=weightArr)
                    yNew = pd.Series(df['y'].values - lr.predict(df['x'].values.reshape(-1, 1)),
                                     index=df.index).unstack().loc[y.index, y.columns]
                    return yNew

                return updateYBenchRegression

            @staticmethod
            def xListAve():
                def ave(xList, y):
                    if len(xList) == 1:
                        return xList[0]
                    else:
                        return pd.Panel({str(i): x for i, x in enumerate(xList)}).mean(axis=0)
                return ave

            @staticmethod
            def xListRegCom(weight=None):
                def regCom(xList, y):
                    df = STWorkFlow.Utils.Basic.stack(xList + [y],
                                                      xNames=['x' + str(i + 1) for i in range(len(xList))] + ['y'],
                                                      dropNa='any')
                    weightArr = weight.stack().loc[df.index].values if weight is not None else None
                    lr = linear_model.LinearRegression(fit_intercept=True)
                    lr.fit(df.drop('y', 1).values, df['y'].values, sample_weight=weightArr)
                    yHatArray = lr.predict(df.drop('y', 1).values)
                    return pd.Series(yHatArray, index=df.index).unstack().loc[y.index, y.columns]

                return regCom

            @staticmethod
            def yHatICMean():
                def icMean(yHat, y):
                    return yHat.corrwith(y, axis=1).mean()
                return icMean

            @staticmethod
            def yHatICIR():
                def icir(yHat, y):
                    ic = yHat.corrwith(y, axis=1)
                    return ic.mean() / ic.std() * np.sqrt(255)
                return icir

            @staticmethod
            def yHatPortIR(fe, portVarType=2, portDeMean=True):
                def portIR(yHat, y):
                    port = RxFundamental.AlphaFunc.convertPortfolioFloat(fe, x, scale=100, varType=portVarType,
                                                                         demean=portDeMean)

                    pnl = port.mul(y).shift(1).sum(axis=1)
                    return pnl.mean() / pnl.std() * np.sqrt(255)
                return portIR

            @staticmethod
            def getStatWrapper(yHatFunc, yHatArgs, statFunc, statArgs):
                def getStat(xList, y):
                    yHat = yHatFunc(**yHatArgs)(xList, y)
                    stat = statFunc(**statArgs)(yHat, y)
                    return stat

                return getStat

            @staticmethod
            def LR(xListIn, yIn, xListOut, weightIn=None):
                dfIn = STWorkFlow.Utils.Basic.stack(xListIn + [yIn],
                                                  xNames=['x' + str(i + 1) for i in range(len(xListIn))] + ['y'],
                                                  dropNa='any')

                weightArr = weightIn.stack().loc[dfIn.index].values if weightIn is not None else None
                lr = linear_model.LinearRegression(fit_intercept=True)
                lr.fit(dfIn.drop('y', 1).values, dfIn['y'].values, sample_weight=weightArr)

                dfOut = STWorkFlow.Utils.Basic.stack(xListOut,
                                                  xNames=['x' + str(i + 1) for i in range(len(xListIn))],
                                                  dropNa='any')

                yHatArray = lr.predict(dfOut.values)
                return pd.Series(yHatArray, index=dfOut.index).unstack().loc[xListOut[0].index,
                                                                             xListOut[0].columns]

        class XPlotWrapper():

            @staticmethod
            def distributionWrapper(ax):
                def hist(x, **kwargs):
                    ax.hist(x, **kwargs)
                    ax.set_title('distribution')

                return hist

            @staticmethod
            def TSPlotWrapper(ax):
                def TSPlot(x, qOrStd='q', isMean=True, quantiles=(20, 40, 60, 80), stds=(-3, 1, 1, 3), **kwargs):
                    assert qOrStd in ('q', 'quantile', 's', 'std',)
                    plots = []
                    if isMean:
                        plots.append(('mean', x.mean(axis=1)))

                    if qOrStd in ('q', 'quantile'):
                        quantileData = np.nanpercentile(x, quantiles, axis=1)
                        for i, q in enumerate(quantiles):
                            plots.append(('q' + str(int(q)), quantileData[i]))
                    else:
                        std = x.std(axis=1)
                        for i, s in enumerate(stds):
                            plots.append(('std' + str(s), x.mean(axis=1) + s * std))

                    plotsDf = pd.DataFrame([np.array(p[1]) for p in plots]).T
                    plotsDf.columns = [p[0] for p in plots]
                    plotsDf.index = x.index
                    plotsDf.plot(ax=ax, **kwargs)
                    # raise Exception
                    # for plotTuple in plots:
                    #     ax.plot(np.array(plotTuple[1]), label=plotTuple[0], **kwargs)
                    #     ax.set_xticks(range(len(x))[::xTickSampleGap])
                    #     ax.set_xticklabels(x.index[::xTickSampleGap])
                    ax.legend(loc='best')
                    ax.set_title('ts')

                return TSPlot

            @staticmethod
            def countPlotWrapper(ax):
                def countPlot(x, **kwargs):
                    count = x.notnull().sum(axis=1)
                    count.plot(ax=ax, **kwargs)
                    ax.set_title('count')

                return countPlot

            @staticmethod
            def seasonalWrapper(ax):

                def seasonalPlot(x, normed=True, **kwargs):
                    data = x.sub(x.min(axis=0), axis=1).div(x.max(axis=0) - x.min(axis=0), axis=1) if normed else x
                    data.index = [i[-4:] for i in data.index]
                    labels = sorted(list(set(data.index)))
                    stackList = [data.loc[label].stack().values for label in labels]
                    ax.boxplot(stackList, labels=labels, **kwargs)
                    ax.set_title('seasonal')

                return seasonalPlot

        class XYPlotWrapper():

            @staticmethod
            def xyQuantileWrapper(ax):
                def plotQuantile(x, y, plotNum=20, isReg=True, isStd=False, isShowCorr=False, **plotKwargs):

                    x, y = np.array(x).ravel(), np.array(y).ravel()
                    valid = (~np.isnan(x)) & (~np.isnan(y))
                    x, y = x[valid], y[valid]
                    xArg = np.argsort(x)
                    x, y = x[xArg], y[xArg]
                    xMean = np.array(
                        [np.mean(x[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)]) for i in range(plotNum)])
                    yMean = np.array(
                        [np.mean(y[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)]) for i in range(plotNum)])
                    df = pd.DataFrame({'x': xMean, 'y': yMean})
                    df.plot.scatter('x', 'y', ax=ax, **plotKwargs)
                    ax.set_title('quantile plot')

                    if isStd:
                        yStd = np.array(
                            [np.std(y[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)], ddof=1) for i in
                             range(plotNum)])
                        ax.fill_between(xMean, yMean + yStd, yMean - yStd, alpha=0.3)

                    if isReg:
                        model = sm.OLS(yMean, sm.add_constant(xMean)).fit()
                        yHat = xMean * model.params[1] + model.params[0]
                        ax.plot(xMean, yHat)

                    if isShowCorr:
                        ax.text(0.01, 0.99, 'corr: %s' % (np.corrcoef(x, y)[0, 1],),
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=ax.transAxes, color='red', size=16)

                return plotQuantile

            @staticmethod
            def xyDecayWrapper(ax):
                def xyDecay(x, yFunc, numPort=5, backDays=30, forwardDays=60, **kwargs):
                    try:
                        perc = np.percentile(x.stack().values, [(i * 100) / numPort for i in range(1, numPort)])
                        xEvent = RxModeling.Basic.floatToEvent(x, perc)
                        yPanel = pd.Panel([yFunc(i).values for i in range(-backDays, forwardDays)],
                                          items=range(-backDays, forwardDays),
                                          major_axis=x.index, minor_axis=x.columns)
                        RxModeling.XY.plotEventDecay(*RxModeling.XY.getEventDecay(xEvent, yPanel, None, False),
                                                     ax=ax, **kwargs)
                    except:
                        return

                return xyDecay

            @staticmethod
            def xyFracWrapper(ax):
                def plotFractile(x, y, numPort=5, randSeed=0, **kwargs):
                    x, rtn = x.copy(), y.copy()
                    rtn[x.isnull()] = np.nan
                    x[rtn.isnull()] = np.nan

                    fracList = range(numPort, 0, -1)
                    fracRtn = pd.DataFrame(np.nan, x.index, fracList)
                    stkList = x.columns.tolist()
                    np.random.seed(randSeed)
                    for ds, r in rtn.iterrows():
                        if x.loc[ds].notnull().sum() > numPort:
                            np.random.shuffle(stkList)
                            grp = pd.qcut(x.loc[ds, stkList].rank(method="first"), numPort, labels=fracList)
                            fracRtn.loc[ds] = r.groupby(grp).mean()
                    fracRtn.sort_index(axis=1, inplace=True)

                    fracRtn.shift(1).groupby(level=0).sum().cumsum().plot(legend=True, ax=ax, **kwargs)
                    ax.set_title("frac")

                return plotFractile

            @staticmethod
            def xyPnlWrapper(ax):
                def plotLongShort(x, y, ):
                    pnl = x.mul(y).shift(1).sum(axis=1).cumsum()
                    longPnl = x[x > 0].mul(y).shift(1).sum(axis=1).cumsum()
                    shortPnl = x[x < 0].mul(y).shift(1).sum(axis=1).cumsum()
                    pnl.plot(label="Pnl", legend=True, ax=ax)
                    longPnl.plot(label="Long Pnl", legend=True, ax=ax)
                    shortPnl.plot(label="Short Pnl", legend=True, ax=ax)
                    ax.set_title("Pnl")

                return plotLongShort

        class changeXWrapper():

            @staticmethod
            def toForecastWrapper(x, index=None, columns=None):
                xF = x.copy()
                xF.columns = [('180' + i) if i.startswith('6') else ('190' + i) for i in x.columns]
                if columns is not None:
                    xF = xF.loc[:, columns]
                xF.index = pd.to_datetime(xF.index)
                if index is not None:
                    xF = xF.loc[index]
                return xF

            @staticmethod
            def rpDataToDailyDataWrapper(x, **kwargs):
                ee = kwargs.pop('ee')
                return ee.markEventsUsingDf(x, **kwargs)

            @staticmethod
            def aToPortFix(x, **kwargs):
                fe = kwargs.pop('fe')
                return RxFundamental.AlphaFunc.convertPortfolioFix(fe, x, **kwargs)

            @staticmethod
            def aToPortFloat(x, **kwargs):
                fe = kwargs.pop('fe')
                return RxFundamental.AlphaFunc.convertPortfolioFloat(fe, x, **kwargs)

            @staticmethod
            def inSample(x, startDate=None, endDate=None):
                startDate = x.index[0] if startDate is None else startDate
                endDate = x.index[-1] if endDate is None else endDate
                return x.loc[startDate:endDate]

            @staticmethod
            def deIndusMean(x, industry, minNum=10):
                funcs = {
                    'mean': lambda x: np.nanmean(x, axis=1),
                    'num': lambda x: x.notnull().sum(axis=1),
                }
                res = RxModeling.PdTools.calGroupInfo(x, industry, funcs)
                indusMean = res['mean'][res['num'] >= minNum]
                indusMeanExpand = RxModeling.PdTools.getValueByGroup(indusMean, industry)
                return x - indusMeanExpand

            @staticmethod
            def deIndusStd(x, industry, minNum=10):
                funcs = {
                    'mean': lambda x: np.nanmean(x, axis=1),
                    'std': lambda x: np.nanstd(x, axis=1),
                    'num': lambda x: x.notnull().sum(axis=1),
                }
                res = RxModeling.PdTools.calGroupInfo(x, industry, funcs)
                indusMean = res['mean'][res['num'] >= minNum]
                indusStd = res['std'][res['num'] >= minNum]
                indusMeanExpand = RxModeling.PdTools.getValueByGroup(indusMean, industry)
                indusStdExpand = RxModeling.PdTools.getValueByGroup(indusStd, industry)
                return (x - indusMeanExpand) / indusStdExpand

        class IndustryAnaWrapper():

            @staticmethod
            def indusBoxWrapper(ax):
                def indusBox(x, industry, **kwargs):
                    industries = sorted(set(industry.stack().values))
                    stackList = [x[industry == ind].stack().values for ind in industries]
                    ax.boxplot(stackList, labels=industries, **kwargs)
                    ax.set_title('seasonal')

                return indusBox

            @staticmethod
            def indusMeanWrapper(ax):
                def indusMean(x, industry, minNum=10, **kwargs):
                    funcs = {
                        'mean': lambda x: np.nanmean(x, axis=1),
                        'num': lambda x: x.notnull().sum(axis=1),
                    }
                    res = RxModeling.PdTools.calGroupInfo(x, industry, funcs)
                    indusMean = res['mean'][res['num'] >= minNum]
                    indusMean.plot(ax=ax, **kwargs)
                    # raise Exception

                return indusMean

            @staticmethod
            def indusStdWrapper(ax):
                def indusStd(x, industry, minNum=10, **kwargs):
                    funcs = {
                        'std': lambda x: np.nanstd(x, axis=1),
                        'num': lambda x: x.notnull().sum(axis=1),
                    }
                    res = RxModeling.PdTools.calGroupInfo(x, industry, funcs)
                    indusMean = res['std'][res['num'] >= minNum]
                    indusMean.plot(ax=ax, **kwargs)
                    # raise Exception

                return indusStd
