import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class RxStatics(object):

    class XY(object):

        @staticmethod
        def plotQuantile(x, y, plotNum=100, isReg=True, **plotKwargs):
            valid = (~np.isnan(x)) & (~np.isnan(y))
            x, y = x[valid], y[valid]
            xArg = np.argsort(x)
            x, y = x[xArg], y[xArg]
            xMean = np.array([np.mean(x[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)]) for i in range(plotNum)])
            yMean = np.array([np.mean(y[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)]) for i in range(plotNum)])

            pd.DataFrame({'alpha': xMean, 'return': yMean}).plot.scatter('alpha', 'return', **plotKwargs)
            plt.title('qq plot')
            if isReg:
                model = sm.OLS(yMean, sm.add_constant(xMean)).fit()
                yHat = xMean * model.params[1] + model.params[0]
                plt.plot(xMean, yHat)
            plt.show()
