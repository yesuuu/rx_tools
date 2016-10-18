import numpy as np
import scipy.stats as stat
import statsmodels.api as sm
import matplotlib.pyplot as plt


def jb_test(series, level=0.05, is_plot=True, is_print=True):
    """
    output: (is_h0_true, p_value, jb_stat, critical value)
    """
    series = series[~np.isnan(series)]
    if len(series) < 100:
        print 'Warning(in JB test): data length: %d' % (len(series),)
    skew = stat.skew(series)
    kurt = stat.kurtosis(series)
    N = len(series)
    jb = (N - 1) * (skew ** 2 + kurt ** 2 / 4) / 6
    p_value = 1 - stat.chi2.cdf(jb, 2)
    cv = stat.chi2.ppf(1 - level, 2)
    is_h0_true = False if p_value < level else True
    if is_plot:
        from smart_plot import plot_distribution
        plot_distribution(series, )
    if is_print:
        print ''
        print '*******  JB TEST  *******'
        print 'skew: %.4f' % (skew, )
        print 'kurt: %.4f' % (kurt, )
        if is_h0_true:
            print 'h0 is True: data is normal'
        else:
            print 'h0 is False: data is not normal'
        print 'p value: %f' % (p_value, )
        print 'jb stat: %f' % (jb, )
        print 'critical value: %f' %(cv, )
    return is_h0_true, p_value, jb, cv


def box_test(series, lag=10, type='ljungbox',
             level=0.05, is_plot=True, is_print=True):
    """
    output: (is_h0_true, p_value, q_stat, critical value)
    """
    series = series[~np.isnan(series)]
    acf = sm.tsa.acf(series, nlags=lag)
    if is_plot:
        sm.graphics.tsa.plot_acf(series, lags=lag)
        plt.show()
    q_stat = sm.tsa.q_stat(acf[1:], len(series), type=type)[0][-1]
    p_value = stat.chi2.sf(q_stat, lag)
    cv = stat.chi2.ppf(1 - level, lag)
    is_h0_true = False if p_value < level else True
    if is_print:
        print ''
        print '*******  Ljung Box TEST  *******'
        if is_h0_true:
            print 'h0 is True: data is independent'
        else:
            print 'h0 is False: data is not independent'
        print 'p value: %f' % (p_value, )
        print 'q stat: %f' % (q_stat, )
        print 'critical value: %f' % (cv, )
    return is_h0_true, p_value, q_stat, cv


def pair_test(series1, series2, series1_name='series1', series2_name='series2',
              level=0.05, is_plot=True, is_print=True):
    assert len(series1) == len(series2)
    if len(series1) <= 100:
        print 'Warning: length of data is %d, smaller than 100' % (len(series1), )
    dif = np.array(series1) - np.array(series2)
    dif_cum = np.cumsum(dif)
    corr1 = np.corrcoef(series1, series2)[0, 1]
    t_value = np.float(np.mean(dif) / np.sqrt(np.var(dif) / len(dif)))
    p_value = 2 * (1 - stat.t.cdf(np.abs(t_value), len(dif)))
    if is_plot:
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('Pair Test')
        ax = fig.add_subplot(211)
        plt.plot(np.cumsum(series1), 'b')
        plt.plot(np.cumsum(series2), 'g')
        plt.title('Cum Return')
        plt.legend([series1_name, series2_name], loc='best')
        ax.text(0.01, 0.99, 'data length: %d' % (len(series1)),
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, color='red', size=16)
        ax = fig.add_subplot(212)
        plt.plot(dif_cum)
        plt.title('Diff Cum Return')
        ax.text(0.01, 0.99, 't_value: %0.4f\np_value: %0.4f\ncorr: %0.4f' % (t_value, p_value, corr1),
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, color='red', size=16)
        plt.show()
    cv = stat.norm.ppf(1 - level / 2)
    is_h0_true = False if p_value < level else True
    if is_print:
        print ''
        print '*******  Pair T TEST  *******'
        if is_h0_true:
            print 'h0 is True: diff is significant'
        else:
            print 'h0 is False: diff is not significant'
        print 'p value: %f' % (p_value, )
        print 't stat: %f' % (t_value, )
        print 'critical value: %f' % (cv, )
    return is_h0_true, p_value, t_value, cv