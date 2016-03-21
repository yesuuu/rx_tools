import matplotlib.pyplot as plt
import scipy.stats as stat
import numpy as np
import seaborn


def plot_distribution(x, bins='auto', plot_title=None, distribution=''):
    """
    input:  bins : 'auto' or int
            distribution: '' or 'norm'
    """
    x = np.array(x).ravel()
    x = x[~np.isnan(x)]
    if len(x) <= 10:
        print 'Warning(in plot distribution): data length: %d' % (len(x),)
    assert bins == 'auto' or type(bins) == int
    if bins == 'auto':
        bins = int(np.sqrt(len(x))) if len(x) >= 100 else (10 if len(x) >= 10 else len(x))
    all_distribution_list = ['', 'norm']
    assert distribution in all_distribution_list
    normed = False if distribution == '' else True
    plt.hist(x, bins=bins, normed=normed)
    title_str = plot_title if plot_title else 'distribution hist'
    plt.title(title_str)
    if distribution == 'norm':
        norm_mean = np.mean(x)
        norm_std = np.std(x)
        xlist_0 = [i/1000. for i in range(1001)]
        xlist_1 = [stat.norm.ppf(i) for i in xlist_0]
        ylist = [stat.norm.pdf(i) for i in xlist_1]
        xlist = norm_mean + norm_std*np.array(xlist_1)
        plt.plot(xlist, ylist)
    plt.show()


