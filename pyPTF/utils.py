"""Module containing helper function to plot and summarise a PTF."""

import matplotlib.pyplot as plt
import numpy as np


def find_optim_cluster(fitted_fkme, y_true, y_pred, conf=0.95):
    """Plot a sequence of :class:`~.fkmeans.FKMEx` objects to find the optimal
    number of clusters.

    Parameters
    ----------
    fitted_fkme : list
        :class:`~.fkmeans.FKMEx` objects to plot.
    y_true : np.ndarray
        Observed values of the target variable.
    y_pred : np.ndarray
        Predicted values of the target variable.
    conf : float
        Confidence level used when estimimating the prediction_interval (the default is 0.95).

    Returns
    -------
    matplotlib.pyplot

    """
    MPIs = [x.MPI(y_true, y_pred) for x in fitted_fkme]
    PICPs = [x.PICP(y_true, y_pred) for x in fitted_fkme]
    ns = [x.nclass for x in fitted_fkme]

    fig, ax1 = plt.subplots()
    l1, = ax1.plot(PICPs, 's-', color='C0')
    ax1.set_ylabel('PICP (%)')
    ax1.set_xlabel('Number of clusters')
    l3 = ax1.axhline(y=conf * 100, color='k', linestyle='--', linewidth=1)

    ax2 = ax1.twinx()
    l2, = ax2.plot(MPIs, '^-', color='C1')
    ax2.set_ylabel('MPI')

    plt.xticks(range(len(fitted_fkme)), ns)

    plt.legend([l1, l2, l3], ['PICP', 'MPI', r'1 - $\alpha$'], ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))


def extragrade_limit(ptf, k):
    """Plot the limit of the extragrade class.

    Parameters
    ----------
    ptf : :class:`~.ptf.PTF`
        Fitted PTF object.
    k : :class:`~.fkmeans.FKMEx`
        Fitted FKMEx object.

    Returns
    -------
    matplotlib.pyplot

    """

    if len(ptf.xs) > 2:
        raise NotImplemented('At the moment, a maximum of 2 predictors is supported.')

    def gen_grid(m, k):
        mins = ptf.cleaned_data[ptf.xs].values.min(0)
        maxes = ptf.cleaned_data[ptf.xs].values.max(0)

        x = np.linspace(mins[0], maxes[0], 100)
        y = np.linspace(mins[1], maxes[1], 100)

        X, Y = np.meshgrid(x, y)

        arr = np.array(list(zip(np.ravel(X), np.ravel(Y))))
        z = k.membership(arr)[:, -1]
        Z = z.reshape(X.shape)
        return X, Y, Z

    X, Y, Z = gen_grid(ptf, k)

    eg_limit = 1.0 / (k.nclass + 1)

    CS = plt.contour(X, Y, Z, [eg_limit], linestyles='dashed')
    # plt.clabel(CS, inline=1, fontsize=10)
    plt.scatter(*ptf.cleaned_data[ptf.xs].values.T,
                c='grey', alpha=0.2, edgecolors='none', label='Training data')
    plt.scatter(*k.centroids.T, edgecolors='w', label='Centroids')
    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    plt.xlabel('Clay (%)')
    plt.ylabel('Sand (%)')

    labels = ['Extragrade class limit ({0:.2f})'.format(eg_limit)]
    for i in range(len(labels)):
        CS.collections[i].set_label(labels[i])

    plt.legend()
    return plt


def summary(ptf, nan_rep='--', decimals=2):
    """Create a PTF summary as LaTeX table.

    Parameters
    ----------
    ptf : :class:`~.ptf.PTF`
        Fitted PTF object.
    nan_rep : str
        How nan data is shown (the default is '--').
    decimals : int
        Number of decimal places to use (the default is 2).

    Returns
    -------
    str
        LaTeX table.

    """
    n_props = ptf.uncertainty.centroids.shape[1]
    begin = '\\begin{{tabular}}{{c{}cc}}\n'.format('c' * len(ptf.xs))
    end = '\\end{tabular}'
    toprule = '\\toprule\n'
    bottomrule = ' \\\\\n\\bottomrule\n'
    midrule = '\\midrule\n'
    sep = '\\cmidrule(rl){{2-{}}} \\cmidrule(rl){{{}-{}}}\n'.format(len(ptf.xs) + 1,
                                                                    len(ptf.xs) + 2,
                                                                    len(ptf.xs) + 3)

    k = ptf.uncertainty.nclass
    formater = '{{0:.{}f}}'.format(decimals)

    centroids = ptf.uncertainty.centroids
    centroids = np.r_[centroids, np.array([[np.nan] * n_props])]

    res = ptf.PIC
#     mean_res = np.mean(res, 1)
#     res = np.c_[res[:, 0], mean_res, res[:, 1]]

    all_cols = np.c_[centroids, res]

    all_rows = [" & ".join(map(lambda x: formater.format(x), line)) for line in all_cols]
    all_rows = ['{} & '.format(i) + l if i <= k else 'Eg & ' + l for i, l in enumerate(all_rows, 1)]

    colnames = '{} & ' + ' & '.join(ptf.xs) + ' & PI_{L} & PI_{U} \\\\\n'
    head = 'Clusters & \\multicolumn{{{}}}{{c}}{{Centroids}} & '.format(n_props) + \
        '\\multicolumn{2}{c}{Cluster residuals} \\\\\n'

    table_str = begin + toprule + colnames + midrule + head + sep + \
        " \\\\\n".join(all_rows) + bottomrule + end
    return table_str.replace('nan', nan_rep)
