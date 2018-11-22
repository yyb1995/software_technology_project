import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


def save_init(savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    plt.rcParams['font.size'] = 16


# static picture section
def static_initial_location(ind_loc_set, inc_loc_set, max_x, max_y, lucky_percentage, savepath, image_format):
    fig1 = plt.figure(1, figsize=(12, 12), dpi=100)
    plt.scatter(ind_loc_set[:, 0], ind_loc_set[:, 1], color='k', marker='^', linewidths=2, label='individual')
    plt.scatter(inc_loc_set[:int(inc_loc_set.shape[0] * lucky_percentage), 0],
                inc_loc_set[:int(inc_loc_set.shape[0] * lucky_percentage), 1],
                color='g', marker='o', label='lucky incident')
    plt.scatter(inc_loc_set[int(inc_loc_set.shape[0] * lucky_percentage):, 0],
                inc_loc_set[int(inc_loc_set.shape[0] * lucky_percentage):, 1],
                color='r', marker='o', label='unlucky incident')
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.legend(bbox_to_anchor=(0.15, 0), loc="lower left", bbox_transform=fig1.transFigure, ncol=3)
    plt.title('Initial location of individual and incident')
    plt.savefig(savepath + 'initial_location' + image_format)
    plt.draw()


def static_talent(tal_set, savepath, image_format):
    xaxis_min = 0.2
    xaxis_max = 1.0
    step = 0.01
    xaxis_step = 0.05
    plt.figure(2, figsize=(12, 8), dpi=100)
    plt.hist(tal_set, bins=np.arange(xaxis_min, xaxis_max, step), label=np.arange(xaxis_min, xaxis_max, step))
    plt.xticks(np.arange(xaxis_min, xaxis_max + xaxis_step, xaxis_step), fontsize=14)
    plt.xlabel('mean = %.2f   standard_variance = %.2f' % (tal_set.mean(), tal_set.std()))
    plt.title('Initial talent distribution')
    plt.savefig(savepath + 'static_talent' + image_format)
    plt.draw()


def static_capital_individual_num(capital_set, savepath, image_format):
    def linear_func(x, k, b):
        return x * k + b

    def power_func(x, k, b):
        return np.power(x, k) * np.power(10, b)

    capital_set = np.array(capital_set)
    plt.figure(3, figsize=(12, 12), dpi=100)
    xaxis_min = 0
    xaxis_max = capital_set.max()
    step = 25
    xaxis_step = 500
    plt.subplot(211)
    plt.hist(capital_set, bins=np.arange(xaxis_min, xaxis_max, step), label=np.arange(xaxis_min, xaxis_max, step),
             color='r', edgecolor='k')
    plt.xticks(np.arange(xaxis_min, xaxis_max + xaxis_step, xaxis_step), fontsize=14)
    plt.xlabel('capital')
    plt.ylabel('individual_num')
    plt.ylim(1, len(capital_set))
    capital_set_sort = np.sort(capital_set)
    ind_perc = 20
    cap_perc = np.sum(capital_set_sort[-int(len(capital_set) * ind_perc / 100):]) / np.sum(capital_set_sort) * 100
    plt.title('Capital distribution\nThe {}% richest individuals own {}% capital'.format(ind_perc, int(cap_perc)))
    plt.semilogy()
    plt.subplot(212)
    x_boundary = np.array([0, 10, 50, 100, 150, 200, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000])
    x_capital_set = np.array([(x_boundary[i] + x_boundary[i + 1]) / 2 for i in range(len(x_boundary) - 1)])
    y_indnum_set = np.array([np.sum(list(map(int, (capital_set > x_boundary[i]) & (capital_set < x_boundary[i + 1]))))
                            for i in range(len(x_boundary) - 1)])
    curve_set = np.concatenate((x_capital_set.reshape(1, -1), y_indnum_set.reshape(1, -1)), axis=0)
    curve_set_nz = np.delete(curve_set, np.argwhere(y_indnum_set == 0), axis=1)
    curve_set_nz_log = np.log10(curve_set_nz)
    plt.scatter(curve_set_nz[0, :], curve_set_nz[1, :], marker='o', linewidths=2, color='silver', edgecolors='k')
    k_, b_ = curve_fit(linear_func, curve_set_nz_log[0, :], curve_set_nz_log[1, :])[0]
    y_ = power_func(curve_set_nz[0, :], k_, b_)
    plt.plot(curve_set_nz[0, :], y_, color='r', label='Fitted Power law\nslope:{:.3f}'.format(k_))
    plt.legend(loc="upper right")
    plt.xlim(1, 10 * x_capital_set[-1])
    plt.ylim(0.5, len(capital_set))
    plt.loglog()
    # plt.tight_layout()
    plt.savefig(savepath + 'static_capital_individual_num' + image_format)
    plt.draw()


def static_capital_talent(cap_set, tal_set, savepath, image_format):
    curve_set = np.concatenate((tal_set.reshape(1, -1), cap_set.reshape(1, -1)), axis=0)
    curve_set_sorted = curve_set[:, curve_set[0].argsort()]
    plt.figure(4, figsize=(12, 8), dpi=100)
    markerline, stemlines, baseline = plt.stem(curve_set_sorted[0, :], curve_set_sorted[1, :], linefmt='-',
                                               markerfmt='.')
    plt.setp(stemlines, color='r', linewidth=1)
    plt.xlabel('talent')
    plt.ylabel('capital')
    plt.title('Talent & capital distribution')
    plt.savefig(savepath + 'static_capital_talent' + image_format)
    plt.draw()


def static_capital_incident(cap_set, lucky_set, unlucky_set, savepath, image_format):
    lucky_curve_set = np.concatenate((lucky_set.reshape(1, -1), cap_set.reshape(1, -1)), axis=0)
    unlucky_curve_set = np.concatenate((unlucky_set.reshape(1, -1), cap_set.reshape(1, -1)), axis=0)
    lucky_curve_set_sorted = lucky_curve_set[:, lucky_curve_set[0].argsort()]
    unlucky_curve_set_sorted = unlucky_curve_set[:, unlucky_curve_set[0].argsort()]
    plt.figure(5, figsize=(12, 12), dpi=100)
    plt.subplot(211)
    markerline, stemlines, baseline = plt.stem(lucky_curve_set_sorted[0, :], lucky_curve_set_sorted[1, :],
                                               linefmt='-', markerfmt='.')
    plt.setp(stemlines, color='g', linewidth=1)
    plt.xlabel('lucky incident num')
    plt.ylabel('capital')
    plt.title('Incident & capital distribution')
    plt.subplot(212)
    markerline, stemlines, baseline = plt.stem(unlucky_curve_set_sorted[0, :], unlucky_curve_set_sorted[1, :],
                                               linefmt='-', markerfmt='.')
    plt.setp(stemlines, color='r', linewidth=1)
    plt.xlabel('unlucky incident num')
    plt.ylabel('capital')
    plt.savefig(savepath + 'static_capital_incident' + image_format)


def select_richest_poorest(talent_set, final_capital_set, full_incident_set, savepath, image_format):
    # get the richest and poorest individual's capital and incident information
    # full_incident_set: shape is (individual_num, incident_num, 3)
    richest_num = np.argmax(final_capital_set)
    poorest_num = np.argmin(final_capital_set)
    richest_record = full_incident_set[richest_num, ::]
    poorest_record = full_incident_set[poorest_num, ::]
    richest_lucky_record = (richest_record[:, 1] == 1).astype(int)
    richest_unlucky_record = (richest_record[:, 1] == -1).astype(int)
    poorest_lucky_record = (poorest_record[:, 1] == 1).astype(int)
    poorest_unlucky_record = (poorest_record[:, 1] == -1).astype(int)
    plt.figure(6, figsize=(12, 12), dpi=100)
    plt.subplot(211)
    plt.plot(richest_record[:, 2])
    plt.title('richest individual capital record\ntalent:{0:.3f}'.format(talent_set[richest_num]))
    plt.subplot(212)
    plt.plot(richest_lucky_record, color='g', label='lucky')
    plt.plot(richest_unlucky_record, color='r', label='unlucky')
    plt.legend(loc='upper right')
    plt.title('richest individual incident record')
    plt.tight_layout()
    plt.savefig(savepath + 'richest_individual_record' + image_format)
    plt.draw()
    plt.figure(7, figsize=(12, 12), dpi=100)
    plt.subplot(211)
    plt.plot(poorest_record[:, 2])
    plt.title('poorest individual capital record\ntalent:{}'.format(talent_set[poorest_num]))
    plt.subplot(212)
    plt.plot(poorest_lucky_record, color='g', label='lucky')
    plt.plot(poorest_unlucky_record, color='r', label='unlucky')
    plt.legend(loc='upper right')
    plt.title('poorest individual incident record')
    plt.tight_layout()
    plt.savefig(savepath + 'poorest_individual_record' + image_format)
    plt.draw()


def static_multiple_richest_talent(multi_tal_set, savepath, image_format):
    multi_tal_set = np.array(multi_tal_set)
    xaxis_min = 0.2
    xaxis_max = 1.0
    step = 0.01
    xaxis_step = 0.05
    plt.figure(8, figsize=(12, 8), dpi=100)
    plt.hist(multi_tal_set, bins=np.arange(xaxis_min, xaxis_max, step))
    plt.xticks(np.arange(xaxis_min, xaxis_max + xaxis_step, xaxis_step), fontsize=14)
    plt.xlabel('mean = %.2f   standard_variance = %.2f' % (multi_tal_set.mean(), multi_tal_set.std()))
    plt.title('Multiple richest talent distribution')
    lnspc = np.linspace(xaxis_min, xaxis_max, len(multi_tal_set))
    # get mean and standard deviation
    m, s = stats.norm.fit(multi_tal_set)
    # now get theoretical values in our interval
    pdf_g = stats.norm.pdf(lnspc, m, s)
    plt.plot(lnspc, pdf_g, label="gaussian distribution curve", linestyle='--')  # plot it
    plt.legend()
    plt.savefig(savepath + 'static_multi_richest_talent' + image_format)
    plt.draw()


# animation section
def anime_incident_location():
    pass  # TODO: Implement incident movement


def anime_capital_individual_num():
    pass  # TODO: Implement capital individual figure movement


def anime_capital_talent():
    pass  # TODO: Implement capital talent figure movement


def anime_capital_lucky_incident():
    pass  # TODO: Implement capital lucky figure movement
