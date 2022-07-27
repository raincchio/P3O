import numpy as np
from collections import defaultdict
from baselines.common.plot_util import smooth,symmetric_ema
import matplotlib.pyplot as plt
from baselines.common import plot_util
import os
import matplotlib
import matplotlib.font_manager
# plt.style.use('seaborn')
rc_fonts = {
#8.5
    # 'lines.markeredgewidth': 1,
    # "lines.markersize":3,
    # "lines.linewidth":1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    "font.family": "times",
    "font.size": 10,
    'axes.titlesize':10,
    "legend.fontsize":10,
    'figure.figsize': (8.5, 4.5),
    # "text.usetex": True,
    # 'text.latex.preview': True,
    # 'text.latex.preamble':
    #     r"""
    #     \usepackage{times}
    #     \usepackage{helvet}
    #     \usepackage{courier}
    #     """,
}
matplotlib.rcParams.update(rc_fonts)
fmts=['-^', '-v', '-.', '-s', '-*']
# fmts=['-+', '-.', '-s','-*', '-^']
X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_REWARD = 'reward'
Y_TIMESTEPS = 'timesteps'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']

COLORS= ['#4c72b0','#55a868','#c44e52','#8172b2','#ccb974']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis, yaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
    else:
        raise NotImplementedError
    if yaxis == Y_REWARD:
        y = ts.r.values
    elif yaxis == Y_TIMESTEPS:
        y = ts.l.values
    else:
        raise NotImplementedError
    return x, y

def plot_curves(xy_list, xaxis, yaxis, title):
    fig = plt.figure(figsize=(8,2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i % len(COLORS)]
        plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    fig.canvas.mpl_connect('resize_event', lambda event: plt.tight_layout())
    plt.grid(True)


def group_by_seed(taskpath):
    return taskpath.dirname.split(os.sep)[-1].split('_')[0]

def group_by_name(taskpath):
    return taskpath.dirname.split(os.sep)[-2]
def default_xy_fn(r):

    # r.progress['misc/total_timesteps'].values[-1]

    try:
        y = np.nan_to_num(r.progress['eprewmean'],0)
        y = smooth(y, radius=10)
        x = r.progress['misc/total_timesteps']

    except:
        x = np.cumsum(r.monitor.l)
        y = smooth(r.monitor.r, radius=10)
    return x,y

def plot_results(
        allresults, *,
        xy_fn=default_xy_fn,
        split_fn=None,
        group_fn=None,
        average_group=False,
        shaded_std=True,
        shaded_err=True,
        shaded_line=False,
        legend_outside=False,
        resample=0,
        smooth_step=1.0,
        xlabel=None,
        ylabel=None,
        row=1,
        col=1
):


    if split_fn is None: split_fn = lambda _ : ''
    if group_fn is None: group_fn = lambda _ : ''
    sk2r = defaultdict(list) # splitkey2results
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)
    assert len(sk2r) > 0
    assert isinstance(resample, int), "0: don't resample. <integer>: that many samples"

    nrows=row
    ncols=col



    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False)
    # f.set_size_inches(inches, inches*0.75/ncols)

    groups = list(set(group_fn(result) for result in allresults))

    default_samples = 512
    if average_group:
        resample = resample or default_samples

    g2ls = []
    g2cs = []
    # for (isplit, sk) in enumerate(sk2r.keys()):
    for (isplit, sk) in enumerate(['Enduro', 'Breakout', 'BeamRider', 'Ant', 'HalfCheetah', 'Walker2d']):
    # for (isplit, sk) in enumerate(['Breakout', 'Ant']):
        g2l = {}
        g2c = defaultdict(int)
        sresults = sk2r[sk]
        gresults = defaultdict(list)
        idx_row = isplit // ncols
        idx_col = isplit % ncols
        ax = axarr[idx_row][idx_col]
        for result in sresults:
            group = group_fn(result)
            g2c[group] += 1
            x, y = xy_fn(result)
            if x is None: x = np.arange(len(y))
            x, y = map(np.asarray, (x, y))
            if average_group:
                gresults[group].append((x,y))
            else:
                if resample:
                    x, y, counts = symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)
                l, = ax.plot(x, y, color=COLORS[groups.index(group) % len(COLORS)])
                g2l[group] = l
        if average_group:
            # print(sorted(groups))
            sort_groups_ = sorted(groups)
            if "ddpo" in sort_groups_:
                id = sort_groups_.index('ddpo')
                sort_groups_.pop(id)
                sort_groups_.append('ddpo')

            for idx, group in enumerate(sort_groups_):
                xys = gresults[group]
                if not any(xys):
                    continue
                if group=='ddpo':
                    color = 'red'
                    fmt = '-'
                else:
                    color = COLORS[idx % len(COLORS)]
                    fmt = fmts[idx % len(fmts)]
                # print(group, color, fmt)
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))
                def allequal(qs):
                    return all((q==qs[0]).all() for q in qs[1:])
                if resample:
                    # print(isplit, sk)
                    low = max(x[0] for x in origxs)
                    high = min(x[-1] for x in origxs)
                    print(high)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]), \
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                # TODO
                need_point=5
                axarr[idx_row][idx_col].locator_params(axis='x', nbins=10)
                axarr[idx_row][idx_col].locator_params(axis='y', nbins=8)
                internal = default_samples//need_point +idx*10
                l, = axarr[idx_row][idx_col].plot(usex, ymean, fmt, color=color,markevery=internal)

                g2l[group] = l
                if shaded_err:
                    if shaded_line:
                        ax.vlines(usex[::default_samples//20], ymean - ystderr, ymean + ystderr, color=color,alpha=.5)
                    else:
                        ax.fill_between(usex, ymean - ystderr, ymean + ystderr, color=color, alpha=.4)
                if shaded_std:
                    if shaded_line:
                        x = usex[::default_samples//need_point]
                        ymin = ymean - ystd
                        ymax = ymean + ystd
                        ax.vlines(x, ymin[::default_samples//need_point], ymax[::default_samples//need_point], color=color,alpha=.5)
                    else:
                        ax.fill_between(usex, ymean - ystd,    ymean + ystd,    color=color, alpha=.2)

        # https://matplotlib.org/users/legend_guide.html
        plt.tight_layout()

        # ax.set_title('('+chr(isplit+97)+') '+sk, y=-0.4)
        ax.set_title('('+chr(isplit+97)+') '+sk)
        # ax.set_title(sk)
        # add xlabels, but only to the bottom row
        if xlabel is not None:
            for ax in axarr.flatten():
                plt.sca(ax)
                plt.xlabel('timesteps')
        # add ylabels, but only to left column
        if ylabel is not None:
            for ax in axarr[:,0]:
                plt.sca(ax)
                plt.ylabel(ylabel)
        g2ls.append(g2l)
    tt= {'ddpo':'P3O',
         'vpgkl':'P3O-S','vpgsigmoid':'P3O-K','vpg':'P3O-SK'}

    tt_s = g2ls[0]
    print(tt_s)

    ad = {}
    for key in tt.keys():
        if key in tt_s.keys():
            ad[key] = tt_s[key]

    legend= axarr[0][0].legend(ad.values(), [tt[g] if g in tt.keys() else g for g in ad],borderaxespad=0)
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    legend.get_frame().set_edgecolor((0, 0, 0, 0))
    return f, axarr

if __name__ == '__main__':
    path = [

        r"C:\Users\chenxing\0323\DDPO\extra_test_ablation"
            ]

    results = plot_util.load_results(path, enable_monitor=True, enable_progress=True)
    plot_results(results, split_fn=group_by_name, group_fn=group_by_seed, average_group=True,
                           shaded_std=True,shaded_err=False, xlabel=X_TIMESTEPS,
                           ylabel=Y_REWARD,row=2, col=3)
    # plt.show()
    fig = plt.gcf()
    save_name = 'ablation'
    fig.savefig('png'+os.sep+save_name+'.pdf',bbox_inches='tight',dpi=300, backend='pdf')

