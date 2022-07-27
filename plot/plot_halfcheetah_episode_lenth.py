import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, namedtuple
from baselines.common.plot_util import smooth,symmetric_ema
import os

rc_fonts = {
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    "font.family": "times",
    "font.size": 10,
    'axes.titlesize':10,
    "legend.fontsize":10,
    'figure.figsize': (5, 3),# 3.5, 5, 7.2
    # 'figure.figsize': (7, 7/2.0*0.75),
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
# plt.style.use('seaborn')
from baselines.common import plot_util

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_REWARD = 'reward'
Y_TIMESTEPS = 'timesteps'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'cyan', 'magenta', 'purple',
          'orange', 'teal', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']

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

def group_by_seed(taskpath):
    path = taskpath.dirname.split(os.sep)[-1].split('_')
    return "_".join(path[2:])


def group_by_name(taskpath):
    return taskpath.dirname.split(os.sep)[-2]
def default_xy_fn(r):
    try:
        x = np.cumsum(r.monitor.l)
        y = smooth(r.monitor.r, radius=10)
    except:
        y = smooth(r.progress['return-average'], radius=10)
        x = r.progress['total-samples']
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
        inches=7
):

    sk2r = defaultdict(list) # splitkey2results
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)
    ll = len(sk2r)
    nrows=row
    ncols=ll//nrows

    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False)


    groups = list(set(group_fn(result) for result in allresults))

    default_samples = 512
    if average_group:
        resample = resample or default_samples
    fmts=['-x', '-+', '-.', '-s','-*', '-^', ]
    g2ls = []
    g2cs = []
    for (isplit, sk) in enumerate(sk2r.keys()):
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
            for idx, group in enumerate(sorted(groups)):
                xys = gresults[group]
                if not any(xys):
                    continue
                if group=='ddpo':
                    color = 'red'
                    fmt = '-'
                else:
                    color = COLORS[idx % len(COLORS)]
                    fmt = fmts[idx % len(fmts)]
                # print(groups.index(group), idx)
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))
                def allequal(qs):
                    return all((q==qs[0]).all() for q in qs[1:])
                if resample:
                    print(isplit, sk)
                    low = max(x[0] for x in origxs)
                    # if sk in ['Enduro', 'Breakout', 'BeamRider']:
                    #     high = 9e6
                    #
                    # else:
                    #     high = min(x[-1] for x in origxs)
                    high = min(x[-1] for x in origxs)
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

                l, = axarr[idx_row][idx_col].plot(usex, ymean, fmt, color=color,markevery=default_samples//need_point)
                # set_size(4, 3,axarr[idx_row][idx_col])
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

        plt.tight_layout()


        ax.set_title('('+chr(isplit+97)+') '+sk)

        if xlabel is not None:
            for ax in axarr.flatten():
                plt.sca(ax)
                # plt.xlabel('('+chr(id+97)+') '+xlabel)
                plt.xlabel('timesteps')
        if ylabel is not None:
            for ax in axarr[:,0]:
                plt.sca(ax)
                plt.ylabel(ylabel)
        g2ls.append(g2l)
    tt= {'ddpo':'P3O','vpgdualclip':'Dual-Clip PPO',
         'acktr':'ACKTR','ppo2':'PPO','trpo':'TRPO','a2c':'A2C'}

    tt_s = g2ls[0]

    ad = tt_s

    # for key in tt.keys():
    #     if key in tt_s.keys():
    #         ad[key] = tt_s[key]

    axarr[0][0].legend(ad.values(), [tt[g] if g in tt.keys() else g for g in ad],edgecolor='None', facecolor='None',loc=(1.05, 0.25))

    return f, axarr





def paper_image():

    path = [r'C:\Users\chenxing\0323\HalfCheetah_episode_length',
            ]
    save_name = 'HalfCheetah_episode_length'

    results = plot_util.load_results(path, enable_monitor=True, enable_progress=False)
    plot_results(results, split_fn=group_by_name, group_fn=group_by_seed, average_group=True,
                 shaded_std=True,shaded_err=False, xlabel=X_TIMESTEPS,
                 ylabel=Y_REWARD,row=1)

    fig = plt.gcf()
    fig.savefig('png'+os.sep+save_name+'.pdf',bbox_inches='tight',dpi=300, backend='pdf')
    # fig.savefig('png/'+save_name+'.pdf',dpi=300, backend='pdf')

if __name__ == '__main__':

    paper_image()
