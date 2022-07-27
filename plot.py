import os

import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.rc('ytick', labelsize=8)
# plt.rc('xtick', labelsize=8)
# plt.rcParams['legend.title_fontsize'] = 12
# plt.rc('font', size=12)

# rc_fonts = {
#     'xtick.direction': 'in',
#     'ytick.direction': 'in',
#     'xtick.labelsize':16,
#     'ytick.labelsize':16,
#     "font.family": "serif",
#     "font.size": 16,
#     'axes.titlesize':16,
#     "legend.fontsize":13,
#     'figure.figsize': (12, 12/3.0*0.75*2),
#     # 'figure.figsize': (7, 7/2.0*0.75),
#     "text.usetex": True,
#     # 'text.latex.preview': True,
#     'text.latex.preamble':
#         r"""
#         \usepackage{times}
#         \usepackage{helvet}
#         \usepackage{courier}
#         """,
# }
# matplotlib.rcParams.update(rc_fonts)
plt.style.use('seaborn')
from baselines.common import plot_util

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_REWARD = 'reward'
Y_TIMESTEPS = 'timesteps'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue']

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

def plot_results(dirs, num_timesteps=10e6, xaxis=X_TIMESTEPS, yaxis=Y_REWARD, title='',row=1,  inches=7):
    results = plot_util.load_results(dirs, enable_monitor=False, enable_progress=True)
    # plot_util.plot_results(results, xy_fn=lambda r: ts2xy(r.monitor, xaxis, yaxis), split_fn=split_fn, average_group=True, resample=int(1e6))
    plot_util.plot_results(results, split_fn=group_by_name, group_fn=group_by_seed, average_group=True,
                           shaded_std=True,shaded_err=False, xlabel=X_TIMESTEPS,
                           ylabel=Y_REWARD,row=row)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    path = [r"C:\Users\chenxing\extra_test"]

    key = path
    parser.add_argument('--dirs', help='List of log directories', nargs = '*', default=key)
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument('--yaxis', help = 'Varible on Y-axis', default = Y_REWARD)
    parser.add_argument('--task_name', help = 'Title of plot', default = 'Breakout')
    args = parser.parse_args()
    # args.dirs = [os.path.abspath(dir) for dir in args.dirs]

    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.yaxis, args.task_name, row=1)
    plt.show()


if __name__ == '__main__':
    main()
    # paper_image()
