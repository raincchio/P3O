with open('plot_data_lr', 'r') as f:
    data = eval(f.read())

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
plt.style.use('seaborn')
rc_fonts = {
    'lines.markeredgewidth': 1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize':12,
    'ytick.labelsize':12,
    "font.family": "serif",
    "font.size": 12,
    'axes.titlesize':12,
    "legend.fontsize":10,
    # 'figure.figsize': (12, 12/3.0*0.75),
    'figure.figsize': (7, 7/2.0*0.7),
    "text.usetex": True,
    # 'text.latex.preview': True,
    'text.latex.preamble':
        r"""
        \usepackage{times}
        \usepackage{helvet}
        \usepackage{courier}
        """,
}

matplotlib.rcParams.update(rc_fonts)
f, axarr = plt.subplots(1, 2, sharex=False, squeeze=False)
ll ={}
fmts=['-xr', '-+y', '-.b', '-s','-*', '-^', ]
for env, fmt in zip(data.keys(), fmts):
    if env in ['HalfCheetah', 'Ant', 'Walker2d']:
        ax = axarr[0][1]
    else:
        ax = axarr[0][0]
    x = [float(k) for k in data[env].keys()]
    y = np.array(list(data[env].values()))
    y = y /(y.max()-y.min())
    # yp = (y - y.min())/(y.max()-y.min())
    # yp = y.std() + y.mean()
    l = ax.plot(list(x), list(y),fmt, label=env)
    ll[env] = l
    legend = ax.legend()
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    legend.get_frame().set_edgecolor((0, 0, 0, 0))


# axarr[0][0].ticklabel_format(style='sci',scilimits=(0,0),axis='x')
axarr[0][0].set_ylabel('Normalized Reward')
# axarr[0][0].set_xlabel('(a)Learning Rate')

axarr[0][1].set_title('(b)Continuous Environment')
axarr[0][0].set_title('(a)Discrete Environment')
plt.tight_layout()
# legend = f.legend(ll, bbox_to_anchor=(0.5,-0.08), loc="lower center",bbox_transform=f.transFigure, ncol=6,borderaxespad=0)
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor((0, 0, 0, 0))
# legend.get_frame().set_edgecolor((0, 0, 0, 0))
# plt.tight_layout()
save_name='lr_para'
f.savefig(save_name+'.pdf',bbox_inches='tight',dpi=300)
# plt.show()