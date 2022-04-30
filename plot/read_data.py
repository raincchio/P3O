import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn')
rc_fonts = {
    'lines.markeredgewidth': 1,
    "lines.markersize":3,
    "lines.linewidth":1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    "font.family": "times",
    'axes.titlesize':11,
    "legend.fontsize":8,
    'figure.figsize': (4, 4*0.8),
    "text.usetex": True,
    # 'text.latex.preview': True,
    'text.latex.preamble':
        r"""
        \usepackage{times}
        \usepackage{helvet}
        \usepackage{courier}
        """,
}
wev = None
for i in range(1,5):

    logp = '../extra_test/Walker2d/ddpo_'+str(i)+'/progress.csv'

    with open(logp, 'r') as f:
        logs = f.read().split('\n')

    data = {}
    keys = logs[0].split(',')
    for key in keys:
        data[key] = []

    for log in logs[1:]:
        if len(log)<2:
            continue
        values = log.split(',')
        a,b = keys[:10], values[:10]
        for key,value in zip(keys, values):
            data[key].append(float(value))
    if wev is None:
        wev = np.array(data['misc/explained_variance'])
    else:
        wev += np.array(data['misc/explained_variance'])

hev = None
for i in range(1,5):

    logp = '../extra_test/HalfCheetah/ddpo_'+str(i)+'/progress.csv'

    with open(logp, 'r') as f:
        logs = f.read().split('\n')

    data = {}
    keys = logs[0].split(',')
    for key in keys:
        data[key] = []

    for log in logs[1:]:
        if len(log)<2:
            continue
        values = log.split(',')
        a,b = keys[:10], values[:10]
        for key,value in zip(keys, values):
            data[key].append(float(value))
    if hev is None:
        hev = np.array(data['misc/explained_variance'])
    else:
        hev += np.array(data['misc/explained_variance'])


aev = None
for i in range(1,5):

    logp = '../extra_test/Ant/ddpo_'+str(i)+'/progress.csv'

    with open(logp, 'r') as f:
        logs = f.read().split('\n')

    data = {}
    keys = logs[0].split(',')
    for key in keys:
        data[key] = []

    for log in logs[1:]:
        if len(log)<2:
            continue
        values = log.split(',')
        a,b = keys[:10], values[:10]
        for key,value in zip(keys, values):
            data[key].append(float(value))
    if aev is None:
        aev = np.array(data['misc/explained_variance'])
    else:
        aev += np.array(data['misc/explained_variance'])

data = {
    'Walker2d':wev/4,
    "HalfCheetah":hev/4,
    "Ant":aev/4,
}



# figsize=(5,5*0.8)
fig = plt.figure(figsize=(4.25,4.25*0.6))
i=0
COLORS= ['#4c72b0','#55a868','#c44e52','#8172b2','#ccb974']
fmts=['-^', '-v', '-.', '-s', '-*']
for key, value in data.items():
    N=30
    weights = np.exp(np.linspace(0,1,N))
    weights = weights/np.sum(weights)
    value = np.convolve(weights, value, mode='valid')
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=8)
    plt.plot(value,fmts[i],color=COLORS[i],label=key,markevery=60)
    i+=1

plt.legend()
plt.ylabel('value')
plt.xlabel('iteration')
fig.suptitle("Explained Variance",y=1)

fig = plt.gcf()

# plt.show()
fig.savefig('../test/explained_variance.pdf',bbox_inches='tight',dpi=300, backend='pdf')