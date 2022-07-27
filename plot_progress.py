

# logp = '/home/chenxing/extra_test/BeamRider/ppo2_1/progress.csv'
logp = '/home/chenxing/extra_test/BeamRider/ddpo_1/progress.csv'
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


from matplotlib import pyplot as plt
import numpy as np

unwanted = ['eplenmean', 'fps', ]
for u in unwanted:
    data.pop(u)
wanted = ['loss/max_ratio']



i=1
fig = plt.figure(figsize=(9,9))

for key, value in data.items():
    if i>9:continue
    N=10
    weights = np.exp(np.linspace(0,1,N))
    weights = weights/np.sum(weights)
    value = np.convolve(weights, value, mode='valid')
    plt.subplot(3, 3, i)
    plt.title(key)
    plt.plot(value)
    i+=1
# plt.subplot(3, 3, i)
# plt.title(logp.split('/')[-1])


env_name = logp.split('/')[-3]
method_name = logp.split('/')[-2]
method_name = method_name.replace('ddpo','P3O').replace('ppo2','PPO').replace('vpgdualclip','Dual-clip PPO')
figname= method_name +' Analysis in '+ env_name
fig.suptitle(figname)

fig = plt.gcf()

plt.show()
# fig.savefig('test/'+figname+'.pdf',bbox_inches='tight',dpi=300, backend='pdf')