
logp = '/home/chenxing/extra_test/BeamRider/ddpo_4/log.txt'
with open(logp, 'r') as f:
    logs = f.read().split('\n')

data = {}
for log in logs:
    if 'loss' in log:
        dat = log.split('|')
        key, value = dat[1].strip(),float(dat[2])
        if key in data.keys():
            data[key].append(value)
        else:
            data[key] = [value]

from matplotlib import pyplot as plt
import numpy as np

i=1
plt.figure(figsize=(12,8))
for key, value in data.items():
    N=10
    weights = np.exp(np.linspace(0,1,N))
    weights = weights/np.sum(weights)
    value = np.convolve(weights, value, mode='valid')
    plt.subplot(2, 3, i)
    plt.title(key)
    plt.plot(value)
    i+=1
plt.subplot(2, 3, i)
plt.title(logp.split('/')[-2])

plt.show()