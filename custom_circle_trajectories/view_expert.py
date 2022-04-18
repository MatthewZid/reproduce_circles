import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

states, _, codes = pkl.load(open("expert_traj.pkl", "rb"))
states = np.concatenate(states)
codes = np.concatenate(codes)

colors = ['red','green','blue']
color_list = []
for i in range(codes.shape[0]):
    argcolor = np.where(codes[i] == 1)[0][0] # find the index of code from one-hot
    color_list.append(colors[argcolor])

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.scatter(states[:,-2], states[:,-1], c=color_list, alpha=0.4)
plt.show()