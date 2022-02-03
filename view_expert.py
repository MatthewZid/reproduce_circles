import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from tqdm import tqdm


s_traj, _, c_traj = pkl.load(open("expert_traj.pkl", "rb"))
s_traj = np.concatenate(s_traj)
c_traj = np.concatenate(c_traj)

colors = ['red','green','blue']
color_list = []
for i in range(c_traj.shape[0]):
    argcolor = np.where(c_traj[i] == 1)[0][0] # find the index of code from one-hot
    color_list.append(colors[argcolor])

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.scatter(s_traj[:,-2], s_traj[:,-1], s=4, c=color_list, alpha=0.4)
plt.show()