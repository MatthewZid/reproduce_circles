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

# fig = plt.figure()
# ax  = fig.add_subplot(111)
# ax.set_aspect(aspect=1.0)
# # mng = plt.get_current_fig_manager()
# # mng.resize(*mng.window.maxsize())
# plt.grid()
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)

# colors = [
# 	[np.array([1.0, 0.0, 0.0])], #red
# 	[np.array([0.0, 1.0, 0.0])], #green
# 	[np.array([0.0, 0.0, 1.0])]  #blue
# ]

# for i in tqdm(range(len(colors))):
# 	for j in range(8):
# 		rand_idx = np.random.randint(i*1024, (i+1)*1024)
# 		xs = []
# 		ys = []

# 		for k in range(len(s_traj[rand_idx])):
# 			xs.append(s_traj[rand_idx][k, -2])
# 			ys.append(s_traj[rand_idx][k, -1])

# 		ax.scatter(xs, ys, s=4, c=colors[i], alpha=0.4)



# plt.show()