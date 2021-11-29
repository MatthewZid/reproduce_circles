import pickle as pkl
import matplotlib.pyplot as plt

states, actions = pkl.load(open("expert_traj.pkl", "rb"))

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.scatter(states[0][:,-2],states[0][:,-1], c=['red'], alpha=0.4)
plt.scatter(states[1][:,-2],states[1][:,-1], c=['blue'], alpha=0.4)
plt.scatter(states[2][:,-2],states[2][:,-1], c=['green'], alpha=0.4)
plt.show()