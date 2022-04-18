import math
import numpy as np
import pickle as pkl
from circle_env import CircleEnv
from tqdm import trange

env = CircleEnv()
s_traj = []
a_traj = []
c_traj = []
r = [0.5, 0.25, 0.25]
n = 128
traj_num = 128
c = ((0,0.25), (0,0), (0,-0.5))
noise = 0.01
idx = 0

for i in range(len(r)):
	for j in trange(traj_num):
		state = env.reset()
		s_traj.append([])
		a_traj.append([])
		c_traj.append([])

		up = 1
		theta = 360.0 / n
		
		if state[-1] < c[i][1]: up = -1

		angle = theta
		for a in np.arange(angle, 361.0, theta):
			if (a + theta) < 0.00000001:
				print('Very high resolution! Terminating...')
				break
			newx = up * r[i] * math.sin(math.radians(a)) + c[i][0] + noise*np.random.randn()
			newy = up * r[i] * math.cos(math.radians(a)) + c[i][1] + noise*np.random.randn()

			dx = newx - state[-2]
			dy = newy - state[-1]

			s_traj[idx].append(state)
			a_traj[idx].append([dx, dy])
			one_hot_code = np.zeros(len(r))
			one_hot_code[i] = 1
			c_traj[idx].append(one_hot_code)

			state, _ = env.step([dx,dy], mode=False)
		
		s_traj[idx] = np.array(s_traj[idx], dtype=np.float32)
		a_traj[idx] = np.array(a_traj[idx], dtype=np.float32)
		c_traj[idx] = np.array(c_traj[idx], dtype=int)
		idx += 1

pkl.dump((s_traj, a_traj, c_traj), open("./expert_traj.pkl", "wb"))