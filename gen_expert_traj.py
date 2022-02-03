import math
import numpy as np
import pickle as pkl
from circle_env import CircleEnv
from tqdm import tqdm


#-----------------------
# Circle equation
#-----------------------
def circle(theta, r):
	return r*math.cos(theta), r*math.sin(theta)


#-----------------------
# Expert policy step
#-----------------------
def policy_step(p, r, up=True, clockwise=True):
	if up:
		offset = r
	else:
		offset = -r

	if clockwise:
		theta = math.atan2(p[1]-offset, p[0]) + 2 * math.asin(0.005 / r)
	else:
		theta = math.atan2(p[1]-offset, p[0]) - 2 * math.asin(0.005 / r)

	p_tar = circle(theta, r)
	dx    = p_tar[0] - p[0]
	dy    = p_tar[1] - p[1] + offset
	norm  = math.sqrt(dx*dx + dy*dy)

	return [dx / norm, dy / norm] if norm > 1e-8 else [0, 0]


#-----------------------
# Main
#-----------------------
def main():
	env       = CircleEnv()
	rs        = [0.4, 0.2, 0.2]
	cs        = [True, True, False]
	us        = [True, True, False]
	n_episode = 1024
	s_traj    = []
	a_traj    = []
	c_traj = []
	idx       = 0

	for i in range(len(rs)):
		print("r = {:.2f}".format(rs[i]))

		for j in tqdm(range(128)):
			state = env.reset()
			s_traj.append([])
			a_traj.append([])
			c_traj.append([])

			while True:
				#env.render()
				action = policy_step(state[-2:], rs[i], up=us[i], clockwise=cs[i])
				
				s_traj[idx].append(state)
				a_traj[idx].append(action)
				one_hot_code = np.zeros(len(rs))
				one_hot_code[i] = 1
				c_traj[idx].append(one_hot_code)

				state, reward, done, info = env.step(action)

				if done:
					s_traj[idx] = np.array(s_traj[idx], dtype=np.float32)
					a_traj[idx] = np.array(a_traj[idx], dtype=np.float32)
					c_traj[idx] = np.array(c_traj[idx], dtype=int)
					idx += 1
					break

	pkl.dump((s_traj, a_traj, c_traj), open("./expert_traj.pkl", "wb"))


if __name__ == '__main__':
	main()