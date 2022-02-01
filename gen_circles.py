import math
import numpy as np
import pickle as pkl
from circle_env import CircleEnv
import random
from tqdm import tqdm

env = CircleEnv()
s_traj = []
a_traj = []
c_traj = []
code_prob = []
# r = [0.5, 0.25, 0.25]
r = [0.55, 0.3, 0.3]
n = (10000,5000,5000)
total_points = n[0] + n[1] + n[2]
# c = ((0,0.25), (0,0), (0,-0.5))
c = ((0,0.25), (0,0), (0,-0.59))
noise = 0.05

for i in range(len(r)):
    state = env.reset()
    s_traj.append([])
    a_traj.append([])
    c_traj.append([])

    up = 1
    theta = 360.0 / n[i]
    
    if state[-1] < c[i][1]: up = -1

    angle = theta
    for a in tqdm(np.arange(angle, 360.0, theta)):
        if (a + theta) < 0.00000001:
            print('Very high resolution! Terminating...')
            break
        newx = up * r[i] * math.sin(math.radians(a)) + c[i][0] + random.uniform(-noise,noise)
        newy = up * r[i] * math.cos(math.radians(a)) + c[i][1] + random.uniform(-noise,noise)

        dx = newx - state[-2]
        dy = newy - state[-1]

        s_traj[i].append(state)
        a_traj[i].append([dx, dy])
        one_hot_code = np.zeros(len(r))
        one_hot_code[i] = 1
        c_traj[i].append(one_hot_code)

        state, _ = env.step([dx,dy], mode=False)
    
    s_traj[i] = np.array(s_traj[i], dtype=np.float32)
    a_traj[i] = np.array(a_traj[i], dtype=np.float32)
    c_traj[i] = np.array(c_traj[i], dtype=int)
    code_prob.append(n[i] / total_points)

pkl.dump((s_traj, a_traj, c_traj, code_prob), open("./expert_traj.pkl", "wb"))