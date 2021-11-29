import math
import numpy as np
import pickle as pkl
from circle_env import CircleEnv
import random
from tqdm import tqdm

env = CircleEnv()
s_traj = []
a_traj = []
code_prob = []
r = [0.5, 0.25, 0.25]
total_points = 1200 + 700 + 700
n = (1200,700,700)
c = ((0,0.25), (0,0), (0,-0.5))
noise = 0.03

for i in range(len(r)):
    state = env.reset()
    s_traj.append([])
    a_traj.append([])

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

        state, _ = env.step([dx,dy], mode=False)
    
    s_traj[i] = np.array(s_traj[i], dtype=np.float32)
    a_traj[i] = np.array(a_traj[i], dtype=np.float32)

pkl.dump((s_traj, a_traj), open("./expert_traj.pkl", "wb"))