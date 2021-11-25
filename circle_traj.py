import numpy as np
import math
import random
import pickle as pkl
from tqdm import tqdm

def genCircle(start, n=100, r=1, c=(0,0), noise=0.02):
    states = [[start[0]], [start[1]]]
    actions = [[],[]]
    up = 1

    theta = 360.0 / n
    
    if states[1][0] < c[1]: up = -1

    angle = theta
    current_state = start
    for a in tqdm(np.arange(angle, 360.0, theta)):
        if (a + theta) < 0.00000001:
            print('Very high resolution! Terminating...')
            break
        newx = up * r * math.sin(math.radians(a)) + c[0] + random.uniform(-noise,noise)
        newy = up * r * math.cos(math.radians(a)) + c[1] + random.uniform(-noise,noise)
        states[0].append(newx)
        states[1].append(newy)

        dx = newx - current_state[0]
        dy = newy - current_state[1]
        actions[0].append(dx)
        actions[1].append(dy)
    
    return states, actions

states = []
actions = []
code_prob = []
total_points = 1200 + 700 + 700
s1, a1 = genCircle((0,-0.25), 1200, 0.5, (0,0.25), noise=0.03)
code_prob.append(1200 / total_points)
states.append(s1)
actions.append(a1)

s2, a2 = genCircle((0,-0.25), 700, 0.25, (0,0), noise=0.03)
code_prob.append(700 / total_points)
states.append(s2)
actions.append(a2)

s3, a3 = genCircle((0,-0.25), 700, 0.25, (0,-0.5), noise=0.03)
code_prob.append(700 / total_points)
states.append(s3)
actions.append(a3)

pkl.dump((states, actions, code_prob), open("./circle_traj.pkl", "wb"))