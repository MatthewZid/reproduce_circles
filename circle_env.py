import math
import numpy as np
import random

class CircleEnv():
	#-------------------------
	# Constructor
	#-------------------------
	def __init__(self, sigma1=0.01, max_step=512, rand=False, starting_point=(0,-0.25)):
		self.state    = np.zeros((5, 2), dtype=np.float32)
		self.max_step = max_step
		self.rand = rand
		self.start = starting_point
		self.n_step   = 1
		self.sigma1   = sigma1
		self.p        = [0, 0]
		self.xs       = []
		self.ys       = []

	#-------------------------
	# Step
	#-------------------------
	def step(self, action, mode=True):
		# norm = math.sqrt(action[0]*action[0] + action[1]*action[1])

		# if mode:
		# 	if norm > 1e-8:
		# 		action[0] /= norm
		# 		action[1] /= norm
		# 	else:
		# 		action[0] = 1.0
		# 		action[1] = 0.0

		self.p[0] += action[0]
		self.p[1] += action[1]
		self.xs.append(self.p[0])
		self.ys.append(self.p[1])
		self.n_step += 1

		for i in range(4):
			self.state[i, :] = self.state[i+1, :]
		self.state[4, :] = np.array(self.p)

		done = False
		if mode:
			# if self.n_step >= self.max_step or abs(self.p[0]) >= 1 or abs(self.p[1]) >= 1:
			if self.n_step >= self.max_step:
				done = True

		return np.copy(self.state.flatten()), done


	#-------------------------
	# Reset
	#-------------------------
	def reset(self):
		if self.rand:
			self.p[0]   = self.sigma1 * np.random.randn()
			self.p[1]   = self.sigma1 * np.random.randn()
		else:
			self.p[0] = self.start[0]
			self.p[1] = self.start[1]
		self.xs     = [self.p[0]]
		self.ys     = [self.p[1]]
		self.n_step = 1

		for i in range(5):
			self.state[i, 0] = self.p[0]
			self.state[i, 1] = self.p[1]

		return np.copy(self.state.flatten())