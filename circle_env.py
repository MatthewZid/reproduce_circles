import math
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

class CircleEnv():
	#-------------------------
	# Constructor
	#-------------------------
	def __init__(self, speed=0.02, sigma1=0.01, sigma2=0.005, max_step=512):
		self.state    = np.zeros((5, 2), dtype=np.float32)
		self.max_step = max_step
		self.n_step   = 1
		self.sigma1   = sigma1
		self.sigma2   = sigma2
		self.color    = [np.array([1.0, 0.0, 0.0])]
		self.p        = [0, 0]
		self.xs       = []
		self.ys       = []
		self.speed    = speed

		self.fig = plt.figure()
		self.ax  = self.fig.add_subplot(111)
		# mng = plt.get_current_fig_manager()
		# mng.resize(*mng.window.maxsize())
	

	#-------------------------
	# Step
	#-------------------------
	def step(self, action):
		norm = math.sqrt(action[0]*action[0] + action[1]*action[1])

		if norm > 1e-8:
			action[0] /= norm
			action[1] /= norm
		else:
			action[0] = 1.0
			action[1] = 0.0

		self.p[0] += action[0] * self.speed + self.sigma2*np.random.randn()
		self.p[1] += action[1] * self.speed + self.sigma2*np.random.randn()
		self.xs.append(self.p[0])
		self.ys.append(self.p[1])
		self.n_step += 1

		for i in range(4):
			self.state[i, :] = self.state[i+1, :]
		self.state[4, :] = np.array(self.p)

		reward = 0.0

		# if self.n_step >= 129 or abs(self.p[0]) >= 1 or abs(self.p[1]) >= 1:
		if self.n_step >= 129:
			done = True
		else:
			done = False

		return np.copy(self.state.flatten()), done


	#-------------------------
	# Reset
	#-------------------------
	def reset(self, start=(0.0,0.0)):
		# self.p[0]   = self.sigma1 * np.random.randn()
		# self.p[1]   = self.sigma1 * np.random.randn()
		self.p[0]   = start[0]
		self.p[1]   = start[1]
		self.xs     = [self.p[0]]
		self.ys     = [self.p[1]]
		self.n_step = 1

		for i in range(5):
			self.state[i, 0] = self.p[0]
			self.state[i, 1] = self.p[1]

		return np.copy(self.state.flatten())


	#-------------------------
	# Render
	#-------------------------
	def render(self):
		self.ax.cla()
		self.ax.set_aspect(aspect=1.0)
		self.ax.scatter(self.xs, self.ys, s=4, c=self.color)
		plt.grid()
		plt.xlim(-1, 1)
		plt.ylim(-1, 1)
		plt.pause(0.002)


	#-------------------------
	# Get all points
	#-------------------------
	def get_points(self):
		return self.xs, self.ys


	#-------------------------
	# Close the environment
	#-------------------------
	def close(self):
		pass


#-------------------------
# Make an environment
#-------------------------
def make(max_step=512):
	return CircleEnv(max_step=max_step)