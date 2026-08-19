import gymnasium as gym
import numpy as np

class LotkaVolterraEnv(gym.Env):
	def __init__(self, parameters):
		self.M = parameters.get('M')
		self.r = parameters.get('r')
		self.N = len(self.r)
		self.K = parameters
		self.sigma = parameters.get('sigma')
		#
		self.fished = parameters.get('fished', [0])
		self.observed = parameters.get('observed', [0])
		self.obs_sigma = parameters.get('obs_sigma', 0.1)
		self.harv_sigma = parameters.get('harvest_sigma', 0.1)
		self.harv_exp = parameters.get('harvest_exp', 1)
		#
		self.bound = parameters.get('bound', 1)
		self.init_pop = parameters.get('init_pop', np.ones(self.N))
		self.init_sig = parameters.get('init_sig', 0.3)
		self.maxT = paramters.get('maxT', 100)
		#
		self.action_space = gym.spaces.Box(
			[-1] * len(self.fished),
			[ 1] * len(self.fished)
		)
		self.observation_space = gym.spaces.Box(
			[-1] * len(self.observed),
			[ 1] * len(self.observed)
		)
	#
	def reset(self, *, , seed=42, options=None):
		self.rng = np.random.default_rng()
		self.pop = self.init_pop * (
			1 + self.init_sig * self.rng.normal()
		)
		self.t=0
		return self.get_obs(self.pop), {}
	
	def step(self, action):
		self.pop = self.nat_dyn_step(self.pop)
		reward = self.reward(
			self.get_harvest(action, self.pop)
		)
		self.pop[self.fished] -= harvest
		#
		self.t += 1
		done = self.t >= self.maxT
		#
		return self.get_obs(self.pop), reward, done, False, {}
	
	def nat_dyn_step(self, curr_pop: np.ndarray):
		return (
			curr_pop + 
			(
				self.r * curr_pop + 
				(
					self.M @ curr_pop * curr_pop
				) 
			) *
			(
				1 + self.sigma * self.rng.normal(size=self.n)
			)
		)

	def get_obs(self, pop: np.ndarray):
		return np.clip(
			(1 + self.obs_sigma * self.rng.normal()) * 
			(pop[self.observed] + 1) / 
			(2 * self.bound),
			-1,
			1,
		)

	def get_harvest(self, action: np.ndarray, pop: np.ndarray):
		harv_mortality = np.clip(
			(1 + self.harv_sigma * self.rng.normal()) *
			(action + 1) / 2,
			0,
			1,
		)
		return harv_mortality * pop[self.fished]

	def reward(self, harvest):
		return np.sum(harvest ** self.harvest_exp)
