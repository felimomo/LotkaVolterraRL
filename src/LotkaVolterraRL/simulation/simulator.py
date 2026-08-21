import numpy as np
import pandas as pd

from LotkaVolterraRL import LotkaVolterraEnv

DEFAULT_SAVE = (
		Path(__file__).resolve()
			.parent # simulation
			.parent # LotkaVolterraRL
			.parent # src
			.parent # repo main dir
		/ 'sim_data' 
	).resolve() # generate absolute path

class UnfishedSimulator:
	def __init__(self, M, r, **LV_kwargs):
		self.sim_env = LotkaVolterraEnv(
			parameters = {
				'M':M,
				'r':r,
				'harvest_sigma': 0.0 # important for unfished simulations
				**LV_kwargs
			}
		)

	def SimulateTimeseries(self):
		observs = np.empty(len(self.sim_env.observed))
		pops = np.empty(self.sim_env.N)
		times = np.empty(1)
		rewards = np.empty(1) # will be relevant for non-harvest rewards
		#
		action = - np.ones(len(
			self.sim_env.fished
		))

		new_obs, info = self.sim_env.reset()
		observs = np.vstack((observs, new_obs))
		pops = np.vstack((pops, self.sim_env.pop))
		times = np.vstack((times, 0))
		rewards = np.vstack((rewards, 0))
		#
		for t in range(self.sim_env.maxT):
			new_obs, rew, done, terminated, info = self.sim_env.step(action)
			#
			observs = np.vstack((observs, new_obs))
			pops = np.vstack((pops, self.sim_env.pop))
			times = np.vstack((times, t))
			rewards = np.vstack((rewards, rew))
		#
		data = np.hstack(
			(times, observs, pops, rewards)
		)
		data_df = pd.DataFrame(
			data,
			columns = [
				't',
				*[f'obs_{i}' for i in range(len(observs.T))],
				*[f'pop_{i}' for i in range(len(pops.T))],
				'rew'
			]
		)
		return data_df

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-f', '--src-file', 
		help= '.npy file containing an N x (N+1) ' 
					'r as the first row and M stacked '
					'vertically below it.'
	)
	parser.add_argument(
		'-s', '--save-file',
		help='Name of the csv to save the data in.',
		default=DEFAULT_SAVE/'sim_data.csv'
	)
	args = parser.parse_args()
	#
	r_and_M = np.load(args.src_file)
	r = r_and_M[0]
	M = r_and_M[1:]
	Simulator = UnfishedSimulator(M=M, r=r)
	data_df = Simulator.SimulateTimeseries()
	data_df.to_csv(args.save_file)



