import numpy as np
from itertools import product
from pathlib import Path

RNG = np.random.default_rng()

def sample_gauss_lv(N, M_sigma = 1, r_sigma = 1):
	# M = np.zeroes(size=(N,N))
	M = np.array(
		[
			[
				0 if i==j
				else M_sigma * RNG.normal()
				for i in range(N)
			]
			for j in range(N)
		]
	)
	M = (M - M.T) / 2
	r = (1 + r_sigma * RNG.normal(size=N))
	return np.row_stack((r, M))

def main():
	sampler = sample_gauss_lv
	M_sigma=1
	r_sigma=1
	savepath = (
		Path(__file__).resolve()
			.parent # sampling
			.parent # LotkaVolterraRL
			.parent # src
			.parent # repo main dir
		/ 'samples' 
	).resolve() # generate absolute path
	#
	for N in [5, 10, 15, 20, 25]:
		for version in range(10):
			save_id = f'N-{N:02d}-v{version:02d}.npy'
			r_and_M = sampler(N=N, M_sigma=M_sigma, r_sigma=r_sigma)
			with open(savepath / save_id, 'wb') as file:
				np.save(
					file,
					r_and_M
				)

if __name__ == "__main__":
	main()