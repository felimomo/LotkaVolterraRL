import numpy as np
from itertools import product
from pathlib import Path

RNG = np.random.rng()

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
	r = (1 + r_sigma * RNG.normal())
	return np.row_stack(r, M)

def main():
	sampler = sample_gauss_lv
	M_sigma=1
	r_sigma=1
	savepath = (
		Path(__file__)
			.parent # sampling
			.parent # LotkaVolterraRL
			.parent # src
			.parent # repo main dir
		/ 'samples' 
	)
	#
	for N in [5, 10, 15, 20, 25]:
		for version in range(10):
			save_id = f'N-{N:02d}-v{version:02d}'
			r_and_M = sampler(N=N, M_sigma=M_sigma, r_sigma=r_sigma)
			np.save(
				savepath / save_id,
				r_and_M
			)

if __name__ == "__main__":
	main()