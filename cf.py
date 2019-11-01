import os
import json
import numpy as np
from astropy.io import fits
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import fftconvolve
from scipy.stats import multivariate_normal
from skimage.feature import blob_dog
from multiprocessing import Pool
from argparse import ArgumentParser
from plot import *


# load hyperparameters from file
def load_hyperparameters(params_file: str, printout: bool = False):
	with open(params_file, 'r') as params:
		hp = json.load(params)
		global c
		c = hp['c']  # km/s
		global H0
		H0 = hp['H0']
		global c_over_H0
		c_over_H0 = c / H0
		global h_
		h_ = H0 / 100.
		global Omega_M
		Omega_M = hp['Omega_M']
		global Omega_L
		Omega_L = 1 - Omega_M
		global grid_spacing
		grid_spacing = hp['grid_spacing']  # Mpc/h
		if printout:
			print(f"Hyperparameters loaded successfully from '{params_file}'...")


def load_data(filename: str) -> (np.array, np.array, np.array):
	catalog_data = fits.open(filename)[1].data
	ra = catalog_data['ra']
	dec = catalog_data['dec']
	redshift = catalog_data['z']
	return ra, dec, redshift


# this calculates the transformation from observed redshift to radial distance
def z2r(z: float) -> float:
	return c_over_H0 * integrate.quad(lambda zeta: (Omega_M * (1 + zeta)**3 + Omega_L)**-0.5, 0, z)[0]


def sky2cartesian(ra: np.array, dec: np.array, redshift: np.array) -> np.array:
	# TODO: automate getting the number of spacings instead of setting it to 100
	# this stores quick-lookup tick values for the given range of redshifts
	redshift_ticks = np.linspace(redshift.min(), redshift.max(), 100)
	radii_ticks = np.array([z2r(z) for z in redshift_ticks])

	# this creates a globally accessible lookup table function with interpolated radii values
	global LUT_radii
	LUT_radii = InterpolatedUnivariateSpline(redshift_ticks, radii_ticks)
	global LUT_redshifts
	LUT_redshifts = InterpolatedUnivariateSpline(radii_ticks, redshift_ticks)

	radii = np.array([LUT_radii(z) for z in redshift])
	xs = radii * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
	ys = radii * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
	zs = radii * np.sin(np.deg2rad(dec))
	return np.array([xs, ys, zs])


def cartesian2sky(xs: np.array, ys: np.array, zs: np.array) -> (np.array, np.array, np.array, np.array):
	radii = (xs ** 2 + ys ** 2 + zs ** 2) ** 0.5
	redshift = np.array(LUT_redshifts(radii))
	dec = np.rad2deg(np.arcsin(zs / radii))
	ra = np.rad2deg(np.arctan(ys / xs))
	return ra, dec, redshift, radii


def kernel(bao_radius: float, grid_spacing: float, additional_thickness: float = 0., show_kernel: bool = False, printout: bool = False) -> np.ndarray:
	# this is the number of bins in each dimension axis
	# this calculation ensures an odd numbered gridding
	# the kernel construction has a distinct central bin on any given run
	kernel_bin_count = int(2 * np.ceil(bao_radius / grid_spacing) + 1)

	# this is the kernel inscribed radius in index units
	inscribed_r_idx_units = bao_radius / grid_spacing
	inscribed_r_idx_units_upper_bound = inscribed_r_idx_units + 0.5 + additional_thickness
	inscribed_r_idx_units_lower_bound = inscribed_r_idx_units - 0.5 - additional_thickness

	# central bin index, since the kernel is a cube this can just be one int
	kernel_center_index = int(kernel_bin_count / 2)
	kernel_center = np.array([kernel_center_index, ] * 3)

	# this is where the magic happens: each bin at a radial distance of bao_radius from the
	# kernel's center gets assigned a 1 and all other bins get a 0
	kernel_grid = np.array([[[1 if (np.linalg.norm(np.array([i, j, k]) - kernel_center) >= inscribed_r_idx_units_lower_bound
									and np.linalg.norm(np.array([i, j, k]) - kernel_center) < inscribed_r_idx_units_upper_bound)
							  else 0
							  for i in range(kernel_bin_count)]
							 for j in range(kernel_bin_count)]
							for k in range(kernel_bin_count)])

	if printout:
		print('Kernel constructed successfully...')
		print('Number of kernel bins containing spherical surface:', len(kernel_grid[kernel_grid == 1]))
		print('Number of empty kernel bins:', len(kernel_grid[kernel_grid == 0]))

	# this is here for future sanity checks, it shows the kernel in 3d
	# with blue disks in kernel bins containing spherical surface
	if show_kernel:
		color = 'cornflowerblue'
		fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
		ax.scatter(*np.where(kernel_grid == 1), c=color)
		plt.show()

	return kernel_grid


def vote(filename: str, radius: float, save: bool = False, savename: str = 'saves/saved', plot: bool = False, printout: bool = False) -> (np.ndarray, list, np.ndarray):
	# gets sky data and transforms them to cartesian
	ra, dec, redshift = load_data(filename)
	xyzs = sky2cartesian(ra, dec, redshift)

	# gets the 3d histogram (density_grid) and the grid bin coordintes in cartesian (grid_edges)
	galaxies_cartesian_coords = xyzs.T  # each galaxy is represented by (x, y, z)
	bin_counts_3d = np.array([np.ceil((xyzs[i].max() - xyzs[i].min()) / grid_spacing) for i in range(len(xyzs))], dtype=int)
	density_grid, observed_grid_edges = np.histogramdd(galaxies_cartesian_coords, bins=bin_counts_3d)
	if printout:
		print('Histogramming completed successfully...')
		print('Density grid shape:', density_grid.shape)

	# subtracts the background
	background, _ = project_and_sample(density_grid, observed_grid_edges, printout=printout)
	# TODO: we have negative values here after the subtraction, check that this is okay
	density_grid -= background
	density_grid[density_grid < 0.] = 0.

	# gets kernel
	kernel_grid = kernel(radius, grid_spacing)

	# this scans the kernel over the whole volume of the galaxy density grid
	# calculates the tensor inner product of the two at each step
	# and finally stores this value as the number of voters per that bin in the observed grid
	observed_grid = np.round(fftconvolve(density_grid, kernel_grid, mode='same'))
	if printout:
		print('Observed grid shape:', observed_grid.shape)
		print('Maximum number of voters per single bin:', observed_grid.max())
		print('Minimum number of voters per single bin:', observed_grid.min())

	if save:
		try:
			os.mkdir('saves')
		except FileExistsError:
			pass
		np.save(savename + "_obs_grid.npy", observed_grid)

	if plot:
		vote_threshold_ = 50
		voted_centers_coords = (observed_grid_edges[i][np.where(observed_grid >= vote_threshold_)[i]] for i in range(len(observed_grid_edges)))
		plot_grid_with_true_centers(voted_centers_coords, galaxies_cartesian_coords, 78, vote_threshold=vote_threshold_, savename=filename.split('.')[0] + '.png')

	return observed_grid, observed_grid_edges, galaxies_cartesian_coords


# TODO: figure out a faster way to do the projection.
def alpha_delta_r_projections_from_observed_grid(observed_grid: np.ndarray, N_bins_x: int, N_bins_y: int, N_bins_z: int, sky_coords_grid: np.ndarray, N_bins_alpha: int, N_bins_delta: int, N_bins_r: int) -> (np.ndarray, np.ndarray):
	alpha_delta_grid = np.zeros((N_bins_alpha, N_bins_delta))
	r_grid = np.zeros((N_bins_r,))
	for i in range(N_bins_x):
		for j in range(N_bins_y):
			for k in range(N_bins_z):
				alpha_delta_grid[sky_coords_grid[i, j, k, 0], sky_coords_grid[i, j, k, 1]] += observed_grid[i, j, k]
				r_grid[sky_coords_grid[i, j, k, 2]] += observed_grid[i, j, k]
	return alpha_delta_grid, r_grid


# calculates angular volume adjustment ratio needed for correcting the expected grid
def volume_adjustment(bin_centers_radii: np.array, bin_centers_ra: np.array, bin_centers_dec: np.array, observed_grid_shape: tuple, printout: bool = False) -> np.ndarray:
	# radius
	mid_r = (bin_centers_radii.max() + bin_centers_radii.min()) / 2
	delta_r = bin_centers_radii.max() - bin_centers_radii.min()
	N_bins_r = int(np.ceil(delta_r / grid_spacing))
	d_r = grid_spacing
	r_sqr = bin_centers_radii ** 2

	# alpha
	delta_alpha = np.deg2rad(bin_centers_ra.max() - bin_centers_ra.min())
	N_bins_alpha = int(np.ceil((delta_alpha * mid_r / 2) / grid_spacing))
	d_alpha = delta_alpha / N_bins_alpha

	# delta
	delta_delta = np.deg2rad(bin_centers_dec.max() - bin_centers_dec.min())
	N_bins_delta = int(np.ceil((delta_delta * mid_r / 2) / grid_spacing))
	d_delta = delta_delta / N_bins_delta
	cos_delta = np.cos(np.deg2rad(bin_centers_dec))

	# angular volume differential
	dV_ang = d_alpha * cos_delta * d_delta * r_sqr * d_r
	# euclidean volume differential
	dV_xyz = grid_spacing ** 3
	# volume adjustment ratio grid; contains the volume adjustment ratio per each bin in the expected grid
	vol_adjust_ratio_grid = (dV_xyz / dV_ang).reshape(observed_grid_shape)

	if printout:
		print('Number of bins in r:', N_bins_r)
		print('Number of bins in alpha:', N_bins_alpha)
		print('Number of bins in delta:', N_bins_delta)
		print('Volume adjustment ratio grid shape:', vol_adjust_ratio_grid.shape)

	return vol_adjust_ratio_grid, d_r, d_alpha, d_delta, N_bins_r, N_bins_alpha, N_bins_delta


def project_and_sample(observed_grid: np.ndarray, observed_grid_edges: list, refined_grid_spacing: int = None, save: bool = False, savename: str = 'saves/saved', printout: bool = False) -> (np.ndarray, tuple):
	bin_centers_edges_xs, bin_centers_edges_ys, bin_centers_edges_zs = np.array([(observed_grid_edges[i][:-1] + observed_grid_edges[i][1:]) / 2 for i in range(len(observed_grid_edges))])

	if save:
		np.save(savename + '_xbins.npy', bin_centers_edges_xs)
		np.save(savename + '_ybins.npy', bin_centers_edges_ys)
		np.save(savename + '_zbins.npy', bin_centers_edges_zs)

	bin_centers_xs, bin_centers_ys, bin_centers_zs = np.array([(x, y, z) for x in bin_centers_edges_xs for y in bin_centers_edges_ys for z in bin_centers_edges_zs]).T
	if printout:
		print('Number of bin centers in cartesian coordinates:', len(bin_centers_xs))
	"""
	Why can we be sure that it is okay to interpolate the radii and redshift values for these bin centers coordinates?
	Because we know that the range of values of the bin centers is exactly in between the min and the max of the grid bin edges x, y, z.
	The radii come from the 3d euclidian distance, which preserves this relationship (convex function of x,y,z), and thus it is fine
	to use the beforehand-calculated interpolation lookup table to find the redshifts from the radii.
	"""
	bin_centers_ra, bin_centers_dec, bin_centers_redshift, bin_centers_radii = cartesian2sky(bin_centers_xs, bin_centers_ys, bin_centers_zs)
	if printout:
		print('Number of bin centers in sky coordinates:', len(bin_centers_ra))

	# resetting the global grid_spacing to the refined_grid_spacing for use by the refined grid procedure
	if refined_grid_spacing is not None:
		global grid_spacing
		grid_spacing = refined_grid_spacing

	# total number of votes
	N_tot = np.sum(observed_grid)
	if printout:
		print('Total number of votes:', N_tot)

	# get volume adjustment grid, the differentials in sky coordinate dimensions and the number of bins in each dimension
	vol_adjust_ratio_grid, d_r, d_alpha, d_delta, N_bins_r, N_bins_alpha, N_bins_delta = volume_adjustment(bin_centers_radii, bin_centers_ra, bin_centers_dec, observed_grid.shape, printout=printout)

	# alpha-delta and z counts
	N_bins_x, N_bins_y, N_bins_z = observed_grid.shape[0], observed_grid.shape[1], observed_grid.shape[2]
	sky_coords_grid_shape = (N_bins_x, N_bins_y, N_bins_z, 3)  # need to store a triple at each grid bin
	sky_coords_grid = np.array(list(zip(bin_centers_ra, bin_centers_dec, bin_centers_radii))).reshape(sky_coords_grid_shape)
	if printout:
		print('Shape of grid containing sky coordinates of observed grid\'s bin centers:', sky_coords_grid.shape)

	# getting some variables ready for the projection step
	alpha_min = bin_centers_ra.min()
	d_alpha = np.rad2deg(d_alpha)
	delta_min = bin_centers_dec.min()
	d_delta = np.rad2deg(d_delta)
	r_min = bin_centers_radii.min()

	# vectorial computation of the sky indices
	sky_coords_grid[:, :, :, 0] = (sky_coords_grid[:, :, :, 0] - alpha_min) // d_alpha
	sky_coords_grid[:, :, :, 1] = (sky_coords_grid[:, :, :, 1] - delta_min) // d_delta
	sky_coords_grid[:, :, :, 2] = (sky_coords_grid[:, :, :, 2] - r_min) // d_r
	sky_coords_grid = sky_coords_grid.astype(int)

	# the following fixes any indices that lie beyond the outer walls of the sky grid by pulling them 1 index unit back
	sky_coords_grid[:, :, :, 0][sky_coords_grid[:, :, :, 0] == N_bins_alpha] = N_bins_alpha - 1
	sky_coords_grid[:, :, :, 1][sky_coords_grid[:, :, :, 1] == N_bins_delta] = N_bins_delta - 1
	sky_coords_grid[:, :, :, 2][sky_coords_grid[:, :, :, 2] == N_bins_r] = N_bins_r - 1

	alpha_delta_grid, r_grid = alpha_delta_r_projections_from_observed_grid(observed_grid, N_bins_x, N_bins_y, N_bins_z, sky_coords_grid, N_bins_alpha, N_bins_delta, N_bins_r)
	if printout:
		print('Shape of alpha-delta grid:', alpha_delta_grid.shape)
		print('Shape of r grid:', r_grid.shape)
		print('Maximum number of voters per single bin in alpha-delta grid:', alpha_delta_grid.max())
		print('Minimum number of voters per single bin in alpha-delta grid:', alpha_delta_grid.min())
		print('Maximum number of voters per single bin in r grid:', r_grid.max())
		print('Minimum number of voters per single bin in r grid:', r_grid.min())
		print('N_tot_observed = N_tot_alpha_delta = N_tot_r:', N_tot == np.sum(alpha_delta_grid) == np.sum(r_grid))

	expected_grid = np.array([[[alpha_delta_grid[sky_coords_grid[i, j, k, 0], sky_coords_grid[i, j, k, 1]] * r_grid[sky_coords_grid[i, j, k, 2]]
								for k in range(N_bins_z)]
							   for j in range(N_bins_y)]
							  for i in range(N_bins_x)])

	expected_grid /= N_tot  # normalization
	expected_grid *= vol_adjust_ratio_grid  # volume adjustment
	if printout:
		print('Expected grid shape:', expected_grid.shape)
		print('Maximum number of expected votes:', expected_grid.max())
		print('Minimum number of expected votes:', expected_grid.min())

	if save:
		np.save(savename + "_exp_grid.npy", expected_grid)

	return expected_grid, (bin_centers_edges_xs, bin_centers_edges_ys, bin_centers_edges_zs)


# TODO: the expected grid and sig grid threshold should be dealt with
def significance(observed_grid: np.ndarray, expected_grid: np.ndarray, expected_votes_threshold: float = 15., significance_threshold: float = 4., save: bool = False, savename: str = 'saves/saved', printout: bool = False) -> np.ndarray:
	expected_grid[expected_grid < expected_votes_threshold] = expected_votes_threshold  # resolves division by zero error
	sig_grid = (observed_grid - expected_grid) / np.sqrt(expected_grid)
	sig_grid[sig_grid < significance_threshold] = 0.
	# this prepares the grid for the blobbing procedure, which requires a bright on dark image in grayscale, the values of the sig grid now run from 0 to 255
	if sig_grid.max() != 0.:
		sig_grid /= sig_grid.max()
	sig_grid *= 255.
	if printout:
		print('Maximum significance:', sig_grid.max())
		print('Minimum significance:', sig_grid.min())
	if save:
		np.save(savename + '_sig_grid.npy', sig_grid)
	return sig_grid


def blob(grid: np.ndarray, bin_centers_xyzs: tuple, galaxies_cartesian_coords: np.ndarray, min_sigma_: float = 2., max_sigma_: float = 30., overlap_: float = 0.001, save: bool = False, savename: str = 'saves/saved', plot: bool = False, printout: bool = False) -> (np.ndarray, np.ndarray):
	blob_grid_indices = blob_dog(grid, min_sigma=min_sigma_, max_sigma=max_sigma_, overlap=overlap_)
	blob_centers_xyzs_indices = np.array(blob_grid_indices.T, dtype=int)
	blob_centers_xs_index, blob_centers_ys_index, blob_centers_zs_index = blob_centers_xyzs_indices[0], blob_centers_xyzs_indices[1], blob_centers_xyzs_indices[2]
	bin_centers_xs, bin_centers_ys, bin_centers_zs = bin_centers_xyzs[0], bin_centers_xyzs[1], bin_centers_xyzs[2]
	blob_centers_xs, blob_centers_ys, blob_centers_zs = bin_centers_xs[blob_centers_xs_index], bin_centers_ys[blob_centers_ys_index], bin_centers_zs[blob_centers_zs_index]
	blob_centers_xyzs = blob_centers_xs, blob_centers_ys, blob_centers_zs
	if printout:
		print('Number of blobs found after blobbing:', len(blob_centers_xs))
	if plot:
		plot_grid_with_true_centers(blob_centers_xyzs, galaxies_cartesian_coords, 78, showplot=True)
	if save:
		np.save(savename + '_blob_grid_indices.npy', blob_grid_indices)
		np.save(savename + '_blob_centers_xyzs.npy', blob_centers_xyzs)
	return np.array(blob_grid_indices, dtype=int), np.array(blob_centers_xyzs).T


def refine(blob_x, blob_y, blob_z):
	# voting procedure
	x_bound_lower, x_bound_upper = blob_x - half_grid_length, blob_x + half_grid_length
	y_bound_lower, y_bound_upper = blob_y - half_grid_length, blob_y + half_grid_length
	z_bound_lower, z_bound_upper = blob_z - half_grid_length, blob_z + half_grid_length
	range_ = ((x_bound_lower, x_bound_upper), (y_bound_lower, y_bound_upper), (z_bound_lower, z_bound_upper))
	finer_density_grid, finer_observed_grid_edges = np.histogramdd(galaxies_cartesian_coords, range=range_, bins=N_bins)
	finer_observed_grid = np.round(fftconvolve(finer_density_grid, finer_kernel_grid, mode='same'))

	# projection and sampling procedure
	finer_expected_grid, finer_bin_centers_edges = project_and_sample(finer_observed_grid, finer_observed_grid_edges, refined_grid_spacing=finer_grid_spacing)

	# significance grid and blobbing procedure
	finer_significance_grid = significance(finer_observed_grid, finer_expected_grid)
	if printout:
		print('Finer sig grid max:', finer_significance_grid.max(), 'at position:', np.unravel_index(finer_significance_grid.argmax(), finer_significance_grid.shape))

	# this takes a smaller portion of the significance grid, just a box of side-length 1/3 that of sig grid, centered at the center of the refined sig grid
	sig_grid_x, sig_grid_y, sig_grid_z = finer_significance_grid.shape[0], finer_significance_grid.shape[1], finer_significance_grid.shape[2]
	finer_significance_grid = finer_significance_grid[int(1. / 3. * sig_grid_x):int(2. / 3. * sig_grid_x), int(1. / 3. * sig_grid_y):int(2. / 3. * sig_grid_y), int(1. / 3. * sig_grid_z):int(2. / 3. * sig_grid_z)]
	if printout:
		print('finer_significance_grid max:', finer_significance_grid.max())
		print('finer_significance_grid shape:', finer_significance_grid.shape)
	# this does the same as above but for the finer bin edges
	bin_edges_x, bin_edges_y, bin_edges_z = len(finer_bin_centers_edges[0]), len(finer_bin_centers_edges[1]), len(finer_bin_centers_edges[2])
	finer_bin_centers_edges = (finer_bin_centers_edges[0][int(1. / 3. * bin_edges_x):int(2. / 3. * bin_edges_x)], finer_bin_centers_edges[1][int(1. / 3. * bin_edges_y):int(2. / 3. * bin_edges_y)], finer_bin_centers_edges[2][int(1. / 3. * bin_edges_z):int(2. / 3. * bin_edges_z)])
	finer_blob_grid_indices, finer_blob_cartesian_coords = blob(finer_significance_grid, finer_bin_centers_edges, galaxies_cartesian_coords)

	# grab highest significance blob if there are any blobs
	if len(finer_blob_grid_indices) > 0:
		max_significance_blob = np.array([finer_significance_grid[blob[0], blob[1], blob[2]] for blob in finer_blob_grid_indices])
		max_significance_blob_coords = finer_blob_cartesian_coords[np.argmax(max_significance_blob)]
		if printout:
			print('Maximum significance blob index:', max_significance_blob.argmax())
		return max_significance_blob_coords
	return None


def setup_parallel_env(galaxies_coords, half_grid_length_, N_bins_, finer_kernel_, finer_grid_spacing_, printout_):
	global galaxies_cartesian_coords
	galaxies_cartesian_coords = galaxies_coords
	global half_grid_length
	half_grid_length = half_grid_length_
	global N_bins
	N_bins = N_bins_
	global finer_kernel_grid
	finer_kernel_grid = finer_kernel_
	global finer_grid_spacing
	finer_grid_spacing = finer_grid_spacing_
	global printout
	printout = printout_


def parallel_refine(blob_cartesian_coords, galaxies_cartesian_coords, radius, padding=5, finer_grid_spacing=2, save: bool = False, printout: bool = False):
	half_grid_length = radius + padding
	N_bins = int(np.ceil(2 * half_grid_length / finer_grid_spacing))
	# TODO: There's additional thickness here but not during kernelization, flatten this difference?
	finer_kernel_grid = kernel(radius, finer_grid_spacing, additional_thickness=1)

	# default number of processes opened in pool is max number of system cores, i.e. os.cpu_count()
	# the initializer and ititargs globalize the input data such that the parallelized children processes can access them during mapping of refine
	with Pool(initializer=setup_parallel_env, initargs=(galaxies_cartesian_coords, half_grid_length, N_bins, finer_kernel_grid, finer_grid_spacing, printout)) as pool:
		result = pool.starmap(refine, blob_cartesian_coords)  # the starmap just unpacks each iteration of the iterable blob_cartesian_coords

	finer_coords = np.array([coords for coords in result if coords is not None]).T
	if save:
		np.save(savename + '_finer_blob_coords.npy', finer_coords)
		# np.save(savename + '_blob_displacements.npy', blob_displacements)
	# plot_grid_with_true_centers(finer_coords, galaxies_cartesian_coords, 79, showplot=True)
	return finer_coords


def radial_scan(r_min: float, r_max: float, step: float, file_: str, refine: bool = False, save__: bool = False, savename__: str = 'saves/saved', printout__: bool = False):
	for r in np.arange(r_min, r_max, step):
		observed_grid, observed_grid_edges, galaxies_cartesian_coords = vote(file_, r, save=save__, savename=savename__, printout=printout__)
		expected_grid, bin_centers_edges = project_and_sample(observed_grid, observed_grid_edges, save=save__, savename=savename__, printout=printout__)
		significance_grid = significance(observed_grid, expected_grid, expected_votes_threshold=8., save=save__, savename=savename__, printout=printout__)
		blob_grid_indices, blob_cartesian_coords = blob(significance_grid, bin_centers_edges, galaxies_cartesian_coords, save=save__, savename=savename__, printout=printout__)
		
		if (refine):
			finer_coords = parallel_refine(blob_cartesian_coords, galaxies_cartesian_coords, r, save=save__, printout=printout__)
			# TODO: output is just a printout of the number of centers found now, may change to saved output later
			print(r, finer_coords.shape[1])



def main():
	parser = ArgumentParser(description="( * ) Center Finder ( * )")
	parser.add_argument('file', metavar='DATA_FILE', type=str, help='Name of fits file to be fitted.')
	parser.add_argument('-k', '--kernelization', type=int, default=None, help='If this argument is present, kernelization will occur at radius entered as argument.')
	parser.add_argument('-r', '--refinement', action='store_true', help='If this argument is present along with \'-k\', after kernelization, the refinement procedure will occur.')
	parser.add_argument('-s', '--radial_scan', action='store_true', help='If this argument is present, center finder will be scanned over a given interval in BAO radius.')
	parser.add_argument('-p', '--params_file', type=str, default='params.json', help='If this argument is present, the cosmological parameters will be uploaded from given file instead of the default.')
	parser.add_argument('-v', '--save', action='store_true', help='If this argument is present, the x, y and z bin centers will be saved in the "saves" folder along with the observed, expected and significance grids.')
	parser.add_argument('-o', '--printout', action='store_true', help='If this argument is present, the progress of center-finder will be printed out to standard output.')
	args = parser.parse_args()

	savename_ = 'saves/' + args.file.split('.')[0]
	load_hyperparameters(args.params_file, printout=args.printout)

	if (args.kernelization is not None):
		bao_radius = args.kernelization
		observed_grid, observed_grid_edges, galaxies_cartesian_coords = vote(args.file, bao_radius, save=args.save, savename=savename_, printout=args.printout)
		expected_grid, bin_centers_edges = project_and_sample(observed_grid, observed_grid_edges, save=args.save, savename=savename_, printout=args.printout)
		significance_grid = significance(observed_grid, expected_grid, save=args.save, savename=savename_, printout=args.printout)
		blob_grid_indices, blob_cartesian_coords = blob(significance_grid, bin_centers_edges, galaxies_cartesian_coords, save=args.save, savename=savename_, printout=args.printout)
		if (args.refinement):
			parallel_refine(blob_cartesian_coords, galaxies_cartesian_coords, bao_radius, save=args.save, printout=args.printout)

	if args.radial_scan:
		if args.refinement:
			radial_scan(90.5, 120.5, 2.5, args.file, refine=True, save__=args.save, savename__=savename_, printout__=args.printout)
		else:
			radial_scan(90.5, 120.5, 2.5, args.file, save__=args.save, savename__=savename_, printout__=args.printout)


if __name__ == '__main__':
	main()
