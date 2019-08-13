import os
import json
import numpy as np
from astropy.io import fits
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import fftconvolve
from argparse import ArgumentParser
from skimage.feature import blob_dog
from plot import *


# load hyperparameters from file
def load_hyperparameters(params_file: str):
    with open(params_file, 'r') as params:
        hp = json.load(params)
        print(f"Hyperparameters loaded successfully from '{params_file}'...")
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
    # TODO: ground this process, i am uncomfortable just throwing this here
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


def kernel(bao_radius: int, grid_spacing: int, additional_thickness=0, show_kernel: bool = False) -> np.ndarray:
    # this is the number of bins in each dimension axis
    kernel_bin_count = int(np.ceil(2 * bao_radius / grid_spacing))

    # this is the kernel inscribed radius in index units
    inscribed_r_idx_units = bao_radius / grid_spacing
    inscribed_r_idx_units_ceil = np.ceil(inscribed_r_idx_units)
    inscribed_r_idx_units_floor = np.floor(inscribed_r_idx_units)

    # central bin index, since the kernel is a cube this can just be one int
    kernel_center_index = int(kernel_bin_count / 2)
    kernel_center = np.array([kernel_center_index, ] * 3)

    # this is where the magic happens: each bin at a radial distance of bao_radius from the
    # kernel's center gets assigned a 1 and all other bins get a 0
    kernel_grid = np.array([[[1 if (np.linalg.norm(np.array([i, j, k]) - kernel_center) >= inscribed_r_idx_units_floor - additional_thickness
                                    and np.linalg.norm(np.array([i, j, k]) - kernel_center) < inscribed_r_idx_units_ceil + additional_thickness)
                              else 0
                              for i in range(kernel_bin_count)]
                             for j in range(kernel_bin_count)]
                            for k in range(kernel_bin_count)])

    # these are just sanity checks
    print('Kernel constructed successfully...')
    print('Number of kernel bins containing spherical surface:', len(kernel_grid[kernel_grid == 1]))
    print('Number of empty kernel bins:', len(kernel_grid[kernel_grid == 0]))

    # this is here for future sanity checks, it shows the kernel in 3d
    # with blue disks on kernel bins containing spherical surface
    if show_kernel:
        color = 'cornflowerblue'
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        ax.scatter(*np.where(kernel_grid == 1), c=color)
        plt.show()

    return kernel_grid


def single_radius_vote(filename: str, radius: int, save: bool = False, savename: str = 'saves/saved', plot: bool = False) -> (np.ndarray, list, np.ndarray):
    # gets sky data and transforms them to cartesian
    ra, dec, redshift = load_data(filename)
    xyzs = sky2cartesian(ra, dec, redshift)  # this returns (xs, ys, zs) as a tuple ready for unpacking

    # gets the 3d histogram (density_grid) and the grid bin coordintes in cartesian (grid_edges)
    galaxies_cartesian_coords = xyzs.T  # each galaxy is represented by (x, y, z)
    bin_counts_3d = np.array([np.ceil((xyzs[i].max() - xyzs[i].min()) / grid_spacing) for i in range(len(xyzs))], dtype=int)
    density_grid, observed_grid_edges = np.histogramdd(galaxies_cartesian_coords, bins=bin_counts_3d)
    print('Histogramming completed successfully...')
    print('Density grid shape:', density_grid.shape)

    # gets kernel
    kernel_grid = kernel(radius, grid_spacing)

    # this scans the kernel over the whole volume of the galaxy density grid
    # calculates the tensor inner product of the two at each step
    # and finally stores this value as the number of voters per that bin in the observed grid
    observed_grid = np.round(fftconvolve(density_grid, kernel_grid, mode='same'))
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
        vote_threshold_ = 60
        voted_centers_coords = (observed_grid_edges[i][np.where(observed_grid >= vote_threshold_)[i]] for i in range(len(observed_grid_edges)))
        plot_grid_with_true_centers(voted_centers_coords, galaxies_cartesian_coords, N_true_centers, vote_threshold=vote_threshold_, savename=filename.split('.')[0] + '.png')

    return observed_grid, observed_grid_edges, galaxies_cartesian_coords


def alpha_delta_r_projections_from_observed(observed_grid: np.ndarray, N_bins_x: int, N_bins_y: int, N_bins_z: int, sky_coords_grid: np.ndarray, N_bins_alpha: int, N_bins_delta: int, N_bins_r: int) -> (np.ndarray, np.ndarray):
    alpha_delta_grid = np.zeros((N_bins_alpha, N_bins_delta))
    r_grid = np.zeros((N_bins_r,))
    # TODO: np.vectorize(); actually the docs say that the method is just for convenience and that the implementation is basically a for loop, hmmm...
    for i in range(N_bins_x):
        for j in range(N_bins_y):
            for k in range(N_bins_z):
                alpha_delta_grid[int(sky_coords_grid[i, j, k, 0]), int(sky_coords_grid[i, j, k, 1])] += observed_grid[i, j, k]
                r_grid[int(sky_coords_grid[i, j, k, 2])] += observed_grid[i, j, k]
    return alpha_delta_grid, r_grid


# TODO: figure out a faster way to do the projection.
# def alpha_delta_r_projections_from_observed(observed_grid, N_bins_x, N_bins_y, N_bins_z, sky_coords_grid, N_bins_alpha, N_bins_delta, N_bins_r):
#     alpha_delta_grid = np.zeros((N_bins_alpha, N_bins_delta))
#     r_grid = np.zeros((N_bins_r,))
#     # TODO: np.vectorize()
#     alpha_delta_grid[sky_coords_grid[:, :, :, 0].astype(int), sky_coords_grid[:, :, :, 1].astype(int)] += observed_grid[:, :, :]
#     r_grid[sky_coords_grid[:, :, :, 2].astype(int)] += observed_grid[:, :, :]
#     return alpha_delta_grid, r_grid


def project_and_sample(observed_grid: np.ndarray, observed_grid_edges: list, refined_grid_spacing: int = None, save: bool = False, savename: str = 'saves/saved') -> (np.ndarray, tuple):
    bin_centers_edges_xs, bin_centers_edges_ys, bin_centers_edges_zs = np.array([(observed_grid_edges[i][:-1] + observed_grid_edges[i][1:]) / 2 for i in range(len(observed_grid_edges))])

    if save:
        np.save(savename + '_xbins.npy', bin_centers_edges_xs)
        np.save(savename + '_ybins.npy', bin_centers_edges_ys)
        np.save(savename + '_zbins.npy', bin_centers_edges_zs)

    bin_centers_xs, bin_centers_ys, bin_centers_zs = np.array([(x, y, z) for x in bin_centers_edges_xs for y in bin_centers_edges_ys for z in bin_centers_edges_zs]).T
    print('Number of bin centers in cartesian coordinates:', len(bin_centers_xs))
    """
	Why can we be sure that it is okay to interpolate the radii and redshift values for these bin centers coordinates?
	Because we know that the range of values of the bin centers is exactly in between the min and the max of the grid bin edges x, y, z.
	The radii come from the 3d euclidian distance, which preserves this relationship (convex function of x,y,z), and thus it is fine
	to use the beforehand-calculated interpolation lookup table to find the redshifts from the radii.
	"""
    bin_centers_ra, bin_centers_dec, bin_centers_redshift, bin_centers_radii = cartesian2sky(bin_centers_xs, bin_centers_ys, bin_centers_zs)

    print('Number of bin centers in sky coordinates:', len(bin_centers_ra))

    # total number of votes
    N_tot = np.sum(observed_grid)
    print('Total number of votes:', N_tot)

    # resetting the grid_spacing to the refined_grid_spacing for use by the refined grid procedure
    if refined_grid_spacing is not None:
        global grid_spacing
        grid_spacing = refined_grid_spacing

    # angular volume adjustment calculations
    # radius
    mid_r = (bin_centers_radii.max() + bin_centers_radii.min()) / 2
    delta_r = bin_centers_radii.max() - bin_centers_radii.min()
    N_bins_r = int(np.ceil(delta_r / grid_spacing))
    d_r = grid_spacing
    r_sqr = bin_centers_radii ** 2
    print('Number of bins in r:', N_bins_r)

    # TODO: do we really need delta_alpha N_bins_alpha and d_alpha? there's a simplification available here.
    # alpha
    delta_alpha = np.deg2rad(bin_centers_ra.max() - bin_centers_ra.min())
    N_bins_alpha = int(np.ceil((delta_alpha * mid_r / 2) / grid_spacing))
    d_alpha = delta_alpha / N_bins_alpha
    print('Number of bins in alpha:', N_bins_alpha)

    # delta
    delta_delta = np.deg2rad(bin_centers_dec.max() - bin_centers_dec.min())
    N_bins_delta = int(np.ceil((delta_delta * mid_r / 2) / grid_spacing))
    d_delta = delta_delta / N_bins_delta
    cos_delta = np.cos(np.deg2rad(bin_centers_dec))
    print('Number of bins in delta:', N_bins_delta)

    # angular volume differential
    dV_ang = d_alpha * cos_delta * d_delta * r_sqr * d_r
    # euclidean volume differential
    dV_xyz = grid_spacing ** 3
    # volume adjustment ratio grid; contains the volume adjustment ratio per each bin in the observed grid
    vol_adjust_ratio_grid = (dV_xyz / dV_ang).reshape(observed_grid.shape)
    print('Volume adjustment ratio grid shape:', vol_adjust_ratio_grid.shape)

    # alpha-delta and z counts
    N_bins_x, N_bins_y, N_bins_z = len(observed_grid), len(observed_grid[0]), len(observed_grid[0, 0])
    sky_coords_grid_shape = (N_bins_x, N_bins_y, N_bins_z, 3)  # need to store a triple at each index
    sky_coords_grid = np.array(list(zip(bin_centers_ra, bin_centers_dec, bin_centers_radii))).reshape(sky_coords_grid_shape)
    print('Shape of grid containing sky coordinates of observed grid\'s bin centers:', sky_coords_grid.shape)

    # getting some variables ready for the projection step
    alpha_min = bin_centers_ra.min()
    d_alpha = np.rad2deg(d_alpha)
    delta_min = bin_centers_dec.min()
    d_delta = np.rad2deg(d_delta)
    r_min = bin_centers_radii.min()

    # vectorial computation of the sky indices
    sky_coords_grid[:, :, :, 0] = (sky_coords_grid[:, :, :, 0] - alpha_min) // d_alpha
    sky_coords_grid[:, :, :, 0][sky_coords_grid[:, :, :, 0] == N_bins_alpha] = N_bins_alpha - 1
    sky_coords_grid[:, :, :, 1] = (sky_coords_grid[:, :, :, 1] - delta_min) // d_delta
    sky_coords_grid[:, :, :, 1][sky_coords_grid[:, :, :, 1] == N_bins_delta] = N_bins_delta - 1
    sky_coords_grid[:, :, :, 2] = (sky_coords_grid[:, :, :, 2] - r_min) // d_r
    sky_coords_grid[:, :, :, 2][sky_coords_grid[:, :, :, 2] == N_bins_r] = N_bins_r - 1

    alpha_delta_grid, r_grid = alpha_delta_r_projections_from_observed(observed_grid, N_bins_x, N_bins_y, N_bins_z, sky_coords_grid, N_bins_alpha, N_bins_delta, N_bins_r)
    print('Shape of alpha-delta grid:', alpha_delta_grid.shape)
    print('Shape of r grid:', r_grid.shape)

    # the code below is provided as a laid-out walkthrough of what just happened upstairs
    # for i in range(N_bins_x):
    #     for j in range(N_bins_y):
    #         for k in range(N_bins_z):
    #             sky_grid_alpha = sky_coords_grid[i, j, k, 0]
    #             sky_grid_delta = sky_coords_grid[i, j, k, 1]
    #             sky_grid_r = sky_coords_grid[i, j, k, 2]
    #             alpha_index = int((sky_grid_alpha - alpha_min) // d_alpha)
    #             delta_index = int((sky_grid_delta - delta_min) // d_delta)
    #             r_index = int((sky_grid_r - delta_r) // d_r)
    #             bin_vote_count = observed_grid[i, j, k]
    #             alpha_delta_grid[alpha_index, delta_index] += bin_vote_count
    #             r_grid[r_index] += bin_vote_count

    print('Maximum number of voters per single bin in alpha-delta grid:', alpha_delta_grid.max())
    print('Minimum number of voters per single bin in alpha-delta grid:', alpha_delta_grid.min())
    print('Maximum number of voters per single bin in r grid:', r_grid.max())
    print('Minimum number of voters per single bin in r grid:', r_grid.min())
    print('N_tot_observed = N_tot_alpha_delta = N_tot_r:', N_tot == np.sum(alpha_delta_grid) == np.sum(r_grid))

    expected_grid = np.array([[[alpha_delta_grid[int(sky_coords_grid[i, j, k, 0]), int(sky_coords_grid[i, j, k, 1])] * r_grid[int(sky_coords_grid[i, j, k, 2])]
                                for k in range(N_bins_z)]
                               for j in range(N_bins_y)]
                              for i in range(N_bins_x)])

    expected_grid /= N_tot  # normalization
    expected_grid *= vol_adjust_ratio_grid  # volume adjustment
    print('Expected grid shape:', expected_grid.shape)
    print('Maximum number of expected votes:', expected_grid.max())
    print('Minimum number of expected votes:', expected_grid.min())

    if save:
        np.save(savename + "_exp_grid.npy", expected_grid)

    return expected_grid, (bin_centers_edges_xs, bin_centers_edges_ys, bin_centers_edges_zs)


# TODO: the expected grid and sig grid threshold should be dealt with
def significance(observed_grid: np.ndarray, expected_grid: np.ndarray, expected_votes_threshold: float = 5., significance_threshold: float = 5., save: bool = False, savename: str = 'saves/saved') -> np.ndarray:
    expected_grid[expected_grid < expected_votes_threshold] = expected_votes_threshold
    # expected_grid[expected_grid == 0.] = 0.01  # this resolves the division by zero error
    sig_grid = (observed_grid - expected_grid) / np.sqrt(expected_grid)
    sig_grid[sig_grid < significance_threshold] = 0.
    # this prepares the grid for the blobbing procedure, which requires a bright on dark image in grayscale, the values of the sig grid now run from 0 to 255
    sig_grid -= sig_grid.min()
    sig_grid /= (sig_grid.max())
    sig_grid *= 255
    print('Maximum significance:', sig_grid.max())
    print('Minimum significance:', sig_grid.min())
    if save:
        np.save(savename + '_sig_grid.npy', sig_grid)
    return sig_grid


def blob(grid: np.ndarray, bin_centers_xyzs: tuple, galaxies_cartesian_coords: np.ndarray, min_sigma_: float = 10., max_sigma_: float = 50., overlap_: float = 0.05, save: bool = False, savename: str = 'saves/saved', plot: bool = False) -> (list, np.ndarray):
    blob_grid_indices = blob_dog(grid, min_sigma=min_sigma_, max_sigma=max_sigma_, overlap=overlap_)  # TODO: experiment with overlap
    # print(blobs_indices)
    blob_centers_xyzs = np.array(blob_grid_indices.T, dtype=int)
    blob_centers_xs, blob_centers_ys, blob_centers_zs = blob_centers_xyzs[0], blob_centers_xyzs[1], blob_centers_xyzs[2]
    bin_centers_xs, bin_centers_ys, bin_centers_zs = bin_centers_xyzs[0], bin_centers_xyzs[1], bin_centers_xyzs[2]
    blob_centers_xs = bin_centers_xs[blob_centers_xs]
    blob_centers_ys = bin_centers_ys[blob_centers_ys]
    blob_centers_zs = bin_centers_zs[blob_centers_zs]
    blob_centers_xyzs = blob_centers_xs, blob_centers_ys, blob_centers_zs
    if plot:
        plot_grid_with_true_centers(blob_centers_xyzs, galaxies_cartesian_coords, 83, showplot=True)
    if save:
        np.save(savename + '_blob_grid_indices.npy', blob_grid_indices)
        np.save(savename + '_blob_centers_xyzs.npy', blob_centers_xyzs)
    return blob_grid_indices, np.array(blob_centers_xyzs).T


def refine_grid(blob_cartesian_coords, galaxies_cartesian_coords, radius, padding=5, finer_grid_spacing=2):
    finer_coords = []
    grid_length = radius + padding
    N_bins = int(np.ceil(2 * grid_length / finer_grid_spacing))
    finer_kernel_grid = kernel(radius, finer_grid_spacing, additional_thickness=1)
    for blob_x, blob_y, blob_z in blob_cartesian_coords:

        # voting procedure
        x_bound_lower, x_bound_upper = blob_x - grid_length, blob_x + grid_length
        y_bound_lower, y_bound_upper = blob_y - grid_length, blob_y + grid_length
        z_bound_lower, z_bound_upper = blob_z - grid_length, blob_z + grid_length
        range_ = ((x_bound_lower, x_bound_upper), (y_bound_lower, y_bound_upper), (z_bound_lower, z_bound_upper))
        finer_density_grid, finer_observed_grid_edges = np.histogramdd(galaxies_cartesian_coords, range=range_, bins=N_bins)
        finer_observed_grid = np.round(fftconvolve(finer_density_grid, finer_kernel_grid, mode='same'))

        # projection and sampling procedure
        finer_expected_grid, finer_bin_centers_edges = project_and_sample(finer_observed_grid, finer_observed_grid_edges, refined_grid_spacing=finer_grid_spacing)

        # significance grid and blobbing procedure
        finer_significance_grid = significance(finer_observed_grid, finer_expected_grid)
        blob_grid_indices, blob_cartesian_coords = blob(finer_significance_grid, finer_bin_centers_edges, galaxies_cartesian_coords)

        # grab highest significance blob
        if len(blob_grid_indices) > 0:
            blob_grid_indices = blob_grid_indices.astype(int)
            max_significance_blob = np.array([finer_significance_grid[blob[0], blob[1], blob[2]] for blob in blob_grid_indices])
            print('Maximum significance blob index:', np.argmax(max_significance_blob))
            finer_coords.append(blob_cartesian_coords[np.argmax(max_significance_blob)])

    finer_coords = np.array(finer_coords).T
    print(finer_coords)
    print(finer_coords.shape)
    plot_grid_with_true_centers(finer_coords, galaxies_cartesian_coords, 83, showplot=True)


def main():
    parser = ArgumentParser(description="( * ) Center Finder ( * )")
    parser.add_argument('file', metavar='DATA_FILE', type=str, help='Name of fits file to be fitted.')
    parser.add_argument('-t', '--test', type=int, default=None, help='If this argument is present, testing will occur at radius entered as argument.')
    parser.add_argument('-p', '--params_file', type=str, default='params.json', help='If this argument is present, the cosmological parameters will be uploaded from given file instead of the default.')
    parser.add_argument('-s', '--save', action='store_true', help='If this argument is present, the x, y and z bin centers will be saved in the "saves" folder along with the observed, expected and significance grids.')
    args = parser.parse_args()

    if (args.test is not None):
        load_hyperparameters(args.params_file)
        savename_ = 'saves/' + args.file.split('.')[0]
        bao_radius = args.test
        observed_grid, observed_grid_edges, galaxies_cartesian_coords = single_radius_vote(args.file, bao_radius, save=args.save, savename=savename_)
        expected_grid, bin_centers_edges = project_and_sample(observed_grid, observed_grid_edges, save=args.save, savename=savename_)
        significance_grid = significance(observed_grid, expected_grid, save=args.save, savename=savename_)
        blob_grid_indices, blob_cartesian_coords = blob(significance_grid, bin_centers_edges, galaxies_cartesian_coords, min_sigma_=2., max_sigma_=30., overlap_=0.05, save=args.save, savename=savename_)
        refine_grid(blob_cartesian_coords, galaxies_cartesian_coords, bao_radius)


if __name__ == '__main__':
    main()
