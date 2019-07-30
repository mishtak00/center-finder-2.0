import json
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
plt.rc('figure', figsize=[12, 12])
plt.rc('font', family='serif')
plt.rc('font', size=16)
plt.rc('lines', lw=3)
plt.rc('axes', labelsize=14)
plt.rc('axes', titlesize=16)
plt.rc('axes', grid=True)
plt.rc('grid', ls='dotted')
plt.rc('xtick', labelsize=10)
plt.rc('xtick', top=True)
plt.rc('xtick.minor', visible=True)
plt.rc('ytick', labelsize=10)
plt.rc('ytick', right=True)
plt.rc('ytick.minor', visible=True)
plt.rc('savefig', bbox='tight')
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import fftconvolve
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser
from skimage.feature import blob_dog


# load hyperparameters from file
def load_hyperparameters(params_file):
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


def load_data(filename):
    catalog_data = fits.open(filename)[1].data
    ra = catalog_data['ra']
    dec = catalog_data['dec']
    redshift = catalog_data['z']
    return ra, dec, redshift


# this calculates the transformation from observed redshift to radial distance
def z2r(z):
    return c_over_H0 * integrate.quad(lambda zeta: (Omega_M * (1 + zeta)**3 + Omega_L)**-0.5, 0, z)[0]


def sky2cartesian(ra, dec, redshift):
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


def cartesian2sky(xs, ys, zs):
    radii = (xs ** 2 + ys ** 2 + zs ** 2) ** 0.5
    redshift = np.array(LUT_redshifts(radii))
    dec = np.rad2deg(np.arcsin(zs / radii))
    ra = np.rad2deg(np.arctan(ys / xs))
    return ra, dec, redshift, radii


def kernel(radius, grid_spacing, show_kernel=False):
    bao_radius = radius  # Mpc/h
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
    kernel_grid = np.array([[[1 if (np.linalg.norm(np.array([i, j, k]) - kernel_center) >= inscribed_r_idx_units_floor
                                    and np.linalg.norm(np.array([i, j, k]) - kernel_center) < inscribed_r_idx_units_ceil)
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


def plot_observed_grid(vote_threshold, voted_centers_xyzs, glxs_crtsn_coords, N_true_centers, saveplot=True, savename='3d_plot.png', showplot=False):
    color = 'cornflowerblue'
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.scatter(*voted_centers_xyzs, s=80, c=color, alpha=0.3, marker='.', edgecolor='k', label='Found Centers')
    true_centers = glxs_crtsn_coords[:N_true_centers].T
    ax.scatter(true_centers[0], true_centers[1], true_centers[2], s=150, c='r', marker='x', edgecolor='k', label='True Centers')
    ax.set_xlabel('$x$ [$h^{-1}$Mpc]')
    ax.set_ylabel('$y$ [$h^{-1}$Mpc]')
    ax.set_zlabel('$z$ [$h^{-1}$Mpc]')
    ax.set_title('True and Found Centers' + '\n' + 'Vote Threshold = ' + str(vote_threshold))
    ax.legend(loc='best')
    if saveplot:
        plt.savefig(savename)
    if showplot:
        plt.show()


def plot_expected_vs_observed(observed_grid, expected_grid, glxs_crtsn_coords, N_true_centers, lower_bound, upper_bound, diagonal=False, style='density', savename='graphs/exp_vs_obs.jpeg'):
    flat_obs = np.ravel(observed_grid)
    flat_exp = np.ravel(expected_grid)

    if style == 'density':
        # plt.hist2d(flat_obs, flat_exp, bins=upper_bound - lower_bound, range=[[lower_bound, upper_bound], [lower_bound, upper_bound]])
        plt.hist2d(flat_obs, flat_exp, bins=100, range=[[lower_bound, upper_bound], [lower_bound, upper_bound]])
        cb = plt.colorbar()
        cb.set_label('Aggregate Vote Counts')
    elif style == 'scatter':
        plt.scatter(flat_obs, flat_exp)

    if diagonal:
        x = np.linspace(lower_bound, upper_bound, 100)
        plt.plot(x, x, color='black')

    plt.xlabel("$N$_${observed}$")
    plt.ylabel("$N$_${expected}$")
    plt.title('Number of expected versus observed voters for catalog with {} true centers'.format(N_true_centers))

    # TODO: there's an issue here with the gridding length of the obs and exp grids. sometimes points at the outer wall fall outside the range. solve this.
    # true_centers = glxs_crtsn_coords[:N_true_centers].T
    # true_centers_indices = np.array([[np.ceil((true_centers[i, j] - true_centers[i].min()) / grid_spacing) for j in range(len(true_centers[i]))] for i in range(len(true_centers))], dtype=int).T
    # true_centers_observed_grid = np.array([observed_grid[index_tuple[0], index_tuple[1], index_tuple[2]] for index_tuple in true_centers_indices])
    # true_centers_expected_grid = np.array([expected_grid[index_tuple[0], index_tuple[1], index_tuple[2]] for index_tuple in true_centers_indices])
    # plt.scatter(true_centers_observed_grid, true_centers_expected_grid, marker='*', color='red')
    plt.savefig(savename)


def single_radius_vote(filename, radius, **kwargs):
    # TODO: fix the kwargs for showing plots
    # show_kernel, saveplot, savename = kwargs

    # gets sky data and transforms them to cartesian
    ra, dec, redshift = load_data(filename)
    xyzs = sky2cartesian(ra, dec, redshift)  # this returns (xs, ys, zs) as a tuple ready for unpacking

    # gets the 3d histogram (density_grid) and the grid bin coordintes in cartesian (grid_edges)
    glxs_crtsn_coords = xyzs.T  # each galaxy is represented by (x, y, z)
    # TODO: fix the outer wall binning out of range issue
    bin_counts_3d = np.array([np.ceil((xyzs[i].max() - xyzs[i].min()) / grid_spacing) for i in range(len(xyzs))], dtype=int)
    density_grid, observed_grid_edges = np.histogramdd(glxs_crtsn_coords, bins=bin_counts_3d)
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

    # vote_threshold = 60
    # voted_centers_xyzs = (observed_grid_edges[i][np.where(observed_grid >= vote_threshold)[i]] for i in range(len(observed_grid_edges)))
    # plot_observed_grid(vote_threshold, voted_centers_xyzs, glxs_crtsn_coords, N_true_centers, savename=filename.split('.')[0] + '.png')

    return observed_grid, observed_grid_edges, glxs_crtsn_coords


def alpha_delta_r_projections_from_observed(observed_grid, N_bins_x, N_bins_y, N_bins_z, sky_coords_grid, N_bins_alpha, N_bins_delta, N_bins_r):
    alpha_delta_grid = np.zeros((N_bins_alpha, N_bins_delta))
    r_grid = np.zeros((N_bins_r,))
    # TODO: np.vectorize()
    for i in range(N_bins_x):
        for j in range(N_bins_y):
            for k in range(N_bins_z):
                alpha_delta_grid[int(sky_coords_grid[i, j, k, 0]), int(sky_coords_grid[i, j, k, 1])] += observed_grid[i, j, k]
                r_grid[int(sky_coords_grid[i, j, k, 2])] += observed_grid[i, j, k]
    return alpha_delta_grid, r_grid


def sample(observed_grid, observed_grid_edges, save=False, savename='saves/saved'):
    bin_centers_xs, bin_centers_ys, bin_centers_zs = np.array([(observed_grid_edges[i][:-1] + observed_grid_edges[i][1:]) / 2 for i in range(len(observed_grid_edges))])
    # print(bin_centers_xs, bin_centers_ys, bin_centers_zs)
    if save:
        try:
            os.mkdir('saves')
        except FileExistsError:
            pass
        np.save(savename + '_xbins.npy', bin_centers_xs)
        np.save(savename + '_ybins.npy', bin_centers_ys)
        np.save(savename + '_zbins.npy', bin_centers_zs)
    # return bin_centers_xs, bin_centers_ys, bin_centers_zs
    bin_centers_xs, bin_centers_ys, bin_centers_zs = np.array([(x, y, z) for x in bin_centers_xs for y in bin_centers_ys for z in bin_centers_zs]).T
    print('Number of bin centers in cartesian coordinates:', len(bin_centers_xs))
    """
    Why can we be sure that it is okay to interpolate the radii and redshift values for these bin centers coordinates?
    Because we know that the range of values of the bin centers is exactly in between the min and the max of the grid bin edges x, y, z.
    The radii come from the 3d euclidian distance, which preserves this relationship (convex function of x,y,z), and thus it is fine
    to use the beforehand-calculated interpolation lookup table to find the redshifts from the radii.
    """
    bin_centers_ra, bin_centers_dec, bin_centers_redshift, bin_centers_radii = cartesian2sky(bin_centers_xs, bin_centers_ys, bin_centers_zs)
    # print(bin_centers_ra, bin_centers_dec, bin_centers_redshift, bin_centers_radii)
    print('Number of bin centers in sky coordinates:', len(bin_centers_ra))

    # total number of votes
    N_tot = np.sum(observed_grid)
    print('Total number of votes:', N_tot)

    # angular volume adjustment calculations
    # NOTE: the N_bins need a last bin because the points at the outermost wall will be out of bounds otherwise

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

    # TODO: fix the indexing here
    # vectorial computation of the sky indices
    sky_coords_grid[:, :, :, 0] = (sky_coords_grid[:, :, :, 0] - alpha_min) // d_alpha
    sky_coords_grid[:, :, :, 0][sky_coords_grid[:, :, :, 0] == N_bins_alpha] = N_bins_alpha - 1
    print('Maximum alpha bin:', sky_coords_grid[:, :, :, 0].max())
    sky_coords_grid[:, :, :, 1] = (sky_coords_grid[:, :, :, 1] - delta_min) // d_delta
    sky_coords_grid[:, :, :, 1][sky_coords_grid[:, :, :, 1] == N_bins_delta] = N_bins_delta - 1
    print('Maximum delta bin:', sky_coords_grid[:, :, :, 1].max())
    sky_coords_grid[:, :, :, 2] = (sky_coords_grid[:, :, :, 2] - r_min) // d_r
    sky_coords_grid[:, :, :, 2][sky_coords_grid[:, :, :, 2] == N_bins_r] = N_bins_r - 1
    print('Maximum r bin:', sky_coords_grid[:, :, :, 2].max())
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
        np.save(savename + "_obs_grid.npy", observed_grid)
        np.save(savename + "_exp_grid.npy", expected_grid)

    return expected_grid


def blob(observed_grid):
    blobs_dog = blob_dog(observed_grid, min_sigma=3., max_sigma=15., threshold=4)
    # print(blobs_dog)
    # TODO: why isn't this working at all?
    return blobs_dog


'''
# TODO: Integrate the blobs
We want a grid with shape of observed grid that has spheres of 1's centered at each found blob. this is needed in order to convolve with the blobbed observed
# grid in order to get the integrated vote value around each blob in the observed grid.


# TODO: Evaluation
After blobbing, we want to find how many blobs were within a max threshold distance (18 Mpc). Put these blobs into true_blobs category, the others are fake.
Another next step is a finer jittering around true blobs.
'''

'''
		ARCHIVE
        # bin_centers_xs, bin_centers_ys, bin_centers_zs = sample(observed_grid, observed_grid_edges)
        # # plot_expected_vs_observed(observed_grid, expected_grid, glxs_crtsn_coords, 667, 5, 525, diagonal=False, style='density')
        # blobs = blob(np.array(observed_grid, dtype=float))
        # blob_centers_xyzs = np.array(blobs.T, dtype=int)  # this returns the blobs' grid indices, the multiplication by 5 turns them into cart coords
        # blob_centers_xs, blob_centers_ys, blob_centers_zs = blob_centers_xyzs[0], blob_centers_xyzs[1], blob_centers_xyzs[2]
        # # print(observed_grid_edges[0].min(), observed_grid_edges[1].min(), observed_grid_edges[2].min())
        # # blob_centers_xyzs = [(bin_centers_xyzs[0][int(blob_centers_xyzs[i, 0])], bin_centers_xyzs[1][int(blob_centers_xyzs[i, 1])], bin_centers_xyzs[2][int(blob_centers_xyzs[i, 2])]) for i in range(len(blob_centers_xyzs))].T

        # blob_centers_xs = bin_centers_xs[blob_centers_xs]
        # blob_centers_ys = bin_centers_ys[blob_centers_ys]
        # blob_centers_zs = bin_centers_zs[blob_centers_zs]
        # blob_centers_xyzs = blob_centers_xs, blob_centers_ys, blob_centers_zs
        # plot_observed_grid(0, blob_centers_xyzs, glxs_crtsn_coords, 83, showplot=True)
'''


def main():
    parser = ArgumentParser(description="( * ) Center Finder ( * )")
    parser.add_argument('file', metavar='DATA_FILE', type=str, help='Name of fits file to be fitted.')
    parser.add_argument('-t', '--test', type=int, default=None, help='If this argument is present, testing will occur.')
    parser.add_argument('-p', '--params_file', type=str, default='params.json', help='If this argument is present, the cosmological parameters will be uploaded from given file instead of the default.')
    args = parser.parse_args()
    load_hyperparameters(args.params_file)

    if (args.test is not None):
        # kernel(108, 5, show_kernel=True)
        observed_grid, observed_grid_edges, glxs_crtsn_coords = single_radius_vote(args.file, args.test)
        expected_grid = sample(observed_grid, observed_grid_edges)


if __name__ == '__main__':
    main()
