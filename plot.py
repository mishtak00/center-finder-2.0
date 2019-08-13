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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_grid_with_true_centers(voted_centers_xyzs, galaxies_cartesian_coords, N_true_centers, vote_threshold=0, saveplot=True, savename='3d_plot.png', showplot=False):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.scatter(*voted_centers_xyzs, s=80, alpha=0.7, marker='.', edgecolor='k', label='Found Centers', c='b')
    true_centers = galaxies_cartesian_coords[:N_true_centers].T
    ax.scatter(true_centers[0], true_centers[1], true_centers[2], s=150, c='r', marker='x', edgecolor='k', label='True Centers')
    ax.set_xlabel('$x$ [$h^{-1}$Mpc]')
    ax.set_ylabel('$y$ [$h^{-1}$Mpc]')
    ax.set_zlabel('$z$ [$h^{-1}$Mpc]')
    ax.set_title('True and Blobbed Centers')
    # ax.set_title('True and Found Centers' + '\n' + 'Vote Threshold = ' + str(vote_threshold))
    ax.legend(loc='best')
    if saveplot:
        plt.savefig(savename)
    if showplot:
        plt.show()


def plot_expected_vs_observed(observed_grid, expected_grid, galaxies_cartesian_coords, N_true_centers, lower_bound, upper_bound, diagonal=False, style='density', savename='graphs/exp_vs_obs.jpeg'):
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
    # true_centers = galaxies_cartesian_coords[:N_true_centers].T
    # true_centers_indices = np.array([[np.ceil((true_centers[i, j] - true_centers[i].min()) / grid_spacing) for j in range(len(true_centers[i]))] for i in range(len(true_centers))], dtype=int).T
    # true_centers_observed_grid = np.array([observed_grid[index_tuple[0], index_tuple[1], index_tuple[2]] for index_tuple in true_centers_indices])
    # true_centers_expected_grid = np.array([expected_grid[index_tuple[0], index_tuple[1], index_tuple[2]] for index_tuple in true_centers_indices])
    # plt.scatter(true_centers_observed_grid, true_centers_expected_grid, marker='*', color='red')
    plt.savefig(savename)
