import numpy as np
from scipy import stats


def _mode_scalar(values):
    mode = stats.mode(values, keepdims=True).mode
    if np.isscalar(mode):
        return float(mode)
    return float(mode[0])


def compatible_get_separation_value(data_2d_global_arg):
    n_sample = 100
    x_data = np.unique(np.sort(data_2d_global_arg[:, 0]))
    y_data = np.unique(np.sort(data_2d_global_arg[:, 1]))

    diffs_x = np.zeros(shape=[n_sample])
    diffs_y = np.zeros(shape=[n_sample])

    for p in range(n_sample):
        x_rand_num = int(np.random.rand() * (len(x_data) - 1))
        y_rand_num = int(np.random.rand() * (len(y_data) - 1))
        diffs_x[p] = np.abs(x_data[x_rand_num] - x_data[x_rand_num + 1])
        diffs_y[p] = np.abs(y_data[y_rand_num] - y_data[y_rand_num + 1])

    return (_mode_scalar(diffs_x) + _mode_scalar(diffs_y)) / 2


def patch_adabox_mode_compat(proc, tools):
    """Patch adabox modules to work with SciPy versions that return scalar mode results."""
    proc.get_separation_value = compatible_get_separation_value
    tools.get_separation_value = compatible_get_separation_value
