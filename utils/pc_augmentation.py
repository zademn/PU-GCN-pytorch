import numpy as np
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from numpy.random import Generator


def nonuniform_sampling(num: int, sample_num: int, rng: Generator = None):
    """Selecta `sample_num` indices from [0, num]

    Parameters
    ----------
    num : int
        max index

    sample_num : int
        how many indexes to sample

    rng: Generator
        random generator


    Returns
    -------
    list
        list of selected indices
    """

    if rng is None:
        rng = np.random.default_rng()

    assert num > sample_num
    sample = set()
    loc = rng.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(rng.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)


def rotate_point_cloud_and_gt(
    input_data: np.ndarray, gt_data: np.ndarray = None, rng: Generator = None
):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction

    Parameters
    ----------
    input_data : np.ndarray of shape (n_points, 3)
        Original point cloud

    gt_data : np.ndarray,  default=None
        Grount truth point cloud

    rng: Generator
        random generator

    Returns
    -------
    np.ndarray, Optional[np.ndarray]
         Nx3 array(s), rotated point cloud
    """
    input_data = input_data.copy()

    if rng is None:
        rng = np.random.default_rng()

    angles = rng.uniform(size=(3)) * 2 * np.pi
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    Ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    Rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    input_data[:, :3] = np.dot(input_data[:, :3], rotation_matrix)
    if input_data.shape[1] > 3:
        input_data[:, 3:] = np.dot(input_data[:, 3:], rotation_matrix)

    if gt_data is not None:
        gt_data = gt_data.copy()

        gt_data[:, :3] = np.dot(gt_data[:, :3], rotation_matrix)
        if gt_data.shape[1] > 3:
            gt_data[:, 3:] = np.dot(gt_data[:, 3:], rotation_matrix)

    return input_data, gt_data


def random_scale_point_cloud_and_gt(
    input_data: np.ndarray,
    gt_data: np.ndarray = None,
    scale_low: float = 0.5,
    scale_high: float = 2,
    rng: Generator = None,
):
    """Randomly scale the point cloud. Scale is per point cloud

    Parameters
    ----------
    input_data : np.ndarray of shape (n_points, 3)
        Nx3 array, original point cloud

    gt_data : np.ndarray, default=None of shape (n_points, 3)
        NX3 array, ground truth point cloud

    scale_low : float, default=0.5
        Lower bound of the scale range

    scale_high : float, default=2
        higher bound of the scale range

    rng: Generator
        random generator

    Returns
    -------
    np.ndarray, Optional[np.ndarray]
         Nx3 array(s), scaled point cloud
    """
    input_data = input_data.copy()
    if rng is None:
        rng = np.random.default_rng()

    scale = rng.uniform(scale_low, scale_high)
    input_data[:, :3] *= scale
    if gt_data is not None:
        gt_data = gt_data.copy()
        gt_data[:, :3] *= scale

    return input_data, gt_data, scale


def shift_point_cloud_and_gt(
    input_data: np.ndarray,
    gt_data: np.ndarray = None,
    shift_range: float = 0.3,
    rng: Generator = None,
):
    """Randomly shift point cloud. Shift is per point cloud.

    Parameters
    ----------
    input_data : np.ndarray of shape (n_points, 3)
        Nx3 array, original point cloud

    gt_data : np.ndarray, default=None of shape (n_points, 3)
        NX3 array, ground truth point cloud

    shift_range : float, default=0.3
        shift range

    rng: Generator
        random generator


    Returns
    -------
    np.ndarray, Optional[np.ndarray]
         Nx3 array(s), shifted point cloud
    """
    input_data = input_data.copy()
    if rng is None:
        rng = np.random.default_rng()

    shifts = rng.uniform(-shift_range, shift_range, 3)
    input_data[:, :3] += shifts
    if gt_data is not None:
        gt_data = gt_data.copy()
        gt_data[:, :3] += shifts
    return input_data, gt_data


def jitter_perturbation_point_cloud(
    input_data: np.ndarray,
    sigma: float = 0.005,
    clip: float = 0.02,
    rng: Generator = None,
):
    """Randomly jitter point clouds

    Parameters
    ----------
    input_data : np.ndarray of shape (n_points, 3)
        Nx3 array, original point cloud

    sigma : float, default=0.005
        _description_

    clip : float, default=0.02
        _description_

    rng: Generator
        random generator


    Returns
    -------
    np.ndarray
        Nx3 array, point cloud
    """
    input_data = input_data.copy()
    if rng is None:
        rng = np.random.default_rng()

    assert clip > 0
    jitter = np.clip(sigma * rng.standard_normal(input_data.shape), -1 * clip, clip)
    jitter[:, 3:] = 0
    input_data += jitter
    return input_data


def rotate_perturbation_point_cloud(
    input_data, angle_sigma=0.03, angle_clip=0.09, rng: Generator = None
):

    """Rotate

    Parameters
    ----------
    input_data : np.ndarray of shape (n_points, 3)
        Nx3 array, original point cloud

    angle_sigma : float, default=0.03
        _description_

    angle_clip : float, default=0.09
        _description_

    rng: Generator, default = None
            random generator

    Returns
    -------
    np.ndarray
        Nx3 array, point clouds
    """
    if rng is None:
        rng = np.random.default_rng()
    input_data = input_data.copy()

    # print(rng.choice(23))
    angles = np.clip(angle_sigma * rng.standard_normal((3,)), -angle_clip, angle_clip)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    Ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    Rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(Rz, np.dot(Ry, Rx))
    input_data[:, :3] = np.dot(input_data[:, :3], R)
    if input_data.shape[1] > 3:
        input_data[:, 3:] = np.dot(input_data[:, 3:], R)
    return input_data
