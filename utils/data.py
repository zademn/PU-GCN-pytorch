import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os
from typing import Union
from .pc_augmentation import (
    nonuniform_sampling,
    jitter_perturbation_point_cloud,
    random_scale_point_cloud_and_gt,
    rotate_perturbation_point_cloud,
    shift_point_cloud_and_gt,
    rotate_point_cloud_and_gt,
)


def normalize_pc(data, gt):
    data_radius = np.ones(shape=(len(data)))
    centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True
    )
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    data[:, :, 0:3] = data[:, :, 0:3] - centroid
    data[:, :, 0:3] = data[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    return data, gt, data_radius


# ======================== Load and save from file ========================


def load_h5_data(h5_filename, num_point, up_ratio=4, skip_rate=1, use_randominput=True):
    """
    skip_rate: {int} -- step_size when loading the dataset
    """
    # num_point = num_point
    num_4X_point = int(num_point * 4)
    num_out_point = int(num_point * up_ratio)

    print("h5_filename : ", h5_filename)
    if use_randominput:
        print("use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f["poisson_%d" % num_4X_point][:]
        gt = f["poisson_%d" % num_out_point][:]
    else:
        print("Do not randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f["poisson_%d" % num_point][:]
        gt = f["poisson_%d" % num_out_point][:]

    # name = f['name'][:]
    assert len(input) == len(gt)

    print("Normalization the data")

    input, gt, data_radius = normalize_pc(input, gt)

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    print("total %d samples" % (len(input)))
    return input, gt, data_radius


def save_xyz_file(numpy_array: np.ndarray, path: str):
    """Save a point cloud into a xyz file

    Parameters
    ----------
    numpy_array : nd.array
        Point cloud to save

    path : str
        File path
    """
    num_points = numpy_array.shape[0]
    with open(path, "w") as f:
        for i in range(num_points):
            line = "%f %f %f\n" % (
                numpy_array[i, 0],
                numpy_array[i, 1],
                numpy_array[i, 2],
            )
            f.write(line)
    return


def load_xyz_file(path):
    return np.genfromtxt(os.path.join(path), delimiter=" ")


# ======================== Data classes ========================


class PairData(Data):
    def __init__(self, pos_s: torch.Tensor, pos_t: torch.Tensor):
        """
        PyG Data object that handles a source and a target point cloud

        Parameters:
        ----------
        pos_s: Tensor
            source points positions [N, 3]
        pos_t: Tensor
            target points positions [N, 3]
        """
        super().__init__()
        self.pos_s = pos_s
        self.pos_t = pos_t


class PCDDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        ground_truth: np.ndarray,
        data_radius: Union[np.ndarray, None],
        augment: bool = False,
        seed: int = None,
    ):
        """Initialize a Point Cloud Dataset

        Parameters
        ----------
        data : np.ndarray of shape [n_clouds, n_points, n_dimensions]
            Point cloud data

        ground_truth : np.ndarray of shape [n_clouds, n_points, n_dimensions]
            Ground truth data

        data_radius : Union[np.ndarray, None]

        augment : bool, default=False
            If the data should be augumented

        seed : int, default=None
            random seed

        """
        if data_radius is None:
            data, ground_truth, data_radius = normalize_pc(data, ground_truth)

        self.rng = np.random.default_rng(
            seed
        )  # behaviour might be changed in the future

        self.data = data
        self.ground_truth = ground_truth
        self.data_radius = data_radius
        self.augment = augment
        assert len(self.data) == len(self.ground_truth), "invalid data"

    @classmethod
    def from_h5(
        cls,
        data_path: str,
        num_point: int,
        up_ratio: int = 4,
        skip_rate: int = 1,
        augment: bool = False,
    ):
        """Generate a PCDDataset from an h5 file

        Parameters
        ----------
        data_path : str
            path to the h5 file

        num_point : int
            number of points for the input data

        up_ratio : int, default=4
            upsampling ratio

        skip_rate : int, default=1


        augment : bool, default=False
            If the dataset should be augmented


        Returns
        -------
        PCDDataset
            The dataset constructed from the h5 file
        """
        f = h5py.File(data_path, "r")
        data, ground_truth, data_radius = load_h5_data(
            h5_filename=data_path,
            num_point=num_point,
            up_ratio=up_ratio,
            skip_rate=skip_rate,
            use_randominput=False,
        )
        assert len(data) == len(ground_truth), "invalid data"

        return cls(data, ground_truth, data_radius, augment)

    def __getitem__(self, idx):
        input_data, gt_data, radius_data = (
            self.data[idx],
            self.ground_truth[idx],
            self.data_radius,
        )

        if self.augment:
            # for data aug
            input_data, gt_data = rotate_point_cloud_and_gt(
                input_data, gt_data, rng=self.rng
            )
            input_data, gt_data, scale = random_scale_point_cloud_and_gt(
                input_data, gt_data, scale_low=0.9, scale_high=1.1, rng=self.rng
            )
            input_data, gt_data = shift_point_cloud_and_gt(
                input_data, gt_data, shift_range=0.1, rng=self.rng
            )
            radius_data = radius_data * scale

        return PairData(pos_s=torch.tensor(input_data), pos_t=torch.tensor(gt_data))

    def __len__(self):
        return len(self.data)
