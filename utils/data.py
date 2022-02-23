import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os

from .pc_augmentation import (
    nonuniform_sampling,
    jitter_perturbation_point_cloud,
    random_scale_point_cloud_and_gt,
    rotate_perturbation_point_cloud,
    shift_point_cloud_and_gt,
    rotate_point_cloud_and_gt,
)

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
    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True
    )
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    print("total %d samples" % (len(input)))
    return input, gt, data_radius


def save_xyz_file(numpy_array, path):
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

        Args:
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
        data_path: str,
        num_point: int,
        up_ratio: int = 4,
        skip_rate: int = 1,
        augment: bool = False,
    ):
        # f = h5py.File(data_path, 'r')
        data, ground_truth, data_radius = load_h5_data(
            h5_filename=data_path,
            num_point=num_point,
            up_ratio=up_ratio,
            skip_rate=skip_rate,
            use_randominput=False,
        )
        self.data = data
        self.ground_truth = ground_truth
        self.data_radius = data_radius
        self.augment = augment
        assert len(self.data) == len(self.ground_truth), "invalid data"

    def __getitem__(self, idx):
        input_data, gt_data, radius_data = (
            self.data[idx],
            self.ground_truth[idx],
            self.data_radius,
        )

        if self.augment:
            # for data aug
            input_data, gt_data = rotate_point_cloud_and_gt(input_data, gt_data)
            input_data, gt_data, scale = random_scale_point_cloud_and_gt(
                input_data, gt_data, scale_low=0.9, scale_high=1.1
            )
            input_data, gt_data = shift_point_cloud_and_gt(
                input_data, gt_data, shift_range=0.1
            )
            radius_data = radius_data * scale

        return PairData(pos_s=torch.tensor(input_data), pos_t=torch.tensor(gt_data))

    def __len__(self):
        return len(self.data)
