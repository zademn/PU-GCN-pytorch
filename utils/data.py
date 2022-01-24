import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def load_h5_data(h5_filename="", opts=None, skip_rate=1, use_randominput=True):
    """
    skip_rate: {int} -- step_size when loading the dataset
    """
    num_point = opts.num_point
    num_4X_point = int(opts.num_point * 4)
    num_out_point = int(opts.num_point * opts.up_ratio)

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
    def __init__(self, data_path, opts, skip_rate=1):
        f = h5py.File(data_path, "r")
        data, ground_truth, data_radius = load_h5_data(
            h5_filename=data_path, opts=opts, skip_rate=skip_rate, use_randominput=False
        )
        self.data = torch.tensor(data)
        self.ground_truth = torch.tensor(ground_truth)

    def __getitem__(self, idx):
        return PairData(pos_s=self.data[idx], pos_t=self.ground_truth[idx])

    def __len__(self):
        return len(self.data)
