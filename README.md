## PU-GCN Pytorch (Work in progress)

The [PU-GCN paper](https://arxiv.org/abs/1912.03264) rewritten in Pytorch using [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) and layers provided in the [DeepGCN](https://github.com/lightaime/deep_gcns_torch) repo

`gcn_lib` folder is taken from the DeepGCN repository

Needed libraries
- Pytorch, Pytorch geometric, open3d, h5py, tqdm

## Relevant repositories

- [PU-GCN](https://github.com/guochengqian/PU-GCN) -- dataset downloaded from here
- [DeepGCN](https://github.com/lightaime/deep_gcns_torch)
- [PUGAN-pytorch](https://github.com/UncleMEDM/PUGAN-pytorch)
- [chamferdist](https://github.com/krrish94/chamferdist) -- couldn't make it work
- [chamfer_distance](https://github.com/otaheri/chamfer_distance) -- couldn't make it work tho)
- [Chamfer distance](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) -- This works


Known bugs / need help (marked are fixed):
- [x] doesn't learn after an epoch (all predicted points are the same)
- [x] Find and run a chamfer loss
- [ ] Understand the bottleneck and global pooling layer
- [ ] Implement batch training
- [ ] Reproduce results

