## PU-GCN Pytorch (Work in progress)

The [PU-GCN paper](https://arxiv.org/abs/1912.03264) rewritten in Pytorch using [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) and layers provided in the [DeepGCN] repo (https://github.com/lightaime/deep_gcns_torch)

`gcn_lib` folder is taken from the DeepGCN repository

## Relevant repositories

- [PU-GCN](https://github.com/guochengqian/PU-GCN)
- [DeepGCN](https://github.com/lightaime/deep_gcns_torch)
- [PUGAN-pytorch](https://github.com/UncleMEDM/PUGAN-pytorch)
- [chamferdist](https://github.com/krrish94/chamferdist) 
- [chamfer_distance](https://github.com/otaheri/chamfer_distance) (I couldn't make it work tho)


Known bugs / need help:
- doesn't learn after an epoch (all predicted points are the same)
- still need to implement batch training
- can't manage to run a chamfer loss
