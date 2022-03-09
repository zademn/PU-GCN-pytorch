# PU-GCN Pytorch (Work in progress)

The [PU-GCN paper](https://arxiv.org/abs/1912.03264) rewritten in Pytorch using [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) and layers provided in the [DeepGCN](https://github.com/lightaime/deep_gcns_torch) repository.

## Instalation
1. Clone the repository
```
```
2. Make sure to have [pytorch](https://pytorch.org/) and [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/) installed.  


Torch versions used:
```bash
torch==1.10.2+cu113
torch-cluster==1.5.9
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
torchaudio==0.10.2+cu113
torchvision==0.11.3+cu113
```

3. Extra libraries are in the `requirements.txt`

4. Chamfer distance is added as a submodule from [this repository](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)
```bash
git submodule add https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
```

## How to run?
### Training

The `train/` directory contains configurations (model, train and data configurations) in `yaml` format.  
To train a model open the  `Training.ipynb` notebook, load the configuration and run the notebook. **Make sure to specify the data directory**.

Trained models will be saved in a `trained-models` directory. Each training session will create a new directory with python datetime `%Y_%m_%d_%H_%M` format. These directories will contain torch checkpoints with the name `f"ckpt_epoch_{epoch}"`.


### Evaluation
Open the `Evaluation.ipynb` notebook and specify the path to the model you want to train. 

## Repository directory structure
```bash
- gcn_lib # The gcn_lib folder in the DeepGCN repo
- pugcn_lib
    - feature_extractor.py # DenseGCN, InceptionDenseGCN and other compounded modules
    - models.py # PUGCN model implementation
    - torch_nn.py # Extra torch layers / modules
    - upsample.py # Upsample layers (NodeShuffle, PointShuffle)
- train # Contains training configurations
    - config*.yaml
- utils
    - data.py # DataLoaders and Data classes
    - pc_augmentation.py # augmentation functions for point clouds
    - viz.py # Point cloud visualizations 
- evaluation # code taken from https://github.com/yulequan/PU-Net.
- Training.ipynb # Training  notebook. Run this to train a model with a config from train/
- Evaluation.ipynb # Evaluation notebook. Run this to evaluate a trained model
```


## Relevant repositories

- [PU-GCN](https://github.com/guochengqian/PU-GCN) -- Original repository. Dataset downloaded from [here](https://drive.google.com/file/d/1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg/view)
- [DeepGCN](https://github.com/lightaime/deep_gcns_torch)
- [PUGAN-pytorch](https://github.com/UncleMEDM/PUGAN-pytorch)
- [chamferdist](https://github.com/krrish94/chamferdist) -- couldn't make it work
- [chamfer_distance](https://github.com/otaheri/chamfer_distance) -- couldn't make it work tho)
- [Chamfer distance](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) -- Working chamfer distance. [This](https://github.com/otaheri/chamfer_distance) and [this]((https://github.com/krrish94/chamferdist)) didn't work

Known bugs / need help:
- [ ] Reproduce results

