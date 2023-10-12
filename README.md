# PU-GCN Pytorch (Work in progress)

The [PU-GCN paper](https://arxiv.org/abs/1912.03264) rewritten in Pytorch using [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) and layers provided in the [DeepGCN](https://github.com/lightaime/deep_gcns_torch) repository.

Demo colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/pu-gcn-pytorch/blob/master/demo/demo.ipynb)

## Instalation
1. Clone the repository
```bash
git clone https://github.com/zademn/PU-GCN-pytorch.git
```
2. Make sure to have [pytorch](https://pytorch.org/) and [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/) installed.  


Torch versions used:
```bash
torch==2.1.0
torch-cluster==1.6.3+pt21cu121
torch-scatter==2.1.2+pt21cu121
torch-sparse==0.6.18+pt21cu121
torch-spline-conv==1.2.2+pt21cu121
torch_geometric==2.4.0
torchaudio==2.1.0
torchinfo==1.8.0
torchvision==0.16.0
```

3. Extra libraries are in the `requirements.txt`. `pip freeze` is in  `requirements_all.txt`


## How to run?
### Training

The  `conf` directory contains configurations (model, train and data configurations) in `yaml` format. The `config.yaml` will hold the current configurations. `<model-name>_config.yaml` are examples of configurations. 
To train a model open the  `Training.ipynb` notebook, load the configuration and run the notebook. **Make sure to specify the data directory**. 

Trained models will be saved in a `trained-models` directory. Each training session will create a new directory with python datetime `%Y-%m-%d-%H-%M-<model-name>` format. These directories will contain torch checkpoints with the name `f"ckpt_epoch_{epoch}"`.


### Evaluation
Open the `Evaluation.ipynb` notebook and specify the path to the model you want to evaluate. The results will be in the `results` directory. 

## Repository directory structure
```bash
- pugcn_lib
    - feature_extractor.py # InceptionDenseGCN and other compounded modules
    - models.py # PUGCN model implementation
    - torch_geometric_nn.py # Extra torch layers / modules
    - upsample.py # Upsample layers (NodeShuffle, PointShuffle)
- conf # Contains training configurations
    - config*.yaml
- utils
    - losses.py # loss functions
    - data.py # DataLoaders and Data classes
    - pc_augmentation.py # augmentation functions for point clouds
    - viz.py # Point cloud visualizations 
- evaluation # code taken from https://github.com/yulequan/PU-Net.
- Training.ipynb # Training  notebook. Run this to train a model with a config from train/
- Evaluation.ipynb # Evaluation notebook. Run this to evaluate a trained model
- results # Contains results of different models / configurations
```

## Relevant repositories
- [PU-GCN](https://github.com/guochengqian/PU-GCN) -- Original repository. Dataset downloaded from [here](https://drive.google.com/file/d/1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg/view)
- [DeepGCN](https://github.com/lightaime/deep_gcns_torch)
- [PUGAN-pytorch](https://github.com/UncleMEDM/PUGAN-pytorch)
- [Chamfer distance](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) -- Working chamfer distance. [This](https://github.com/otaheri/chamfer_distance) and [this](https://github.com/krrish94/chamferdist) didn't work.


