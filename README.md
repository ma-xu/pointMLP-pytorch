# pointMLP-pytorch
Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework


## Install
Please ensure that python3.7+ is installed. We suggest user use conda to create a new environment.

Install dependencies
```bash
pip install -r requirement.txt
```

Install CUDA kernels
```bash
pip install pointnet2_ops_lib/.
```

## Classification ModelNet40
The dataset will be automatically downloaded, directly run following command to train
```bash
# train pointMLP
python main.py --model pointMLP
# train pointMLP-elite
python main.py --model pointMLPElite
# please add other paramemters as you wish.
```


## Classification ScanObjectNN


## Part segmentation
