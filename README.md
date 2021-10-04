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
The dataset will be automatically downloaded, run following command to train
```bash
# train pointMLP
python main.py --model pointMLP
# train pointMLP-elite
python main.py --model pointMLPElite
# please add other paramemters as you wish.
```
By default, it will create a fold named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.

To conduct voting experiments, run
```bash
# please modify the msg accrodingly
python voting.py --model pointMLP --msg demo
```


## Classification ScanObjectNN


## Part segmentation
