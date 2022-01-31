# pointMLP-pytorch
Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework

<div align="center">
  <img src="overview.png" width="650px" height="300px">
</div>

Overview of one stage in PointMLP. Given an input point cloud, PointMLP progressively extract local features using residual point MLP blocks. In each stage, we first transform local point using a geometric affine module, then local points are are extracted before and after aggregation respectively. By repeating multiple stages, PointMLP progressively enlarge the receptive field and model entire point cloud geometric information.

## Updates Jan/31/2022:

We will release an official code here: [PointMLP-pytorch](https://github.com/13952522076/pointMLP-pytorch)

This anonymous link will expire on: **4/2/2022**

**Note:** this anonymous link is synchronized with [PointMLP-pytorch@d2b8dba
](https://github.com/13952522076/pointMLP-pytorch/tree/d2b8dbaa06eb6176b222dcf2ad248f8438582026).


## For rebuttal


The link to CurveNet on ScanObjectNN : [[link]](https://drive.google.com/drive/folders/1u02_2aK6hfT3Ds81vtd4wv3n3tFGQ3bX?usp=sharing)

The link to Residual PointNet++ on MOdelNet40: [[link]](https://drive.google.com/drive/folders/1klIpv2QLTVhDWusfQCAMXq-DkYawr-yA?usp=sharing)

The link to intergrating our Affine to other models: [[DGCNN]](https://drive.google.com/drive/folders/1qDkCKVtF-QXrDceBKAvcoZ4mv9vTaYnR?usp=sharing) [[PointNet++]](https://drive.google.com/drive/folders/1jPfB_8xJjkCQfdRAsL1u6FfABpfFKEC9?usp=sharing)

To link to more pre-MLP blocks withou pos-MLP blocks: [[link]](https://drive.google.com/drive/folders/1KORIIUZmEJ3FHKPeKj-p8u9m7o5DKnmQ?usp=sharing)


## Pre-trained models

Please download the pre-trained models and log files here: [[anonymous google drive]](https://drive.google.com/drive/folders/1Jn9HNpPsrq-1XqSmOUtw4cwPMjsIiIpz?usp=sharing)


## Install
Please ensure that python3.7+ is installed. We suggest user use conda to create a new environment.

Install dependencies
```bash
pip install -r requirements.txt
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

- Make data folder and download the dataset
```bash
cd pointMLP-pytorch/classification_ScanObjectNN
mkdir data
cd data
wget http://103.24.77.34/scanobjectnn/h5_files.zip
unzip h5_files.zip
```

- Train pointMLP/pointMLPElite 
```bash
# train pointMLP
python main.py --model pointMLP
# train pointMLP-elite
python main.py --model pointMLPElite
# please add other paramemters as you wish.
```
By default, it will create a fold named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.


## Part segmentation

- Make data folder and download the dataset
```bash
cd pointMLP-pytorch/part_segmentation
mkdir data
cd data
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

- Train pointMLP
```bash
# train pointMLP
python main.py --model pointMLP
# please add other paramemters as you wish.
```

