# Synapse
Fine-tuning of pretrained CNN model on specific classes of MS COCO Dataset using Faster RCNN 

### Fork of py-faster-rcnn-ft
We forked the original version of [py-faster-rcnn-ft](https://github.com/DFKI-Interactive-Machine-Learning/py-faster-rcnn-ft) for adding changes relevant to our research. For quick introduction/license/etc. please see the original repository of [py-faster-rcnn-ft](https://github.com/DFKI-Interactive-Machine-Learning/py-faster-rcnn-ft).

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Installation](#installation)
4. [Demo](#demo)
5. [Beyond the demo: training and testing VOC](#beyond-the-demo-installation-for-training-and-testing-models-for-voc)
6. [Beyond the demo: training and testing MSCOCO](#beyond-the-demo-installation-for-training-and-testing-models-for-mscoco)
7. [Available Classes to Train On (MSCOCO)](#available-classes-to-train-on-mscoco)
8. [Train on custom classes](#train-on-custom-classes)
9. [Usage](#usage)

### Requirements: software

You can't use the distribution version of caffe or protobuf, those need to be compiled as some code needed special adaptations.

  **Note:** You don't need to install pycaffe separately, we included pycaffe in our repository with small changes needed for our research([cuDNN](https://github.com/rbgirshick/caffe-fast-rcnn/issues/14)).
  
### Requirements: hardware
1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training Fast R-CNN with VGG16, you'll need a K40 (~11G of memory)
3. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation

#### Ubuntu 17.04
Follow these detailed installation instructions to get the software running with CUDA.

0. Install Ubuntu 17.04 and use the **Additional Drivers** app to activate the Nvidia binary driver (nvidia-375)

1. Install dependencies (you need a root shell):
```Shell
apt install python-pip python-opencv libboost-dev-all libboost-dev libgoogle-glog-dev libgflags-dev libsnappy-dev libatlas-dev libatlas4-base libatlas-base-dev libhdf5-serial-dev liblmdb-dev libleveldb-dev libopencv-dev g++-5 nvidia-cuda-toolkit cython python-numpy python-setuptools python-protobuf python-skimage python-tk python-yaml
pip2 install easydict
```
2. Append to /etc/bash.bashrc (and execute in the shell for immediate effect):
```Shell
export CUDAHOME=/usr/local/cuda
export PYTHONPATH=~/py-faster-rcnn-ft/lib:$PYTHONPATH # we assume that you will clone py-faster-rcnn-ft directly into your home directory, else adapt this accordingly - on multiuser systems a path like /opt/... might be more suitable.

```

3. Download https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7/prod/8.0_20170802/cudnn-8.0-linux-x64-v7-tgz (requires login)
  ```Shell
tar -xvzf cudnn-8.0-linux-x64-v7.tgz -C /usr/local
mkdir /usr/local/cuda/bin;ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf
ldconfig

```
4. We need to compile protobuf with gcc-5:
```Shell
git clone https://github.com/google/protobuf.git
cd protobuf
./autogen.sh &&
./configure --prefix=/usr CC=/usr/bin/gcc-5 &&
make -j8 &&
make check &&
make install
```
5. Build Cython modules, Caffe, pycaffe (was tested with Ubuntu 17.04, in case of errors consult http://caffe.berkeleyvision.org/installation.html)
```Shell
git clone https://github.com/DFKI-Interactive-Machine-Learning/py-faster-rcnn-ft.git
cd py-faster-rcnn-ft/lib &&
make -j8 &&
cd ../caffe-fast-rcnn &&
make -j8 &&
make pycaffe
```

6. Download pre-computed Faster R-CNN detectors
```Shell
cd ..
data/scripts/fetch_imagenet_models.sh
```

This will populate the `data` folder with `faster_imagenet_models`. See `data/README.md` for details.
These models were trained on the ImageNet ILSVRC-2012 train dataset.


### Demo

*After successfully completing [basic installation](#installation)*, you'll be ready to run the demo.

The demo performs detection using a VGG_CNN_M_1024 network trained for detection on MS COCO 2014 with the two classes person and car.

To run the demo you need our caffemodel and the MS COCO dataset:
```Shell
cd data
mkdir -p coco/images
cd coco/images
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
unzip train2014.zip
unzip val2014.zip
unzip test2014.zip
cd ..
mkdir annotations
cd annotations
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
unzip instances_train-val2014.zip
cd ../..
wget -P data/faster_rcnn_models/ http://www.dfki.de/~jan/vgg_cnn_m_1024_faster_rcnn_iter_490000.caffemodel
cd tools
./demo.py
```

### Available Classes To Train On (MS COCO)
After downloading datasets you can run a script to find out about possible classes you can train on.
The script is located at data/MSCOCO_API_categories.py

For example: to get all classes with its ids (Note : you need the id of classes for later) call the following function: print_categories_from_name([]) and you will get the following output :

```Shell
cd data
./MSCOCO_API_categories.py

{u'supercategory': u'person', u'id': 1, u'name': u'person'}
{u'supercategory': u'vehicle', u'id': 2, u'name': u'bicycle'}
{u'supercategory': u'vehicle', u'id': 3, u'name': u'car'}
{u'supercategory': u'vehicle', u'id': 4, u'name': u'motorcycle'}d
{u'supercategory': u'vehicle', u'id': 5, u'name': u'airplane'}
{u'supercategory': u'vehicle', u'id': 6, u'name': u'bus'}
{u'supercategory': u'vehicle', u'id': 7, u'name': u'train'}
{u'supercategory': u'vehicle', u'id': 8, u'name': u'truck'}
{u'supercategory': u'vehicle', u'id': 9, u'name': u'boat'}
{u'supercategory': u'outdoor', u'id': 10, u'name': u'traffic light'}
{u'supercategory': u'outdoor', u'id': 11, u'name': u'fire hydrant'}
{u'supercategory': u'outdoor', u'id': 13, u'name': u'stop sign'}
...
```

### Train on custom classes

After identifying the classes you want to train on [(Available Classes To Train On (MSCOCO)](#available-classes-to-train-on-mscoco), you need to do the following changes to train the classes:

open the file : **experiments/cfgs/faster_rcnn_end2end.yml** and fill up CAT_IDS with the ids you're interested in.

Note : if you leave the list empty it will train on all classes. Then save the file and run the end2end script
The default and recommended setting for iterations is 490000, this takes approximately one day with one NVIDIA GTX1080 cards. If you want to change this you need to change line 39 of experiments/scripts/faster_rcnn_end2end.sh

```Shell
./faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 coco
mv output/faster_rcnn_end2end/coco_2014_train/vgg_cnn_m_1024_faster_rcnn_iter_490000.caffemodel data/faster_rcnn_models/
```

After creating the caffemodel with specific classes you are interested in, you need to change the model name in tools/demo.py, see the variable called **NETS** and its key **vgg16**. Note that the demo.py script will use the entry vgg16 if the ```--net``` argument is omitted.

```Shell
NETS = {'vgg16': ('VGG16',
                  'YOUR_CUSTOM_MODEL.caffemodel'),
```

Then run the demo script :

```Shell
cd tools
./demo.py
```


### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF, VGG_CNN_M_1024 and VGG16.

```Shell
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `faster_rcnn_end2end.sh`.
Output is written underneath `output`.

```Shell
./faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```

### References

If you use our software for a research project, we appreciate a reference to our corresponding [paper](https://arxiv.org/pdf/1709.01476).

BibTeX entry:
```
	@article{Sonntag2017a,
		title = {{Fine-tuning deep CNN models on specific MS COCO categories}},
		author = {Sonntag, Daniel and Barz, Michael and Zacharias, Jan and Stauden, Sven and Rahmani, Vahid and Fóthi, Áron and Lőrincz, András},
		archivePrefix = {arXiv},
		arxivId = {1709.01476},
		eprint = {1709.01476},
		pages = {0--3},
		url = {http://arxiv.org/abs/1709.01476},
		year = {2017}
	}
```
