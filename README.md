# Install the mmsegmentation framework 
```shell
conda create -n segmentation python=3.9
conda activate segmentation 
pip3 install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e .
pip install opencv-python pillow matplotlib seaborn tqdm pytorch-lightning 'mmdet>=3.1.0'
```
# train the network 
For the U-Net
```shell
python tools/train.py lakeSegConfig/LakeSegDataset_UNetR_20231001.py 
```
For the KNet
```shell
python tools/train.py lakeSegConfig/LakeSegDataset_KNnetR_20231001.py 
```

# Inference the network
Fill out the config_file and checkpoint_file.
The checkpoint_file could be found at the work_dirs
Then run
```python
python inference.py
```

# Processed dataset
The processed dataset could be located at the data/geo_datasetsR
data_processing.ipynb file contains modules as follows:
clip_regions: clip and padding targeted 6 lake regions from raw Geotiffs;
label_regions: label each region tiff pixels
clip_tiles: split region tiff and labels into small labels
