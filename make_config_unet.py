from mmengine import Config
cfg = Config.fromfile('./configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py')
dataset_cfg = Config.fromfile('mmseg/configs/_base_/datasets/lakeSeg_pipeline.py')
cfg.merge_from_dict(dataset_cfg)

NUM_CLASS = 2
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.data_preprocessor.test_cfg = dict(size_divisor=128)

# Single GPU BN, else SyncBN 
cfg.norm_cfg = dict(type='BN', requires_grad=True) 
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

#  decode/auxiliary 
cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head.num_classes = NUM_CLASS

# Batch Size
cfg.train_dataloader.batch_size = 4

# output saving dir
cfg.work_dir = './work_dirs/LakeSeg-UNetR'

# model save & log
cfg.train_cfg.max_iters = 40000 
cfg.train_cfg.val_interval = 500 
cfg.default_hooks.logger.interval = 100 
cfg.default_hooks.checkpoint.interval = 2500  
cfg.default_hooks.checkpoint.max_keep_ckpts = 1 
cfg.default_hooks.checkpoint.save_best = 'mIoU' 

# random seed
cfg['randomness'] = dict(seed=42)

cfg.dump('lakeSegConfig/LakeSegDataset_UNetR_20231001.py')