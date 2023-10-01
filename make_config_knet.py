from mmengine import Config
cfg = Config.fromfile('./configs/knet/knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512.py')
dataset_cfg = Config.fromfile('mmseg/configs/_base_/datasets/lakeSeg_pipeline.py')
cfg.merge_from_dict(dataset_cfg)

NUM_CLASS = 2
# SyncBN -> BN:single gpu
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.data_preprocessor.size = cfg.crop_size

# 
# cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.decode_head.kernel_generate_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head.num_classes = NUM_CLASS

# 训练 Batch Size
cfg.train_dataloader.batch_size = 4

# 结果保存目录
cfg.work_dir = './work_dirs/LakeSeg-KNet'

cfg.train_cfg.max_iters = 20000 
cfg.train_cfg.val_interval = 500 
cfg.default_hooks.logger.interval = 100 
cfg.default_hooks.checkpoint.interval = 2500 
cfg.default_hooks.checkpoint.max_keep_ckpts = 2 
cfg.default_hooks.checkpoint.save_best = 'mIoU' 

# 随机数种子
cfg['randomness'] = dict(seed=0)
cfg.dump('lakeSegConfig/LakeSegDataset_KNnetR_20231001.py')