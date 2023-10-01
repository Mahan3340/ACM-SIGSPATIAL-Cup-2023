import os 
import numpy as np
import cv2
from tqdm import tqdm

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv

import matplotlib.pyplot as plt

config_file = "./lakeSegConfig/LakeSegDataset_UNet_20230930.py"

checkpoint_file = "./work_dirs/LakeSeg-UNet/best_mIoU_iter_500.pth"
device = 'cuda:0'

model = init_model(config_file,checkpoint_file,device=device)
palette = [
    ['background',[0,0,0]]
    ['lake',[1,1,1]],
]
palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]

if not os.path.exists('outputs/testset-pred'):
    os.mkdir('./outputs/testset-pred')

#where we put test images
PATH_IMAGE = './data/geo_dataset/img_dir/val'

opacity = 0.3 

def process_single_img(img_path, save=False):
    print("img_path is: "+ img_path) 
    img_bgr = cv2.imread(PATH_IMAGE+"/"+img_path)

    # prediction
    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
    print(pred_mask.shape)
    print(pred_mask)
    # we do not map integer to color class since we only have one label-lake
    # pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    # for idx in palette_dict.keys():
    #     pred_mask_bgr[np.where(pred_mask==idx)] = palette_dict[idx]
    # pred_mask_bgr = pred_mask_bgr.astype('uint8')


    
    # save predicted masks into outputs/testset-pred dir
    if save:
        save_path = os.path.join('./outputs', 'testset-pred', 'pred-'+img_path.split('/')[-1])
        cv2.imwrite(save_path, pred_mask)

for path in tqdm(os.listdir(PATH_IMAGE)):
    process_single_img(path,save=True)