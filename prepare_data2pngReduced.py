import glob
import os
import random
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import rasterio
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
IMAGE_FOLDER = "/home/Shared/competition/SupraglacialLakesDetection/data/train/tiles/*"
IMG_DIR = "/home/Shared/competition/SupraglacialLakesDetection/data/train/img_dir"
ANN_DIR = "/home/Shared/competition/SupraglacialLakesDetection/data/train/ann_dir"
IMG_TRAIN_DIR = "/home/Shared/competition/SupraglacialLakesDetection/data/train/img_dir/train"
IMG_VAL_DIR = "/home/Shared/competition/SupraglacialLakesDetection/data/train/img_dir/val"
ANN_TRAIN_DIR = "/home/Shared/competition/SupraglacialLakesDetection/data/train/ann_dir/train"
ANN_VAL_DIR = "/home/Shared/competition/SupraglacialLakesDetection/data/train/ann_dir/val"
IMG_LIST = []
LABEL_LIST = []
POSITIVE_IMG_LIST = []
POSITIVE_LABEL_LIST = []
NEGATIVE_IMG_LIST = []
NETGATIVE_LABEL_LIST = []
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)
if not os.path.exists(ANN_DIR):
    os.mkdir(ANN_DIR)
if not os.path.exists(IMG_TRAIN_DIR):
    os.mkdir(IMG_TRAIN_DIR)
if not os.path.exists(IMG_VAL_DIR):
    os.mkdir(IMG_VAL_DIR)
if not os.path.exists(ANN_TRAIN_DIR):
    os.mkdir(ANN_TRAIN_DIR)
if not os.path.exists(ANN_VAL_DIR):
    os.mkdir(ANN_VAL_DIR)

for path in glob.glob(IMAGE_FOLDER):
    if path.endswith("npy"):
        LABEL_LIST.append(path)
    else:
        IMG_LIST.append(path)

# fix label names
for i in range(len(LABEL_LIST)):
    # we do not need label string
    # .npy is temp deleted for alignment
    LABEL_LIST[i] = LABEL_LIST[i].replace("_label","").replace(".npy","")

for i in range(len(IMG_LIST)):
    IMG_LIST[i] = IMG_LIST[i].replace(".tif","")         

for label_path in LABEL_LIST:
    temp_label_path = label_path.split(".")[0]+"_label"+".npy"
    label_img = np.load(temp_label_path)
    if len(np.unique(label_img)) == 1:
        NEGATIVE_IMG_LIST.append(label_path)
    else:
        POSITIVE_LABEL_LIST.append(label_path)

for name in IMG_LIST:
    if name in POSITIVE_LABEL_LIST:
        POSITIVE_IMG_LIST.append(name)
    else:
        NEGATIVE_IMG_LIST.append(name)

# random.shuffle(NEGATIVE_IMG_LIST)
# extra_sample_num = int(0.2*len(POSITIVE_IMG_LIST))
# print(f"extra negative sample is {extra_sample_num}")
# POSITIVE_IMG_LIST = POSITIVE_IMG_LIST + NEGATIVE_IMG_LIST[:extra_sample_num]
# print(f"The POSITIVE_IMG_LIST LEN IS {len(POSITIVE_IMG_LIST)}")

# for 

LABEL_LIST= POSITIVE_LABEL_LIST
IMG_LIST = POSITIVE_IMG_LIST


LABEL_LIST_ORDERED = sorted(LABEL_LIST, key=IMG_LIST.index) #!!

img_train, img_val, label_train, label_val = train_test_split(IMG_LIST, LABEL_LIST_ORDERED, test_size = 0.2, random_state = RANDOM_STATE)

for i in range(len(img_train)):
    img_train[i] = img_train[i]+".tif"
for i in range(len(img_val)):
    img_val[i] = img_val[i]+".tif"
for i in range(len(label_train)):
    label_train[i]=label_train[i]+".npy"
for i in range(len(label_val)):
    label_val[i] = label_val[i] + ".npy"


# generating the mask labels
for label_pth in label_train:
    label_pth = label_pth.split(".")[0]+"_label"+".npy"
    label_img = np.load(label_pth)
    name = os.path.join(ANN_TRAIN_DIR,label_pth)
    # first split remove the system path and the second split remove the ".npy" suffix
    name = name.split("/")[-1].split(".")[0]
    name = name.replace("_label","")
    name = name + ".png"
    print(os.path.join(ANN_TRAIN_DIR,name))
    cv2.imwrite(os.path.join(ANN_TRAIN_DIR,name),label_img)

for label_pth in label_val:
    label_pth = label_pth.split(".")[0]+"_label"+".npy"
    label_img = np.load(label_pth)
    name = os.path.join(ANN_VAL_DIR,label_pth)
    # first split remove the system path and the second split remove the ".npy" suffix
    name = name.split("/")[-1].split(".")[0]
    name = name.replace("_label","")
    name = name + ".png"
    print(os.path.join(ANN_VAL_DIR,name))
    cv2.imwrite(os.path.join(ANN_VAL_DIR,name),label_img)


# generate the training images
for img_path in img_train:
    picture = cv2.imread(img_path)
    # similar as label above 
    name = img_path.split("/")[-1].split(".")[0]
    name = name + ".jpg"
    print(os.path.join(IMG_TRAIN_DIR,name))
    cv2.imwrite(os.path.join(IMG_TRAIN_DIR,name),picture)
    
for img_path in img_val:
    picture = cv2.imread(img_path)
    # similar as label above 
    name = img_path.split("/")[-1].split(".")[0]
    name = name + ".jpg"
    print(os.path.join(IMG_VAL_DIR,name))
    cv2.imwrite(os.path.join(IMG_VAL_DIR,name),picture)
