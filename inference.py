import pandas as pd
import cv2
from mmseg.apis import init_model, inference_model, show_result_pyplot
import matplotlib.pyplot as plt


def prediction(config_file, checkpoint_file, device, output_dir, plot = True):
    model = init_model(config_file,checkpoint_file,device=device)
    if not os.path.exists(f'./{output_dir}'):
        os.mkdir(f'./{output_dir}')
    for p in glob.glob(os.path.join(path, f"data/test/tiles_jpg/*.jpg")):
        process_single_img(p, save = True)

    results = gpd.GeoDataFrame(columns=['image', 'region_num', 'geometry'], geometry='geometry')
    for index, row in test_df.iterrows():
        img, reg_num, shape = row["img"], row["reg_num"], row["shape"]
        tile_path = os.path.join(output_dir, 'testset-pred', f'pred-{img}_region{reg_num}*.npy')
        # merge labels of tiles
        region_mask = merge_tiles(tile_path, shape[0], shape[1])
        #find polygons of regions
        poly_list = find_poly(region_mask)
        temp = gpd.GeoDataFrame({"image": [img for i in range(len(poly_list))], "region_num": [reg_num for i in range(len(poly_list))], "geometry": poly_list})
        results = results = pd.concat([results,temp])
        if plot:
            regp = os.path.join(path, f"data/test/{img}_region{reg_num}.tif")
            with rasterio.open(regp) as reg_raster:
                fig, ax = plt.subplots()
                rasterio.plot.show(regp, ax = ax)

                geocoor = results["geometry"].apply(lambda x : x.exterior.coords.xy)
                geo_row = geocoor.apply(lambda x : x[0]).sum()
                geo_col = geocoor.apply(lambda x : x[1]).sum()
                
                ax.scatter(geo_row, geo_col, color = "red", linewidths = 0.1, alpha = 0.5)
                plt.show()
        return results

def process_single_img(img_path, save=False):
    print("img_path is: "+ img_path)
    img_bgr = cv2.imread(PATH_IMAGE+"/"+img_path)

    # prediction
    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
    # we do not map integer to color class since we only have one label-lake
    # pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
    # for idx in palette_dict.keys():
    #     pred_mask_bgr[np.where(pred_mask==idx)] = palette_dict[idx]
    # pred_mask_bgr = pred_mask_bgr.astype('uint8')

    # save predicted masks into outputs/testset-pred dir
    if save:
        save_path = os.path.join(output_dir, 'testset-pred', 'pred-'+img_path.split('/')[-1][:-4]+'.tif')
        cv2.imwrite(save_path, pred_mask)

def merge_tiles(tile_path, img_height, img_width):
    row_num, col_num = int(img_height / 512), int(img_width/512)
    # print(row_num, col_num)
    np_list = [[None for i in range(col_num)] for i in range(row_num)]
    t_list = []
    for t in glob.glob(tile_path):
        img = t.split("/")[-1]
        s = img.split("_")
        img = "_".join(s[:5])
        reg_num = s[-2][-1]

        tile_num = int(s[-2][4:])
        t_list.append(tile_num)
        # print(s, tile_num)
        l = np.load(t) #model inference
        row = tile_num // col_num
        col = tile_num % col_num
        # print(row, col)
        t_list.append((row, col))
        np_list[row][col] = l
    merged_label = np.block(np_list)
    return merged_label

def find_poly(mask):
    merged_label = mask.astype(np.uint8)
    contours, _ = cv2.findContours(merged_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polys = []
    for con in contours:
        yy, xx = list(zip(*con.squeeze()))
        with rasterio.open(p) as reg_raster:
            yy1, xx1 = reg_raster.xy(xx, yy)
            poly = geometry.Polygon([[x, y] for x , y in zip(yy1, xx1)])
            polys.append(poly)
    return polys

if __name__ == "__main__":
    path = "/home/Shared/competition/SupraglacialLakesDetection"
    config_file = "./lakeSegConfig/ LakeSegDataset_UNetR_20231001.py"

    checkpoint_file = "./work_dirs/LakeSeg-UNetR/best_mIoU_iter_39000.pth"
    device = 'cuda:0'

    test_df = pd.read_csv(os.path.join(path, "data/test_df.csv"))
    output_dir = os.path.join(path, "output")
    prediction(config_file, checkpoint_file, device, output_dir, plot = True)
