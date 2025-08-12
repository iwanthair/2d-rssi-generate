# This script is adapted from HouseExpo dataset generation code.
# Source: https://github.com/TeaganLi/HouseExpo/tree/master

import os
import argparse
import numpy as np
import json
import cv2
import bresenham
from utils import sample_random_pixels, check_wall_number, \
                multi_wall_model, downsample
from gp_utils_sklean import train_gp_rssi, plot_rssi_heatmap
import pandas as pd

meter2pixel = 100
border_pad = 60




def draw_map_mask(file_name, json_path, save_path, target_size=128, wall_thickness=8, samples=500):
    print("Processing", file_name)

    with open(os.path.join(json_path, file_name + '.json')) as json_file:
        json_data = json.load(json_file)

    verts = (np.array(json_data['verts']) * meter2pixel).astype(np.int32)
    x_max, x_min = np.max(verts[:, 0]), np.min(verts[:, 0])
    y_max, y_min = np.max(verts[:, 1]), np.min(verts[:, 1])

    height = y_max - y_min + border_pad * 2
    width = x_max - x_min + border_pad * 2

    verts[:, 0] = verts[:, 0] - x_min + border_pad
    verts[:, 1] = verts[:, 1] - y_min + border_pad  

    cnt_map = np.zeros((height, width), dtype=np.uint8)
    floor_plan = np.zeros((height, width), dtype=np.uint8)
    # print(f"Floor plan shape: {floor_plan.shape}")
    cv2.drawContours(cnt_map, [verts], 0, 255, -1)
    # cnt_map = downsample(cnt_map, scale=0.25, pad_value=0)
    # cnt_map = downsample(cnt_map, scale=0.1, pad_value=0)

    # erode the contour to create a floor plan
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*wall_thickness+1, 2*wall_thickness+1)
    )
    floor_plan = cv2.dilate(cnt_map, kernel, iterations=1)
    # print(f"Floor plan shape after dilation: {floor_plan.shape}")
    floor_plan = cv2.subtract(floor_plan, cnt_map)
    cnt_map = downsample(cnt_map, scale=0.25)
    floor_plan = downsample(floor_plan, scale=0.25)

    # erode the contour to create a mask for the cnt_map so no pts will be sampled near the wall
    # pad_kernel = np.ones((1, 1), dtype=np.uint8)
    # cnt_map_new = cv2.erode(cnt_map, pad_kernel, iterations=1)
    sampled_pts = sample_random_pixels(cnt_map, num_pts=samples+1, seed=42)
    # print(f"Sampled {sampled_pts}.")
    print(f"floor_plan shape: {floor_plan.shape}, cnt_map shape: {cnt_map.shape}.")
    # the last point is tx
    tx_r, tx_c = sampled_pts[-1]

    # dict with receiver positions and their corresponding RSSI values
    RSSI_dict = {
        "x": [],
        "y": [],
        "rssi": [],
        # "num_walls": [],
        # "distance": []
    }

    # check the number of walls between tx and each rx
    fp = floor_plan.copy()
    pts_mask = np.zeros_like(floor_plan, dtype=np.uint8)
    for (rx_r, rx_c) in sampled_pts[:500]:
        pts_mask[rx_r, rx_c] = 255
        pts_list = list(bresenham.bresenham(tx_c, tx_r, rx_c, rx_r))
        num_walls = check_wall_number(floor_plan, pts_list)
        distance = np.sqrt((tx_c - rx_c) ** 2 + (tx_r - rx_r) ** 2) / 25
        print(f"Number of walls between tx ({tx_r}, {tx_c}) and rx ({rx_r}, {rx_c}): {num_walls}, distance: {distance:.2f} m")

        path_loss = multi_wall_model(
            distance=distance,
            L0=40,
            na=2.0,
            n=num_walls,
            Lw=15
        )
        RSSI_dict["x"].append(rx_c/10)
        RSSI_dict["y"].append(rx_r/10)
        RSSI_dict["rssi"].append(path_loss)

        # show rxtx
        # print(f"Number of walls between tx ({tx_r}, {tx_c}) and rx ({rx_r}, {rx_c}): {num_walls}")
        # cv2.circle(fp, (rx_c, rx_r), 3, 128, 1)
        # cv2.circle(fp, (tx_c, tx_r), 3, 128, 1)
        # cv2.line(fp, (tx_c, tx_r), (rx_c, rx_r), 128, 1)
        # # print the number of walls
        # cv2.imshow("Floor Plan", fp)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # gp
    gp = train_gp_rssi(
        X_train= np.array([RSSI_dict["x"], RSSI_dict["y"]]).T,
        y_train=np.array(RSSI_dict["rssi"]),
    )


    # # # # plot the RSSI heatmap
    # w,h from min and max points in 
    # min_x = min(sampled_pts[:, 0])
    # max_x = max(sampled_pts[:, 0])
    # min_y = min(sampled_pts[:, 1])
    # max_y = max(sampled_pts[:, 1])
    # H, W = max_x - min_x, max_y - min_y
    H, W = floor_plan.shape[:2]
    plot_rssi_heatmap(
        gp=gp,
        W=W,
        H=H,
        grid_res=target_size,
        txy=(tx_c, tx_r)
    )


    # floor_plan = downsample(floor_plan, scale=0.1, pad_value=0)
    # cnt_map = downsample(cnt_map, scale=0.1, pad_value=0)
    # pts_mask = resize(pts_mask, target_size=target_size, pad_value=0)
    # floorplan = pad_then_resize(floorplan, target_size, pad_value=0)
    # print(f"Floorplan resized: {floorplan.shape}")
    # cnt_map = pad_then_resize(cnt_map, target_size, pad_value=0)
    # print(f"Mask resized: {cnt_map.shape}")
    
    os.makedirs(os.path.join(save_path, "mask"), exist_ok=True)
    cv2.imwrite(os.path.join(save_path, "mask", file_name + '.png'), cnt_map)
    os.makedirs(os.path.join(save_path, "floorPlan"), exist_ok=True)
    cv2.imwrite(os.path.join(save_path, "floorPlan", file_name + '.png'), floor_plan)
    os.makedirs(os.path.join(save_path, "randomTraj"), exist_ok=True)
    cv2.imwrite(os.path.join(save_path, "randomTraj", file_name + '.png'), pts_mask)


def draw_map(file_name, json_path, save_path):
    print("Processing", file_name)

    with open(os.path.join(json_path, file_name + '.json')) as json_file:
        json_data = json.load(json_file)

    verts = (np.array(json_data['verts']) * meter2pixel).astype(np.int32)
    x_max, x_min = np.max(verts[:, 0]), np.min(verts[:, 0])
    y_max, y_min = np.max(verts[:, 1]), np.min(verts[:, 1])
    cnt_map = np.zeros((y_max - y_min + border_pad * 2,
                        x_max - x_min + border_pad * 2), dtype=np.uint8)

    verts[:, 0] = verts[:, 0] - x_min + border_pad
    verts[:, 1] = verts[:, 1] - y_min + border_pad

    cv2.drawContours(cnt_map, [verts], 0, 255, -1)

    os.makedirs(os.path.join(save_path, "floorplan"), exist_ok=True)
    cv2.imwrite(os.path.join(save_path, "floorplan", file_name + '.png'), cnt_map)

def save_selected_ids(selected_ids, save_path):
    """
    Save the selected map IDs to a text file.
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    np.savetxt(os.path.join(save_path, f"selected_map_ids_{len(selected_ids)}.txt"), selected_ids, fmt="%s")
    print(f"Successfully saved the map subset ids to {save_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate floor plan selection based on room count ids set.")
    parser.add_argument("--path", type=str, default='./HouseExpo/HouseExpoJSON/json/',
                        help="path to HouseExpo json files")
    parser.add_argument("--num_map", type=int, default=6000,
                        help="number of the generated map set")
    # parser.add_argument("--mask_save_path", type=str, default='./HouseExpo/MASK')
    parser.add_argument("--img_save_path", type=str, default='./HouseExpo/GroundTruth')
    parser.add_argument("--ids_save_path", type=str, default='./HouseExpo',
                        help="path to save the selected map ids")
    parser.add_argument("--map_ids_path", type=str, default='./HouseExpo/valid_map_ids_26982.txt',
                        help="the path to the existing map ids set, if set, the map will be generated from its complement set")
    result = parser.parse_args()

    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), result.path))
    img_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), result.img_save_path))
    ids_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), result.ids_save_path))
    map_ids_path = os.path.abspath(os.path.join(os.path.dirname(__file__), result.map_ids_path))

    print("---------------------------------------------------------------------")
    print("|json file path        |{}".format(json_path))
    print("---------------------------------------------------------------------")
    print("|Num of target map set | {}".format(result.num_map))
    print("---------------------------------------------------------------------")
    print("|Save path             | {}".format(img_save_path))
    print("---------------------------------------------------------------------")
    print("|IDs save path         | {}".format(ids_save_path))
    print("---------------------------------------------------------------------")
    print("|Map IDs path          | {}".format(map_ids_path))
    print("---------------------------------------------------------------------")

    if result.map_ids_path:
        existing_map_ids = np.loadtxt(map_ids_path, str)
        
        # # randomly select from the existing set
        # map_ids = np.random.choice(existing_map_ids, result.num_map, replace=False)
        # save_selected_ids(map_ids, ids_save_path)

        # test
        map_ids = [
            "6d5a91c5e6d592f4842eba1e9713fb4d",
            "d90b80eefd8ffd5daebd0371d7548c7e",
            "8abf4712d2999fbc67cf8425c89d3c6a",
            "027ecef5fd9613edfa26122d9099480e",
            "c8d47b180fed046c92c7c3bb3db0c14c",
            "8cb7cb359cb4ae3ca1a63401545e1ed8",
            "30d777707b66beea4a24007f24172510",
            "df2a621574d0b4f1865d7f5476f055f6",
            "4bfbd84a47b36902fd0eb82b045caaa2",
            "8f0c0795f2f26d24eafbca1076400f0d"
        ]

        print(f"Selected {len(map_ids)} maps from the existing set.")

        # use the selected map ids to save floor plan images
        for map_id in map_ids:
            draw_map_mask(map_id, json_path, img_save_path, target_size=256, wall_thickness=20, samples=500)

