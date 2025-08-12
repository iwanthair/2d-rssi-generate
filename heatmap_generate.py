import os
import numpy as np
import cv2
import argparse
import torch
import gpytorch

from utils import create_out_contour
from gp_utils import GaussianProcessModel, GaussianProcessModel2, GaussianProcessModel3

SIZE = 256

filtered_idx = [
    "0102", "0103", "0104",
    "0528", "0529", "0530",
    "0858", "0859", "0860",
    "0957", "0958", "0959",
    "1161", "1162", "1163",
    "1482", "1483", "1484",
    "1593", "1594", "1595",
    "2901", "2902", "2903",
    "3498", "3499", "3500",
    "3669", "3670", "3671",
    "4056", "4057", "4058",
    "4992", "4993", "4994",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmap from floor plan.")
    parser.add_argument("--input_dir", type=str, default="HouseExpo/GroundTruth_Scale100", required=False, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, default="HouseExpo/GroundTruth_Scale100/Heatmaps2", required=False, help="Directory to save output heatmaps.")
    parser.add_argument("--start_point", type=int, default=0, required=False, help="Starting idx for heatmap generation.")
    parser.add_argument("--iterations", type=int, default=150, required=False, help="Number of iterations for Gaussian Process.")
    args = parser.parse_args()

    start_idx = args.start_point
    iterations = args.iterations

    os.makedirs(args.output_dir, exist_ok=True)
    input_dir = args.input_dir
    floor_plan_path = os.path.join(input_dir, "FloorPlan")
    receivers_path = os.path.join(input_dir, "Receivers")
    rssi_path = os.path.join(input_dir, "RSSI")
    id2idx_path = os.path.join(input_dir, "id2idx.txt")
    with open(id2idx_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        id_list = [line.split()[0] for line in lines]
        print(f"Total {len(id_list)} IDs found in {id2idx_path}.")
        # fillter out ids that greater than start_idx
        if start_idx > 0:
            id_list = [id for id in id_list if int(id) >= start_idx]
        print(f"Starting heatmap generation from index {start_idx:04d}. Using {len(id_list)} IDs now.")

    floor_plan_files = [os.path.join(floor_plan_path, f"{id}.png") for id in id_list]
    receivers_files = [os.path.join(receivers_path, f"{id}.npy") for id in id_list]
    rssi_files = [os.path.join(rssi_path, f"{id}.npy") for id in id_list]

    assert len(floor_plan_files) == len(receivers_files) == len(rssi_files), "Mismatch in number of files."
    print(f"Found {len(floor_plan_files)} floor plan files, {len(receivers_files)} receivers files, and {len(rssi_files)} RSSI files.")

    # drop the files that in the filtered_idx
    # from 0-5999
    id_list = [id for id in id_list if int(id) not in filtered_idx]
    floor_plan_files = [f for f in floor_plan_files if any(idx in f for idx in id_list)]
    receivers_files = [f for f in receivers_files if any(idx in f for idx in id_list)]
    rssi_files = [f for f in rssi_files if any(idx in f for idx in id_list)]
    print(f"After filtering, found {len(floor_plan_files)} floor plan files, {len(receivers_files)} receivers files, and {len(rssi_files)} RSSI files.")
    assert len(floor_plan_files) == len(receivers_files) == len(rssi_files), "Mismatch in number of existing files after filtering."


    for count, (floor_plan_file, receivers_file, rssi_file) in enumerate(zip(floor_plan_files, receivers_files, rssi_files)):
        print(f"Processing {count + 1}/{len(floor_plan_files)}.")

        # read floor plan image and rssi/receivers data
        try:
            floor_plan_img = cv2.imread(floor_plan_file)
            receivers = np.load(receivers_file)
            # to x.y coordinates
            receivers = receivers[:, [1, 0]]  # swap x and y for (y, x) format
            rssi = np.load(rssi_file)
        except Exception as e:
            print(f"Error reading files for ID {id_list[count]}: {e}")
            continue
        
        # get the contour of the floor plan
        contour_points = create_out_contour(floor_plan_img)
        min_x, min_y = np.min(contour_points, axis=0)
        max_x, max_y = np.max(contour_points, axis=0)
        max_x += 1  # to include the max_x point in the heatmap
        max_y += 1  # to include the max_y point in the heatmap
        # bounder = [(min_x, min_y), (max_x+1, max_y+1)]

        # # show the contour points and min/max coordinates
        # contour_img = floor_plan_img.copy()
        # for pt in contour_points:
        #     contour_img[int(pt[1]), int(pt[0])] = [0, 255, 0]  # Mark contour points in green
        # contour_img[int(min_y), int(min_x)] = [255, 0, 0]  # Mark min point in blue
        # contour_img[int(max_y), int(max_x)] = [0, 0, 255]  # Mark max point in red
        # cv2.imshow("Contour Points", contour_img)
        # cv2.waitKey(0)

        idx = np.indices((SIZE, SIZE)).reshape(2, -1).T
        # filtered points that are within the bounding box
        pts_idx = idx[(idx[:, 0] > min_x) & (idx[:, 0] < max_x) & (idx[:, 1] > min_y) & (idx[:, 1] < max_y)]

        # train Gaussian Process model
        train_x = torch.from_numpy(receivers).double().cuda()
        train_y = torch.from_numpy(rssi).float().double().cuda()
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double().cuda()
        # model = GaussianProcessModel(train_x, train_y, likelihood).double().cuda()
        model = GaussianProcessModel2(train_x, train_y, likelihood).double().cuda()
        # model = GaussianProcessModel3(train_x, train_y, likelihood).double().cuda()

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y) # type: ignore
            total_loss = loss.sum()
            total_loss.backward()
            optimizer.step()

            # if (i+1) % 10 == 0:
            #     print(f"Iteration {i+1}/{iterations}, total loss: {total_loss.item()}")

        # generate heatmap
        model.eval()
        likelihood.eval()
        test_x = torch.from_numpy(pts_idx).double().cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = likelihood(model(test_x))
            # preds_mean = torch.round(preds.mean).cpu().detach().numpy()
            preds_mean = torch.round(preds.mean).cpu().detach().numpy()

        heatmap = np.zeros((SIZE, SIZE))
        # reshape preds_mean to match the size of the heatmap
        # heatmap[pts_idx[:, 1], pts_idx[:, 0]] = preds_mean
        heatmap[pts_idx[:, 1], pts_idx[:, 0]] = preds_mean
        
        max_value = np.max(preds_mean)
        # add max value to the bounder of the heatmap from line (min_x to min_y), (max_x to max_y), (min_x, max_y), (max_x, min_y)
        heatmap[min_y:max_y, min_x] = max_value
        heatmap[min_y:max_y, max_x] = max_value
        heatmap[min_y, min_x:max_x] = max_value
        heatmap[max_y, min_x:max_x] = max_value

        # normalize heatmap to [0, 255]
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
        # change area that outside the bounder to black
        for pts in idx:
            if not (min_x-1 < pts[0] < max_x+1 and min_y-1 < pts[1] < max_y+1):
                heatmap[pts[1], pts[0]] = [0, 0, 0]
        # save heatmap
        heatmap_file = os.path.join(args.output_dir, f"{id_list[count]}.png")
        cv2.imwrite(heatmap_file, heatmap)

        # if count == 5:
        #     print("Stopping after 5 iterations for testing purposes.")
        #     break
        
        torch.cuda.empty_cache()

    
