import os
from xml.parsers.expat import model
import numpy as np
import cv2
import argparse
import torch
import gpytorch
from scipy.io import loadmat

from utils import create_out_contour
from gp_utils import GaussianProcessModel2 as GPModel

import os
import numpy as np
import cv2
from scipy.io import loadmat

SIZE = 256
EXP = 5
PADDING = 10
ROOT = "RealWorldData"

def rasterize_points(points_xy, size):
    img = np.zeros((size, size), dtype=np.uint8)
    pts = points_xy[:, :2].astype(np.int32)
    img[pts[:, 1], pts[:, 0]] = 255
    return img

def center_image_by_bbox(img, heatmap_img):
    """"""
    # 1 channel image or 3 channel image
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return img

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    tgt_x = img.shape[1] // 2
    tgt_y = img.shape[0] // 2

    dx = int(tgt_x - cx)
    dy = int(tgt_y - cy)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_NEAREST, borderValue=0)
    if heatmap_img is not None:
        shifted_heatmap = cv2.warpAffine(heatmap_img, M, (heatmap_img.shape[1], heatmap_img.shape[0]),
                                         flags=cv2.INTER_LINEAR, borderValue=0)
        return shifted, shifted_heatmap
    return shifted

def gp_fit_predict(train_xy, train_val, query_xy, lr=0.1, steps=150, use_cuda=None):
    """
    Fit a GP and predict at query points.
    Inputs are in 2D coordinates (x,y) for both train and query.
    Returns preds_mean: np.ndarray of shape (Q,).
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")

    train_x_t = torch.from_numpy(train_xy).double().to(device)     # (N,2)
    train_y_t = torch.from_numpy(train_val.reshape(-1)).double().to(device)  # (N,)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double().to(device)
    model = GPModel(train_x_t, train_y_t, likelihood).double().to(device)

    model.train(); likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        output = model(train_x_t)
        loss = -mll(output, train_y_t)  # type: ignore
        total_loss = loss.sum()
        total_loss.backward()
        optimizer.step()

    model.eval(); likelihood.eval()
    query_x_t = torch.from_numpy(query_xy).double().to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(query_x_t))
        preds_mean = preds.mean.cpu().detach().numpy()
    return preds_mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RSSI heatmap from real-world data.")
    parser.add_argument("--exp", type=int, default=EXP, help="Experiment number for data.")
    parser.add_argument("--size", type=int, default=SIZE, help="Output heatmap size (pixels).")
    parser.add_argument("--padding", type=int, default=PADDING, help="Padding around the contour.")
    parser.add_argument("--output_dir", type=str, default=f"{ROOT}/Processed_Dataset", help="Directory to save output files.")
    parser.add_argument("--data_type", type=str, choices=["imu", "gmap"], default="gmap",
                        help="Type of data to process: 'imu' for IMU data, 'gmap' for ground map data.")
    parser.add_argument("--iterations", type=int, default=150, help="Number of iterations for GP training.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for GP optimizer.")
    args = parser.parse_args()

    path = os.path.join(ROOT, f"RW_raw_data_{args.data_type}", f"exp{args.exp}.mat")
    data = loadmat(path)
    new_array = data['newArr']
    img = rasterize_points(new_array, SIZE)

    # get contour of the centered images with 10 pixels padding
    contour = create_out_contour(img)
    min_x, min_y = np.min(contour, axis=0)
    max_x, max_y = np.max(contour, axis=0)

    bound = (min_x, min_y), (max_x, max_y)
    min_x -= PADDING
    min_y -= PADDING
    max_x += PADDING
    max_y += PADDING
    # print(f"Contour bounding box: {bound}")

    # gaussian process fitting
    xs = np.arange(max(0, min_x), min(SIZE - 1, max_x) + 1)
    ys = np.arange(max(0, min_y), min(SIZE - 1, max_y) + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    query_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)

    train_xy = new_array[:, :2].astype(np.float64)
    train_val = new_array[:, 2].astype(np.float64)

    preds = gp_fit_predict(train_xy, train_val, query_xy, lr=0.1, steps=150)
    heatmap = np.zeros((SIZE, SIZE))
    heatmap[grid_y, grid_x] = preds.reshape(grid_x.shape)

    max_val = np.max(heatmap)
    heatmap[min_y:max_y, min_x] = max_val
    heatmap[min_y:max_y, max_x] = max_val
    heatmap[min_y, min_x:max_x] = max_val
    heatmap[max_y, min_x:max_x] = max_val

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
    for pts in np.indices((SIZE, SIZE)).reshape(2, -1).T:
        if not (min_x-1 < pts[0] < max_x+1 and min_y-1 < pts[1] < max_y+1):
            heatmap[pts[1], pts[0]] = [0, 0, 0]

    # flip and center the heatmap and image1
    heatmap = cv2.flip(heatmap, 0)
    img = cv2.flip(img, 0)
    img, heatmap = center_image_by_bbox(img, heatmap)
    # cv2.imshow("Heatmap 1 (flipped and centered)", heatmap)
    # cv2.imshow("Image 1 (flipped and centered)", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # save heatmap and trajectory image
    output_dir = os.path.join(args.output_dir, f"sexpe_exp{args.exp}_{args.data_type}")
    os.makedirs(output_dir, exist_ok=True)

    # condition 1 and 2 folders
    out_heatmap_file = os.path.join(output_dir, "Condition_1")
    out_trajectory_file = os.path.join(output_dir, "Condition_2")
    os.makedirs(out_heatmap_file, exist_ok=True)
    os.makedirs(out_trajectory_file, exist_ok=True)
    heatmap_file = os.path.join(out_heatmap_file, f"0.png")
    trajectory_file = os.path.join(out_trajectory_file, f"0.png")
    cv2.imwrite(heatmap_file, heatmap)
    cv2.imwrite(trajectory_file, img)
    print(f"Heatmap saved to {heatmap_file}")
    print(f"Trajectory image saved to {trajectory_file}")
    print("Heatmap generation completed.")