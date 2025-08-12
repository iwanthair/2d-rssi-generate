import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, \
    ExpSineSquared, WhiteKernel, Matern, RationalQuadratic


def train_gp_rssi(X_train, y_train):
    # define the kernel
    k_lin  = DotProduct(
        sigma_0=1.0, sigma_0_bounds=(1e-3, 1e2)
    )

    k_rbf_long  = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 20.0))
    k_rbf_short = ConstantKernel(5.0, (1e-3, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 10.0))

    k_per  = ConstantKernel(5.0, (1e-3, 1e3)) * ExpSineSquared(
        length_scale=5.0,
        periodicity=3.0,
        length_scale_bounds=(1e-1, 30.0),
        periodicity_bounds=(1.0, 50.0),
    )

    k_noise = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))

    k_matern = Matern(length_scale=0.5, length_scale_bounds=(1e-2, 5.0), nu=1.5)

    kernel = k_rbf_short + k_per + k_noise
    print("Initial kernel:", kernel)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0,
        normalize_y=True,
        n_restarts_optimizer=10,
        optimizer="fmin_l_bfgs_b"
    )
    gp.fit(X_train, y_train)
    print("Optimized kernel:", gp.kernel_)
    return gp


def plot_rssi_heatmap(gp, W, H, grid_res=256, txy=None):

    xs = np.arange(W)
    ys = np.arange(H)
    XX, YY = np.meshgrid(xs, ys)
    Xgrid = np.column_stack([XX.ravel(), YY.ravel()])  # (H*W, 2)
    Xgrid = Xgrid / 100
    print(f"Xgrid shape: {Xgrid.shape}, first 5 points:\n{Xgrid[:5]}")

    y_pred, y_std = gp.predict(Xgrid, return_std=True)
    print(f"Pred mean: {y_pred.mean():.2f}, std: {y_std.mean():.2f}")

    heatmap = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
    heatmap = heatmap.reshape(H, W)
    print(f"Predicted RSSI heatmap shape: {heatmap.shape}")
    import cv2
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    print(f"Heatmap shape: {heatmap.shape}")
    cv2.namedWindow("RSSI Heatmap", cv2.WINDOW_NORMAL)
    cv2.imshow("RSSI Heatmap", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return heatmap
