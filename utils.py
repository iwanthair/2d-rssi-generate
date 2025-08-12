import cv2
import numpy as np


def pad_to_square(img, pad_value=0, padding=100, size=None):
    long_side = max(img.shape[0], img.shape[1]) + padding
    height = long_side - img.shape[0]
    width = long_side - img.shape[1]

    img = cv2.copyMakeBorder(
        img,
        height // 2, height - height // 2,
        width // 2, width - width // 2,
        cv2.BORDER_CONSTANT,
        255,
        value=pad_value
    )

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST) if size is not None else img
    return img, long_side


def grad_img(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 2, 0, ksize=5)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 2, ksize=5)
    grad = grad_x + grad_y + cv2.Sobel(blur, cv2.CV_64F, 2, 2, ksize=5)
    # grad = np.sqrt(grad_x**2 + grad_y**2)
    grad[grad > 5] = 255
    grad[grad < 5] = 0
    return grad.astype('uint8')


def create_in_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_pts = np.vstack(contours).squeeze() if contours else np.empty((0, 2))
    return cnt_pts

def create_out_contour(img):
    img = cv2.Canny(img, 50, 100)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_pts = np.vstack(contours).squeeze() if contours else np.empty((0, 2))
    return cnt_pts


def downsample(img, scale=0.5):
    h, w = img.shape[:2]
    new_h, new_w = max(int(h * scale), 1), max(int(w * scale), 1)
    small = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )
    return small


def sample_random_pixels(mask, num_pts, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mask_pos = np.argwhere(mask == 255) # shape (M, 2)
    M = mask_pos.shape[0]
    if M < num_pts:
        raise ValueError(f"Not enough white pixels in the mask: {M} found, {num_pts} requested.")

    choices = np.random.choice(M, size=num_pts, replace=False)
    return mask_pos[choices]


def multi_wall_model(distance, L0, na, n, Lw):

    path_loss = L0 + 10 * na * np.log10(distance) + n * Lw

    return path_loss


def bresenham_line(x0,y0,x1,y1):
    pts = []
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0<x1 else -1
    sy = 1 if y0<y1 else -1
    err = dx-dy
    while True:
        pts.append((x0,y0))
        if x0==x1 and y0==y1: break
        e2 = err*2
        if e2>-dy:
            err -= dy
            x0 += sx
        if e2<dx:
            err += dx
            y0 += sy
    return pts


def check_wall_number(floor_plan, pts):
    vals = [floor_plan[y,x] for x,y in pts]
    diffs = np.abs(np.diff([v/255 for v in vals]))
    return int(diffs.sum() / 2)  # Each wall is counted twice

