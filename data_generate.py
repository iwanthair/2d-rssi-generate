import os
import argparse
import numpy as np
import glob
import cv2
import bresenham

from utils import pad_to_square, grad_img, check_wall_number, multi_wall_model

NUM_SAMPLES = 2000

TARGET_SIZE = 256
PAD_VALUE = 0
BOARD_SIZE = 100
LB = 40.0
NA = 2.0
LW = 15
SEED = [0, 42, 84]
NUM_RECEIVERS = 800

FLOOR_PLAN_PATH = "FloorPlan"
TRAJECTORY_PATH = "Trajectory"
RECEIVERS_PATH = "Receivers"
RSSI_PATH = "RSSI"
# may not be used
NUM_WALLS_PATH = "NumWalls"
TRANSMITTER_PATH = "Transmitter"
ID2IDX = "id2idx.txt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset with padded images.")
    parser.add_argument("--input_dir", type=str, default="HouseExpo/HouseExpoPNG/png", required=False, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, default="HouseExpo/GroundTruth_Scale100/", required=False, help="Directory to save all folders.")
    parser.add_argument("--selected_ids", type=str, default="HouseExpo/valid_map_ids_26982.txt", required=False, help="File containing selected map IDs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    floor_plan_path = os.path.join(args.output_dir, FLOOR_PLAN_PATH)
    trajectory_path = os.path.join(args.output_dir, TRAJECTORY_PATH)
    receivers = os.path.join(args.output_dir, RECEIVERS_PATH)
    rssi_path = os.path.join(args.output_dir, RSSI_PATH)
    num_walls_path = os.path.join(args.output_dir, NUM_WALLS_PATH)
    transmitter_path = os.path.join(args.output_dir, TRANSMITTER_PATH)
    # id2idx_path = os.path.join(args.output_dir, ID2IDX)
    os.makedirs(floor_plan_path, exist_ok=True)
    os.makedirs(trajectory_path, exist_ok=True)
    os.makedirs(receivers, exist_ok=True)
    os.makedirs(rssi_path, exist_ok=True)
    os.makedirs(num_walls_path, exist_ok=True)
    os.makedirs(transmitter_path, exist_ok=True)
    # os.makedirs(id2idx_path, exist_ok=True)
    id2idx_path = os.path.join(args.output_dir, ID2IDX)
    with open(id2idx_path, 'w') as f:
        f.write("")

    # Read selected IDs
    with open(args.selected_ids, 'r') as f:
        selected_ids = [line.strip() for line in f.readlines()]
    if NUM_SAMPLES is not None:
        selected_ids = selected_ids[:NUM_SAMPLES]
    print(f"Selected {len(selected_ids)} IDs from {args.selected_ids}.")
    
    image_files = glob.glob(os.path.join(args.input_dir, "*.png"))
    print(f"Found {len(image_files)} image files in {args.input_dir}.")
    # print(image_files[:5])  # Print first 5 files for debugging

    image_files = [f for f in image_files if os.path.basename(f).split('.')[0] in selected_ids]
    print(f"Filtered to {len(image_files)} image files based on selected IDs.")
    # print(image_files[:5])  # Print first 5 files for debugging

    count = 0
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask, long_side = pad_to_square(image, pad_value=PAD_VALUE, padding=BOARD_SIZE, size=TARGET_SIZE)
        scale =  (long_side / TARGET_SIZE) / 100.0
        mask = mask.astype(np.uint8)

        floor_plan = grad_img(mask)

        # get all pixels that are 255 in mask, and not equal to 255 in  floor_plan
        avi_space = np.argwhere((mask == 255) & (floor_plan != 255))

        # randomly select NUM_RECEIVERS in the available space
        if len(avi_space) < NUM_RECEIVERS:
            print(f"Not enough available space: {NUM_RECEIVERS} requested. Now using all available {len(avi_space)} space.")
            num = len(avi_space) - 1
        else:
            num = NUM_RECEIVERS

        # set seed
        for seed in SEED:
            np.random.seed(seed)
            selected_indices = np.random.choice(len(avi_space), size=num+1, replace=False)
            receiver_coords = avi_space[selected_indices[:-1]]  # Exclude the last point for transmitter
            trajectory_img = np.zeros_like(floor_plan, dtype=np.uint8)

            transmitter_coords = avi_space[selected_indices[-1]]  # Last point is the transmitter
            y_tx, x_tx = transmitter_coords[0], transmitter_coords[1]

            # count the wall number between each selected point and the contour
            data_dict = {
                "transmitter": (y_tx, x_tx),
                "receivers": [],
                # "walls": [],
                "RSSI": []
            }
            # plane = floor_plan.copy()
            for coord in receiver_coords:
                # append to trajectory image
                y_rx, x_rx = coord
                trajectory_img[y_rx, x_rx] = 255
                path = list(bresenham.bresenham(x_tx, y_tx, x_rx, y_rx))
                wall_count = check_wall_number(floor_plan, path)

                # calculate the rssi
                distance = scale * np.sqrt((x_rx - x_tx) ** 2 + (y_rx - y_tx) ** 2) + 1e-3
                rssi = multi_wall_model(distance, LB, NA, wall_count, LW)

                # data_dict["walls"].append(wall_count)
                data_dict["RSSI"].append(rssi)
                data_dict["receivers"].append((y_rx, x_rx))

            #     # Draw the path on the image
            #     print(f"Wall count between transmitter and receiver at ({x_rx}, {y_rx}): {wall_count}")
            #     print(f"Coordinates: ({x_rx}, {y_rx}), Distance: {distance:.2f}, Wall Count: {wall_count}, RSSI: {rssi:.2f} dB")
            #     for point in path:
            #         cv2.circle(plane, (point[0], point[1]), 1, (128), -1)
            # cv2.imshow("Path", trajectory_img)
            # cv2.waitKey(0)
            # cv2.imshow("Floor Plan", floor_plan)
            # cv2.waitKey(0)

            # Save the data
            floor_plan_file = os.path.join(floor_plan_path, f"{count:04d}.png")
            trajectory_file = os.path.join(trajectory_path, f"{count:04d}.png")
            receivers_file = os.path.join(receivers, f"{count:04d}.npy")
            rssi_file = os.path.join(rssi_path, f"{count:04d}.npy")
            # num_walls_file = os.path.join(num_walls_path, f"{count:04d}.npy")
            # transmitter_file = os.path.join(transmitter_path, f"{count:04d}.npy")
            

            cv2.imwrite(floor_plan_file, floor_plan)
            cv2.imwrite(trajectory_file, trajectory_img)
            np.save(receivers_file, np.array(data_dict["receivers"]))
            np.save(rssi_file, np.array(data_dict["RSSI"]))
            # np.save(num_walls_file, np.array(data_dict["walls"]))
            # np.save(transmitter_file, np.array(data_dict["transmitter"]))
            
            # save id2idx mapping
            with open(os.path.join(args.output_dir, ID2IDX), 'a') as f:
                f.write(f"{count:04d} {os.path.basename(image_file).split('.')[0]}\n")

            print(f"Processed {count+1}/{len(SEED)*len(image_files)}")
            count += 1
        # break
