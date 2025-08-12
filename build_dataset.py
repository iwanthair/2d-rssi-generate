import os
import numpy as np
import shutil

# DATA_BASE = "HouseExpo/Dataset_Scale100_SEPE"
DATA_BASE = "HouseExpo/Dataset_Scale100_SExPE"
SOURCES = "HouseExpo/GroundTruth_Scale100"
TRAN_TEST_SPLIT = 0.75
HEATMAP_NAME = "Heatmaps_SExPE"

if __name__ == "__main__":

    os.makedirs(DATA_BASE, exist_ok=True)
    # read source dix files
    source_files = os.listdir(SOURCES)
    source_txt = os.path.join(SOURCES, "id2idx.txt")
    idx_txt = os.path.join(DATA_BASE, "id2idx.txt")
    idx_dict = {}
    with open(source_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            idx, id = line.strip().split()
            idx_dict[idx] = id

    # filter out sepecfic idx in the dict
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
    # not include the idx in the filtered_idx
    filtered_idx_dict = {k: v for k, v in idx_dict.items() if k not in filtered_idx}
    # write the filtered idx dict to the idx_txt
    length = 0
    with open(idx_txt, 'w') as f:
        for idx, id in filtered_idx_dict.items():
            f.write(f"{idx} {id}\n")
            length += 1
    print(f"Wrote {length} filtered IDs to {idx_txt}.")

    # combine the values with same keys in the filtered_idx_dict
    combined_dict = {}
    for idx, id in filtered_idx_dict.items():
        if id not in combined_dict:
            combined_dict[id] = []
        combined_dict[id].append(idx)

    print(f"Combined {len(combined_dict)} unique IDs with multiple indices.")
    print(list(combined_dict.items())[:5])

    # random split the filtered idx dict into train and test sets
    seed = 42
    np.random.seed(seed)
    keys = list(combined_dict.keys())
    np.random.shuffle(keys)
    train_keys = keys[:int(len(keys) * TRAN_TEST_SPLIT)]
    test_keys = keys[int(len(keys) * TRAN_TEST_SPLIT):]
    print(f"Split {len(keys)} keys into {len(train_keys)} train and {len(test_keys)} test.")
    print(f"Train keys: {train_keys[:5]}")
    print(f"Test keys: {test_keys[:5]}")

    # copy and move the train and test imgs
    train_path = os.path.join(DATA_BASE, "train")
    test_path = os.path.join(DATA_BASE, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    out_test_heatmap_file = os.path.join(test_path, "Condition_1")
    out_test_floorpan_file = os.path.join(test_path, "Target")
    out_test_trajectory_file = os.path.join(test_path, "Condition_2")
    out_train_heatmap_file = os.path.join(train_path, "Condition_1")
    out_train_floorpan_file = os.path.join(train_path, "Target")
    out_train_trajectory_file = os.path.join(train_path, "Condition_2")
    os.makedirs(out_test_heatmap_file, exist_ok=True)
    os.makedirs(out_test_floorpan_file, exist_ok=True)
    os.makedirs(out_test_trajectory_file, exist_ok=True)
    os.makedirs(out_train_heatmap_file, exist_ok=True)
    os.makedirs(out_train_floorpan_file, exist_ok=True)
    os.makedirs(out_train_trajectory_file, exist_ok=True)

    for key in train_keys:
        idxs = combined_dict[key]
        for idx in idxs:
            in_heatmap_file = os.path.join(SOURCES, HEATMAP_NAME, f"{idx}.png")
            in_floorpan_file = os.path.join(SOURCES, "FloorPlan", f"{idx}.png")
            in_trajectory_file = os.path.join(SOURCES, "Trajectory", f"{idx}.png")

            if os.path.exists(in_heatmap_file):
                shutil.copy(in_heatmap_file, out_train_heatmap_file)
            if os.path.exists(in_floorpan_file):
                shutil.copy(in_floorpan_file, out_train_floorpan_file)
            if os.path.exists(in_trajectory_file):
                shutil.copy(in_trajectory_file, out_train_trajectory_file)

    for key in test_keys:
        idxs = combined_dict[key]
        for idx in idxs:
            in_heatmap_file = os.path.join(SOURCES, HEATMAP_NAME, f"{idx}.png")
            in_floorpan_file = os.path.join(SOURCES, "FloorPlan", f"{idx}.png")
            in_trajectory_file = os.path.join(SOURCES, "Trajectory", f"{idx}.png")

            if os.path.exists(in_heatmap_file):
                shutil.copy(in_heatmap_file, out_test_heatmap_file)
            if os.path.exists(in_floorpan_file):
                shutil.copy(in_floorpan_file, out_test_floorpan_file)
            if os.path.exists(in_trajectory_file):
                shutil.copy(in_trajectory_file, out_test_trajectory_file)


    print(f"Copied train files to {train_path} and test files to {test_path}.")
    print("Dataset generation completed.")