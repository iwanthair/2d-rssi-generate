import os
import numpy as np
import shutil

# DATA_PATH = "HouseExpo/Dataset_Scale100_SExPE"
DATA_PATH = "HouseExpo/Dataset_Scale100_SExPE"
SELLECT_NUM = 50
SEED = 18


if __name__ == "__main__":
    train_folder = os.path.join(DATA_PATH, f"Selected_{SELLECT_NUM}_train")
    os.makedirs(train_folder, exist_ok=True)
    test_folder = os.path.join(DATA_PATH, f"Selected_{SELLECT_NUM}_test")
    os.makedirs(test_folder, exist_ok=True)

    folders = ["Condition_1", "Target", "Condition_2"]
    for folder in folders:
        os.makedirs(os.path.join(train_folder, folder), exist_ok=True)
        os.makedirs(os.path.join(test_folder, folder), exist_ok=True)


    # read ids from the folder
    train_id_list = os.listdir(os.path.join(DATA_PATH, "train", "Condition_1"))
    test_id_list = os.listdir(os.path.join(DATA_PATH, "test", "Condition_1"))

    print(f"Total {len(train_id_list)} train ids and {len(test_id_list)} test ids found.")

    # randomly select ids
    np.random.seed(SEED)
    selected_train_ids = np.random.choice(train_id_list, SELLECT_NUM, replace=False)
    selected_test_ids = np.random.choice(test_id_list, SELLECT_NUM, replace=False)
    print(f"Selected {len(selected_train_ids)} train ids and {len(selected_test_ids)} test ids.")

    # copy the selected ids to the new folder
    count = 0

    for id in selected_train_ids:
        for folder in folders:
            src_file = os.path.join(DATA_PATH, "train", folder, id)
            dst_file = os.path.join(train_folder, folder, id)
            # copy the file if it exists
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
        count += 1
    print(f"Copied selected train files to {train_folder}. Total {count} files copied.")
    count = 0
    for id in selected_test_ids:
        for folder in folders:
            src_file = os.path.join(DATA_PATH, "test", folder, id)
            dst_file = os.path.join(test_folder, folder, id)
            # copy the file if it exists
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
        count += 1
    print(f"Copied selected test files to {test_folder}. Total {count} files copied.")
    print(f"Copied selected train files to {train_folder} and test files to {test_folder}.")
    
    print("Dataset selection completed.")