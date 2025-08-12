import cv2
import os

data_path = "HouseExpo/GroundTruth_Scale100"
floor_plan_path = os.path.join(data_path, "FloorPlan")

id_list = os.listdir(floor_plan_path)

for count, id in enumerate(id_list):
    # check if the boundary of the img is white
    floor_plan_file = os.path.join(floor_plan_path, id)
    floor_plan_img = cv2.imread(floor_plan_file)
    # print(f"Image Shape: {floor_plan_img.shape}")

    # check the boundary pixels
    top_row = floor_plan_img[0, :, :]
    bottom_row = floor_plan_img[-1, :, :]
    left_col = floor_plan_img[:, 0, :]
    right_col = floor_plan_img[:, -1, :]

    if (top_row == 255).all() or (bottom_row == 255).all() or (left_col == 255).all() or (right_col == 255).all():
        print(f"Boundary of {id} contains white pixels.")
    # else:
    #     print(f"Boundary of {id} does not contain white pixels.")