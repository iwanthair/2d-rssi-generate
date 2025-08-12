import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

JSON_PATH = "./HouseExpo/HouseExpoJSON/json"
ROOM_NUM_THRESHOLD = 5

def is_valid_map(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if data.get("room_num", 0) >= ROOM_NUM_THRESHOLD:
            return os.path.splitext(os.path.basename(json_path))[0]
    except Exception:
        return None
    
if __name__ == '__main__':
    json_dir = JSON_PATH
    print(f"Searching for JSON files in {json_dir}...")
    all_json_files = [
        os.path.join(json_dir, f)
        for f in os.listdir(json_dir)
        if f.endswith(".json")
    ]

    print(f"Totally {len(all_json_files)} JSON files found.")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(is_valid_map, all_json_files))

    # Filter out None results
    valid_map_ids = [mid for mid in results if mid is not None]

    print(f"{len(valid_map_ids)} maps with room_num equal and more than {ROOM_NUM_THRESHOLD} found.")

    np.savetxt(f"./HouseExpo/valid_map_ids_{len(valid_map_ids)}.txt", valid_map_ids, fmt="%s")
    print("Valid_map_ids.txt has been saved.")