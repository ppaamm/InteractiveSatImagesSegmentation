from PIL import Image
import numpy as np
from scipy.ndimage import label
import os
import pickle


path = "../media/satellite/"


def extract_areas_from_segmentation(png_path):
    # Load the image and convert to RGB
    img = Image.open(png_path).convert('RGB')
    arr = np.array(img)

    # Get unique colors in the image
    colors = np.unique(arr.reshape(-1, 3), axis=0)

    all_areas = []

    for color in colors:
        # Create a binary mask where this color is present
        mask = np.all(arr == color, axis=-1)

        # Label connected components within this mask
        labeled_array, num_features = label(mask)

        for label_id in range(1, num_features + 1):
            # Find pixel coordinates for this label
            coords = np.argwhere(labeled_array == label_id)
            area_pixels = [tuple(coord) for coord in coords]  # (row, col)
            all_areas.append(area_pixels)

    return all_areas



def extract_segments(path):
    filenames = [os.path.join(path, f) 
                 for f in os.listdir(path) 
                 if f.lower().endswith('.png') and os.path.isfile(os.path.join(path, f))]
    
    for file_path in filenames:
        print(file_path)
        print("... extracting areas")
        areas = extract_areas_from_segmentation(file_path)
        pkl_path = file_path[:-3] + 'pkl'
        print("... saving files:", pkl_path)
        with open(pkl_path, "wb") as f:
            pickle.dump(areas, f)


def extract_rgb_values(img, px_segment):
    return np.array([img[row, col] for row, col in px_segment])

def extract_segment_statistics(px_segment, rgb_segment):
    # RGB stats
    rgb_means = np.mean(rgb_segment, axis=0)
    rgb_std = np.std(rgb_segment, axis=0)
    
    # Gray level stats
    gray_level = np.array([np.mean(rgb_segment), np.std(rgb_segment)])
    
    # Segment size
    delta_x = max(px_segment, key=lambda t: t[0])[0] - min(px_segment, key=lambda t: t[0])[0]
    delta_y = max(px_segment, key=lambda t: t[1])[1] - min(px_segment, key=lambda t: t[1])[1]
    sizes = np.array([len(px_segment), delta_x, delta_y])
    
    # Contatenation
    return np.concatenate([rgb_means, rgb_std, gray_level, sizes])



extract_segments(path)


filenames = [os.path.join(path, f) 
             for f in os.listdir(path) 
             if f.lower().endswith('.pkl') and os.path.isfile(os.path.join(path, f))]

for file_path in filenames:
    jpg_path = file_path[:-3] + 'jpg'
    img = Image.open(jpg_path).convert('RGB')
    arr = np.array(img)
    
    print(jpg_path)
    
    with open(file_path, "rb") as f:
        loaded_areas = pickle.load(f)
    
    X = []
    for px_segment in loaded_areas:
        rgb_segment = extract_rgb_values(arr, px_segment)
        vector_description = extract_segment_statistics(px_segment, rgb_segment)
        X.append(vector_description)
    X = np.stack(X)
    
    data = {"loaded_areas": loaded_areas, 
            "data": X}
    
    pkl_path = file_path[:-4] + "_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)