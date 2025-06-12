from PIL import Image
import numpy as np
from scipy.ndimage import label
import os
import pickle

DIR_PATH = "../media/satellite/"


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


def build_segments_representations(path):
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
        
        
def scale_segments(all_segments, a):
    """
    DEPRECATED! This method produces segment descriptions with pixels in several
    segments, and repeated pixels inside a same segment. 
    """
    all_segments_a = []

    for segment in all_segments:
        scaled_segment = [
            (int(row * a), int(col * a))
            for row, col in segment
        ]
        all_segments_a.append(scaled_segment)

    return all_segments_a


def scale_segments_to_map(all_segments, a, output_shape=None):
    # First, scale and record where each pixel wants to go
    pixel_to_segment = {}
    
    for segment_id, segment in enumerate(all_segments):
        for row, col in segment:
            scaled_row = int(row * a)
            scaled_col = int(col * a)
            key = (scaled_row, scaled_col)

            # Only assign if pixel hasn't been assigned yet
            if key not in pixel_to_segment:
                pixel_to_segment[key] = segment_id

    # Determine output shape if not provided
    if output_shape is None:
        max_row = max(r for r, _ in pixel_to_segment.keys())
        max_col = max(c for _, c in pixel_to_segment.keys())
        output_shape = (max_row + 1, max_col + 1)

    # Initialize segmentation map
    seg_map = np.full(output_shape, -1, dtype=int)

    for (r, c), seg_id in pixel_to_segment.items():
        seg_map[r, c] = seg_id

    return seg_map


def rescale_images(path, a, dim_txt):
    filenames = [os.path.join(path, f) 
                 for f in os.listdir(path) 
                 if f.lower().endswith('_data.pkl') and not(f.lower().endswith('x400_data.pkl')) 
                 and os.path.isfile(os.path.join(path, f))]
    
    for file_path in filenames:
        print(file_path)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        rescaled_data = { "loaded_areas": scale_segments_to_map(data["loaded_areas"], a),
                          "data": data["data"]}
        pkl_path = f'{file_path[:-9]}-{dim_txt}_data.pkl'
        with open(pkl_path, "wb") as f:
            pickle.dump(rescaled_data, f)
    

def deduplicate_px_segments(px_segments):
    return [list(set(segment)) for segment in px_segments]

    
def compute_pixel_segment_count(px_segments):
    # Determine image size by finding max row/col
    max_row = max(row for segment in px_segments for row, _ in segment)
    max_col = max(col for segment in px_segments for _, col in segment)
    H, W = max_row + 1, max_col + 1

    count_matrix = np.zeros((H, W), dtype=int)

    for segment in px_segments:
        for row, col in segment:
            count_matrix[row, col] += 1

    return count_matrix





#extract_segments(DIR_PATH)
#build_segments_representations(DIR_PATH)

rescale_images(DIR_PATH, 600/5616, "600x400")