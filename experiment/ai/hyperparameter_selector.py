import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random

class HyperparameterSelection:
    def __init__(self, X):
        self.X = X
        self.scaler = StandardScaler().fit(X)
        self.X_scaled = self.scaler.transform(X)
        self.current_step = 0

    def next_step(self):
        """Perform one step of segmentation with increasing complexity."""
        self.current_step += 1
        n_clusters = 1 + self.current_step  # Just an example progression

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.X_scaled)
        
        print("worked")

        return labels



    #segmentation_data_path = settings.MEDIA_ROOT + f'/satellite/{selected_img}-600x400_data.pkl'
    
    # px_segment = data['px_segment']
    
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    
    # # Proposing a clustering
    
    # N_CLUSTERS = 3
    
    # kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    # cluster_labels = kmeans.fit_predict(X_scaled)  # shape (N,)

    # # 2. Create a blank RGB image
    # max_row = max(segment[0] for segment in px_segment)
    # max_col = max(segment[1] for segment in px_segment)
    # H, W = max_row + 1, max_col + 1

    # seg_image = np.zeros((H, W, 3), dtype=np.uint8)

    # # 3. Generate distinct random colors for each cluster
    # colors = {
    #     label: tuple(random.randint(0, 255) for _ in range(3))
    #     for label in range(N_CLUSTERS)
    # }

    # # 4. Fill segments with their cluster's color
    # for seg_index, pixels in enumerate(px_segment):
    #     color = colors[cluster_labels[seg_index]]
    #     print(pixels)
    #     for row, col in pixels:
    #         seg_image[row, col] = color
    
    
    
    
    
    
        # # Process clustering

        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        # N_CLUSTERS = 3
        # kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
        # cluster_labels = kmeans.fit_predict(X_scaled)

        # # Infer shape from px_segment
        # max_row = max(row for segment in px_segment for row, _ in segment)
        # max_col = max(col for segment in px_segment for _, col in segment)
        # H, W = max_row + 1, max_col + 1
        # seg_image = np.zeros((H, W, 3), dtype=np.uint8)

        # # Color clusters
        # colors = {
        #     label: tuple(random.randint(0, 255) for _ in range(3))
        #     for label in range(N_CLUSTERS)
        # }

        # for seg_index, pixels in enumerate(px_segment):
        #     color = colors[cluster_labels[seg_index]]
        #     for row, col in pixels:
        #         seg_image[row, col] = color

        # # Save overlay image to temp
        # from datetime import datetime
        # filename = f"{selected_img}_segmentation_overlay.png"
        # save_path = os.path.join(settings.MEDIA_ROOT, 'temp', filename)
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Image.fromarray(seg_image).save(save_path)

        # Return media URL
        #image_url = settings.MEDIA_URL + f'temp/{filename}'
        #return JsonResponse({'status': 'ok', 'url': image_url})