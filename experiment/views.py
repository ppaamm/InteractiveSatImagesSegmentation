from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.conf import settings
import os
import pickle

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random


OPACITY = 0.5

IMAGES = {"Castle": "2012-04-26-Muenchen-Tunnel_4K0G0080",
          "Commercial area": "2012-04-26-Muenchen-Tunnel_4K0G0020",
          "Farm": "2012-04-26-Muenchen-Tunnel_4K0G0010", 
          "Railway track": "2012-04-26-Muenchen-Tunnel_4K0G0100",
          "Urban area 1": "2012-04-26-Muenchen-Tunnel_4K0G0051",
          "Urban area 2": "2012-04-26-Muenchen-Tunnel_4K0G0070",
          "Urban area 3": "2012-04-26-Muenchen-Tunnel_4K0G0090",
          }



def save_temp_image(img, filename):
    temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    save_path = os.path.join(temp_dir, filename)
    Image.fromarray(img).save(save_path)

    return os.path.join(settings.MEDIA_URL, "temp", filename)





def image_selection(request):
    template = loader.get_template("experiment/image_selection.html")
    
    existing_images = { img_name : IMAGES[img_name]
                        for img_name in IMAGES 
                        if os.path.isfile(settings.MEDIA_ROOT + f'/satellite/{IMAGES[img_name]}_data.pkl')}
    
    context = {'images': existing_images, 
               'MEDIA_URL': settings.MEDIA_URL, }
    return HttpResponse(template.render(context, request))



###############################################################################
## Main page: interactive segmentation using clustering
###############################################################################


def index(request):
    selected_img = request.GET.get('image', 'source-image')  # without extension
    image_url = settings.MEDIA_URL + f'satellite/{selected_img}.jpg'
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
    
    
    
    segmentation_url = settings.MEDIA_URL + f'satellite/{selected_img}.png'
    #segmentation_url = save_temp_image(seg_image, f'{selected_img}_seg.png')
    
    context = {
        'image_url': image_url,
        'background_image_url': image_url,
        'overlay_image_url': segmentation_url,
        'overlay_opacity': OPACITY,
    }

    template = loader.get_template("experiment/index.html")
    return HttpResponse(template.render(context, request))





def load_segmentation(request):
    selected_img = request.GET.get('image', 'source-image')
    segmentation_data_path = os.path.join(settings.MEDIA_ROOT, f'satellite/{selected_img}-600x400_data.pkl')

    try:
        with open(segmentation_data_path, "rb") as f:
            data = pickle.load(f)
        #X = data['data']
        #px_segments = data['loaded_areas']

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
        return JsonResponse({'status': 'ok'})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


    