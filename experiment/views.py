from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.conf import settings
from django.shortcuts import redirect
import os
import pickle
from django.views.decorators.csrf import csrf_exempt
import json
#from django.contrib.sessions.models import Session

import numpy as np
import random
from PIL import Image
from .ai.hyperparameter_selector import KMeansOptimizer, KMeansMahalanobisOptimizer
from .ai.hyperparameter_selector import SpectralClusteringOptimizer, SpectralMahalanobisOptimizer
from .ai.hyperparameter_selector import DBSCANOptimizer



OPACITY = 0.5

IMAGES = {"Castle": "2012-04-26-Muenchen-Tunnel_4K0G0080",
          "Commercial area": "2012-04-26-Muenchen-Tunnel_4K0G0020",
          "Farm": "2012-04-26-Muenchen-Tunnel_4K0G0010", 
          "Railway track": "2012-04-26-Muenchen-Tunnel_4K0G0100",
          "Urban area 1": "2012-04-26-Muenchen-Tunnel_4K0G0051",
          "Urban area 2": "2012-04-26-Muenchen-Tunnel_4K0G0070",
          "Urban area 3": "2012-04-26-Muenchen-Tunnel_4K0G0090",
          }

session_data = {}


def save_temp_image(img, filename):
    temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    save_path = os.path.join(temp_dir, filename)
    Image.fromarray(img).save(save_path)

    return os.path.join(settings.MEDIA_URL, "temp", filename)


def get_cluster_colors(n):
    base_colors = [
        (255, 0, 0),      # Red
        (0, 128, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 165, 0),    # Orange
        (128, 0, 128),    # Purple
        (0, 255, 255),    # Cyan
        (255, 255, 0),    # Yellow
        (255, 192, 203),  # Pink
        (150, 75, 0),     # Brown
        (128, 128, 128),  # Gray
    ]
    random.seed(42)
    colors = base_colors.copy()
    while len(colors) < n:
        colors.append(tuple(random.randint(0, 255) for _ in range(3)))
    return colors[:n]


def image_selection(request):
    template = loader.get_template("experiment/image_selection.html")
    
    existing_images = { img_name : IMAGES[img_name]
                        for img_name in IMAGES 
                        if os.path.isfile(settings.MEDIA_ROOT + f'/satellite/{IMAGES[img_name]}-600x400_data.pkl')}
    
    context = {'images': existing_images, 
               'MEDIA_URL': settings.MEDIA_URL, }
    return HttpResponse(template.render(context, request))



###############################################################################
## Main page: interactive segmentation using clustering
###############################################################################


def index(request):
    selected_img = request.GET.get('image', 'source-image')  # without extension
    image_url = settings.MEDIA_URL + f'satellite/{selected_img}.jpg'
    
    session_key = request.session.session_key or request.session.create()
    
    try:
        X, M_segments = load_segmentation_from_path(selected_img)

        # Force reinitialization of state every time index is called
        session_data[session_key] = {
            'selector': SpectralMahalanobisOptimizer(X),
            'M_segments': M_segments
        }
        request.session['current_image'] = selected_img
    
    except Exception as e:
        return HttpResponse(f"Failed to load segmentation: {str(e)}")
    
    context = {
        'image_url': image_url,
        'background_image_url': image_url,
        #'overlay_image_url': segmentation_url,
        'overlay_image_url': '',
        'overlay_opacity': OPACITY,
    }

    template = loader.get_template("experiment/index.html")
    return HttpResponse(template.render(context, request))




def load_segmentation_from_path(selected_img):
    segmentation_data_path = os.path.join(settings.MEDIA_ROOT, f'satellite/{selected_img}-600x400_data.pkl')
    
    with open(segmentation_data_path, "rb") as f:
        data = pickle.load(f)

    X = data['data']
    M_segments = data['loaded_areas']
    
    return X, M_segments

    

    
# def load_segmentation(request):
#     print("Running load_segmentation")
#     selected_img = request.GET.get('image', 'source-image')
#     session_key = request.session.session_key or request.session.create()
    
#     segmentation_data_path = os.path.join(settings.MEDIA_ROOT, f'satellite/{selected_img}-600x400_data.pkl')

#     try:
#         X, M_segments = load_segmentation_from_path(segmentation_data_path)

#         request.session['current_image'] = selected_img
#         session_data[session_key] = {
#             'selector': KMeansOptimizer(X),
#             'M_segments': M_segments
#         }

#         return JsonResponse({'status': 'ok'})
#     except Exception as e:
#         return JsonResponse({'status': 'error', 'message': str(e)})
    
    



@csrf_exempt
def next_step(request):
    selected_img = request.GET.get('image', 'source-image')
    session_key = request.session.session_key or request.session.create()
    
    if session_key not in session_data:
        return JsonResponse({'status': 'error', 'message': 'Session not initialized. Please reload the page.'})

    # Get session state
    state = session_data[session_key]
    
    selector = state['selector']
    M_segments = state['M_segments']

    H, W = M_segments.shape
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            matrix = np.array(data.get("matrix"))
            #print("Received matrix:", matrix)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid matrix data'})
    
    
    labels = selector.next_step(matrix / 100)

    # Generate segmentation image
    seg_image = np.zeros((H, W, 3), dtype=np.uint8)
    unique_cluster_labels = set(labels)
    colors = get_cluster_colors(len(unique_cluster_labels))

    for row in range(H):
        for col in range(W):
            seg_id = M_segments[row, col]
            if seg_id >= 0:
                cluster_label = labels[seg_id]
                seg_image[row, col] = colors[cluster_label]

    # Save image
    filename = f"{selected_img}_step_{selector.current_step}.png"
    save_path = os.path.join(settings.MEDIA_ROOT, 'temp', filename)
    Image.fromarray(seg_image).save(save_path)


    return JsonResponse({
        'status': 'ok',
        'url': settings.MEDIA_URL + f'temp/{filename}',
        'num_clusters': len(unique_cluster_labels),
        'colors': [colors[label] for label in unique_cluster_labels],
        'step': selector.current_step,  # <- Added
    })




def summary(request):
    segmentation_file = request.GET.get("segmentation")
    if not segmentation_file:
        return redirect("experiment:select")  # fallback to home if no image provided

    segmentation_url = settings.MEDIA_URL + f'temp/{segmentation_file}'
    #segmentation_url = settings.MEDIA_URL + f'satellite/{selected_img}.png'

    selected_img = segmentation_file.split("_step")[0]
    background_url = settings.MEDIA_URL + f'satellite/{selected_img}-600x400.jpg'
    print(background_url, segmentation_url)
    context = {
        "segmentation_url": segmentation_url,
        "background_url": background_url,
        "overlay_opacity": OPACITY,
    }

    template = loader.get_template("experiment/summary.html")
    return HttpResponse(template.render(context, request))