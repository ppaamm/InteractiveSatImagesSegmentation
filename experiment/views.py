from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.conf import settings
import os
import pickle
from django.contrib.sessions.models import Session

import numpy as np
import random
from PIL import Image
from .ai.hyperparameter_selector import HyperparameterSelection


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
    segmentation_url = settings.MEDIA_URL + f'satellite/{selected_img}.png'
    #segmentation_url = save_temp_image(seg_image, f'{selected_img}_seg.png')
    
    session_key = request.session.session_key or request.session.create()
    
    try:
        selector, M_segments = load_segmentation_from_path(selected_img)

        # Force reinitialization of state every time index is called
        session_data[session_key] = {
            'selector': selector,
            'M_segments': M_segments
        }
        request.session['current_image'] = selected_img
    
    except Exception as e:
        return HttpResponse(f"Failed to load segmentation: {str(e)}")
    
    context = {
        'image_url': image_url,
        'background_image_url': image_url,
        'overlay_image_url': segmentation_url,
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
    
    return HyperparameterSelection(X), M_segments

    

    
def load_segmentation(request):
    selected_img = request.GET.get('image', 'source-image')
    session_key = request.session.session_key or request.session.create()
    
    segmentation_data_path = os.path.join(settings.MEDIA_ROOT, f'satellite/{selected_img}-600x400_data.pkl')

    try:
        X, M_segments = load_segmentation_from_path(segmentation_data_path)

        request.session['current_image'] = selected_img
        session_data[session_key] = {
            'selector': HyperparameterSelection(X),
            'M_segments': M_segments
        }

        return JsonResponse({'status': 'ok'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})
    
    



def next_step(request):
    selected_img = request.GET.get('image', 'source-image')
    session_key = request.session.session_key or request.session.create()
    
    if session_key not in session_data:
        return JsonResponse({'status': 'error', 'message': 'Session not initialized. Please reload the page.'})


    # # Check if session has state or image has changed
    # current_image = request.session.get('current_image')
    # if session_key not in session_data or current_image != selected_img:
    #     try:
    #         selector, M_segments = load_segmentation_from_path(selected_img)

    #         session_data[session_key] = {
    #             'selector': selector,
    #             'M_segments': M_segments
    #         }
    #         request.session['current_image'] = selected_img  # Update the session image
    #     except Exception as e:
    #         return JsonResponse({'status': 'error', 'message': f'Failed to load segmentation: {e}'})

    # Get session state
    state = session_data[session_key]
    
    selector = state['selector']
    M_segments = state['M_segments']

    H, W = M_segments.shape
    labels = selector.next_step()

    # Generate segmentation image
    seg_image = np.zeros((H, W, 3), dtype=np.uint8)
    unique_cluster_labels = set(labels)
    colors = {
        label: tuple(random.randint(0, 255) for _ in range(3))
        for label in unique_cluster_labels
    }

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
        'num_clusters': len(unique_cluster_labels)
    })
