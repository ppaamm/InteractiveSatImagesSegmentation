from django.http import HttpResponse
from django.template import loader
from django.conf import settings


OPACITY = 0.5

def index(request):
    template = loader.get_template("experiment/index.html")
    
    image_url = settings.MEDIA_URL + 'source-image.jpg'
    segmentation_url = settings.MEDIA_URL + 'segmentation.png'
    
    context = { 
        'image_url': image_url,
        'background_image_url': image_url, 
        'overlay_image_url': segmentation_url,
        'overlay_opacity': OPACITY,
    }
    
    return HttpResponse(template.render(context, request))