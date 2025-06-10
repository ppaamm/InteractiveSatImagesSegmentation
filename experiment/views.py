from django.http import HttpResponse
from django.template import loader
from django.conf import settings


OPACITY = 0.5

IMAGES = {"Castle": "2012-04-26-Muenchen-Tunnel_4K0G0080",
          "Commercial area": "2012-04-26-Muenchen-Tunnel_4K0G0020",
          "Farm": "2012-04-26-Muenchen-Tunnel_4K0G0010", 
          "Railway track": "2012-04-26-Muenchen-Tunnel_4K0G0100",
          "Urban area 1": "2012-04-26-Muenchen-Tunnel_4K0G0051",
          "Urban area 2": "2012-04-26-Muenchen-Tunnel_4K0G0070",
          "Urban area 3": "2012-04-26-Muenchen-Tunnel_4K0G0090",
          }


def index(request):
    selected = request.GET.get('image', 'source-image')  # without extension
    image_url = settings.MEDIA_URL + f'satellite/{selected}.jpg'
    segmentation_url = settings.MEDIA_URL + f'satellite/{selected}.png'

    context = {
        'image_url': image_url,
        'background_image_url': image_url,
        'overlay_image_url': segmentation_url,
        'overlay_opacity': OPACITY,
    }

    template = loader.get_template("experiment/index.html")
    return HttpResponse(template.render(context, request))



def image_selection(request):
    template = loader.get_template("experiment/image_selection.html")
    context = {'images': IMAGES, 
               'MEDIA_URL': settings.MEDIA_URL, }
    return HttpResponse(template.render(context, request))
    