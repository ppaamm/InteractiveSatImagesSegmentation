from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.template.loader import render_to_string

TOTAL_STEPS = 3


def index(request):
    return redirect('instructions:instruction_step', step=1)

def instruction_step(request, step):

    template = loader.get_template("instructions/index.html")
    
    if step < 1 or step > TOTAL_STEPS:
        return redirect('instruction_step', step=1)

    progress_percent = int((step / TOTAL_STEPS) * 100)
    print(progress_percent)
    
    content_template = f"instructions/steps/step_{step}.html"
    content_html = render_to_string(content_template)

    context = {
        'step': step,
        'total_steps': TOTAL_STEPS,
        'content': content_html,
        'has_prev': step > 1,
        'has_next': step < TOTAL_STEPS,
        'prev_step': step - 1,
        'next_step': step + 1,
        'progress_percent': progress_percent,
    }
    
    
    return HttpResponse(template.render(context, request))