from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader

INSTRUCTION_STEPS = [
    "Step 1: Welcome to the guide!",
    "Step 2: Let's start with basics.",
    "Step 3: Here's something advanced.",
    "Step 4: You're almost done!",
    "Step 5: Finished!",
]

def instruction_step(request, step):

    template = loader.get_template("instructions/index.html")
    
    
    total_steps = len(INSTRUCTION_STEPS)
    if step < 1 or step > total_steps:
        return redirect('instruction_step', step=1)

    progress_percent = int((step / total_steps) * 100)

    context = {
        'step': step,
        'total_steps': total_steps,
        'content': INSTRUCTION_STEPS[step - 1],
        'has_prev': step > 1,
        'has_next': step < total_steps,
        'prev_step': step - 1,
        'next_step': step + 1,
        'progress_percent': progress_percent,
    }
    
    
    return HttpResponse(template.render(context, request))