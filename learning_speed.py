import csv
import logging
import random
from functools import partial

import django
django.setup()
import sys


from actcode.eval import combine
from actcode.ml import train, evaluate

from math import ceil

from actcode.models import Project, Annotation, Label

project = Project.objects.get(pk=1)
label = Label.objects.get(project=project, label="criminaliteit")
all_annotations = list(Annotation.objects.filter(session__project=project, document__gold=False, label=label).order_by("id"))

#project = Project.objects.get(pk=3)
#all_annotations = list(Annotation.objects.filter(session__project=project, document__gold=False).order_by("id"))
LAST_EVAL = None
#x, y = all_annotations[:682], all_annotations[682:]
#random.shuffle(x)
#all_annotations = x + y
N = len(all_annotations)
I = 10
out = csv.writer(sys.stdout)
out.writerow(["n", "i", "label", "tp", "fp", "fn", "tn"])

eval_annotations = list(Annotation.objects.filter(document__gold=True, session__project=project, label=label))

def log_performance(i, model, n):
    global LAST_EVAL
    eval, = evaluate(project, model, annotations=eval_annotations)
    out.writerow([n, i, eval.label, eval.tp, eval.fp, eval.fn, eval.tn])
    LAST_EVAL = eval
    logging.info("[{n}/{N}][{i}/{I}] {}".format(LAST_EVAL.eval_str(), N=N, I=I, **locals()))


percs = [.25, .50, .75, 1]
#percs = [.125, .25, .375, .5, .625, .75, .875, 1.0]
for perc in percs:
    n = ceil(len(all_annotations) * perc)
    annotations = all_annotations[:n]

    model = train(project, annotations, I, callback=partial(log_performance, n=n))
    logging.info("{}: {}".format(len(annotations), LAST_EVAL.eval_str()))

