# Silly django stuff
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'claxon.settings'
import django
django.setup()

import csv
import sys
import logging
import random
from functools import partial
from math import ceil

from actcode.ml import train, evaluate
from actcode.models import Project, Annotation, Label

project = Project.objects.get(pk=1)
label = Label.objects.get(project=project, label="criminaliteit")
all_annotations = list(Annotation.objects.filter(session__project=project, document__gold=False, label=label).order_by("id"))

#project = Project.objects.get(pk=3)
#all_annotations = list(Annotation.objects.filter(session__project=project, document__gold=False).order_by("id"))
LAST_EVAL = None
N = len(all_annotations)
I = 10
out = csv.writer(sys.stdout)
out.writerow(["n", "i", "label", "tp", "fp", "fn", "tn"])

eval_annotations = list(Annotation.objects.filter(document__gold=True, session__project=project, label=label))

def shuffle_annotations(annotations):
    # Would like to use sessions, but at least crime just doesn't have any, so split and shuffle Niek data and add the rest
    x, y = annotations[:682], annotations[682:]
    random.shuffle(x)
    return x + y

def log_performance(i, model, n, j, N, I):
    global LAST_EVAL
    eval, = evaluate(project, model, annotations=eval_annotations)
    out.writerow([n, i, eval.label, eval.tp, eval.fp, eval.fn, eval.tn])
    LAST_EVAL = eval
    logging.info("[{j}/5][{n}/{N}][{i}/{I}] {}".format(LAST_EVAL.eval_str(), **locals()))


percs = [.25, .50, .75, 1]
percs = [.125, .25, .375, .5, .625, .75, .875, 1.0]

for j in range(10):
    # python sorting is guaranteed to be stable, so shuffle then sort on sessionid
    all_annotations = shuffle_annotations(all_annotations)
    for perc in percs:
        n = ceil(len(all_annotations) * perc)
        annotations = all_annotations[:n]

        model = train(project, annotations, I, callback=partial(log_performance, n=n, j=j, N=N, I=I))
        logging.info("{}: {}".format(len(annotations), LAST_EVAL.eval_str()))

