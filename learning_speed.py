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

project = Project.objects.get(pk=3)
#label = Label.objects.get(project=project, label="criminaliteit")
#all_annotations = list(Annotation.objects.filter(session__project=project, document__gold=False, label=label).order_by("id"))
all_annotations = list(Annotation.objects.filter(session__project=project, document__gold=False).order_by("id"))
LAST_EVAL = None
x, y = all_annotations[:682], all_annotations[682:]
random.shuffle(x)
all_annotations = x + y

out = csv.writer(sys.stdout)
out.writerow(["n", "i", "label", "tp", "fp", "fn", "tn"])
def log_performance(i, model, n):
    global LAST_EVAL
    evals = evaluate(project, model)
    for eval in evals:
        out.writerow([n, i, eval.label, eval.tp, eval.fp, eval.fn, eval.tn])
    LAST_EVAL = evals

percs = [.25, .50, .75, 1]
percs = [.125, .25, .375, .5, .625, .75, .875, 1.0]
for perc in percs:
    n = ceil(len(all_annotations) * perc)
    annotations = all_annotations[:n]

    model = train(project, annotations, 10, callback=partial(log_performance, n=n))
    logging.info("{}: {}".format(len(annotations), combine(LAST_EVAL).eval_str()))

