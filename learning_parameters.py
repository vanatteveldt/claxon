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

from actcode.ml import train, evaluate
from actcode.models import Project, Annotation, Label

project = Project.objects.get(pk=1)
label = Label.objects.get(project=project, label="economie")
annotations = list(Annotation.objects.filter(session__project=project, document__gold=False, label=label).order_by("id"))
eval_annotations = list(Annotation.objects.filter(document__gold=True, session__project=project, label=label))

out = csv.writer(sys.stdout)
out.writerow(["j", "drop" ,"i", "label", "tp", "fp", "fn", "tn"])


def log_performance(i, model, n, j, drop):
    global LAST_EVAL
    eval, = evaluate(project, model, annotations=eval_annotations)
    out.writerow([j, i, drop, eval.label, eval.tp, eval.fp, eval.fn, eval.tn])
    logging.info("[{j};{drop};{i}] {}".format(LAST_EVAL.eval_str(), **locals()))


drops = [x/100 for x in range(0, 40, 51)]

for j in range(10):
    for drop in drops:
        random.shuffle(annotations)
        model = train(project, annotations, iterations=10, drop=drop, callback=partial(log_performance, j=j, drop=drop))

