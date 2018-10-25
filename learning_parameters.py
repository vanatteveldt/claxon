# Silly django stuff
import os

from actcode.eval import combine

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

label = None #Label.objects.get(project=project, label="criminaliteit")
if label:
    annotations = list(Annotation.objects.filter(session__project=project, document__gold=False, label=label).order_by("id"))
    eval_annotations = list(Annotation.objects.filter(document__gold=True, session__project=project, label=label))
else:
    annotations = list(Annotation.objects.filter(session__project=project, document__gold=False))
    eval_annotations = list(Annotation.objects.filter(document__gold=True, session__project=project))

out = csv.writer(sys.stdout)
params = ["j", "drop", "l2"]
out.writerow(["i"] + params + ["label", "tp", "fp", "fn", "tn"])


def log_performance(i, model, **cfg):
    evals = evaluate(project, model, annotations=eval_annotations)
    for eval in evals:
        row = [i] + [cfg[p] for p in params] + [eval.label, eval.tp, eval.fp, eval.fn, eval.tn]
        out.writerow(row)
    logging.info("{cfg} {i} {}".format(combine(evals).eval_str(), **locals()))


l2s = ["1e-8", "1e-7", "1e-6", "1e-5", "1e-4"]
drops = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

for j in range(10):
    for l2 in l2s:
        os.environ["SPACY_L2_PENALTY"] = l2
        for drop in drops:
            random.shuffle(annotations)
            model = train(project, annotations, iterations=10, drop=drop,
                          callback=partial(log_performance, j=j, drop=drop, l2=l2))

