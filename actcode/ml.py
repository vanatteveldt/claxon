import datetime
from django.utils import timezone

import spacy

from actcode.models import Label, Annotation

def evaluate(label: Label):
    model_name = label.project.base_model
    print("Evaluating {label.project.id}: {label.label}; model: {model_name}".format(**locals()))

    model = spacy.load(label.project.base_model)
    ann = list(Annotation.objects.filter(document__gold=True, label=label))
    print("Classifying {} annotated documents".format(len(ann)))

    texts = [a.document.text for a in ann]
    result = model.pipe(texts)
    predictions = [x.cats[label.label] > .5 for x in result]

    print("Processing results")
    tp, fp, fn = 0, 0, 0
    for a, p in zip(ann, predictions):
        if a.accept and p:
            tp += 1
        elif a.accept and not p:
            fn += 1
        elif not a.accept and p:
            fp += 1

    label.fn = fn
    label.fp = fp
    label.tp = tp
    label.last_eval = datetime.datetime.now(tz=timezone.utc)

    print("TP={l.tp} FP={l.fp} FN={l.fn} Pr={l.precision} Re={l.recall} F1={l.fscore}".format(l=label))

    print("Storing results")
    label.save()
    print("Done")