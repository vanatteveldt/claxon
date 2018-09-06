import os

import spacy
from django.db import models
from spacy.language import Language
from django.conf import settings


class Project(models.Model):
    name = models.TextField()
    base_model = models.TextField()

    last_model = models.DateTimeField(null=True)
    model_task = models.TextField(null=True)
    model_location = models.TextField(null=True)



class Document(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    text = models.TextField()
    gold = models.BooleanField()
    reference = models.TextField(null=True)
    title = models.TextField()


class Label(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    label = models.TextField()

    last_eval = models.DateTimeField(null=True)
    eval_task = models.TextField(null=True)
    tp = models.IntegerField(null=True)
    fn = models.IntegerField(null=True)
    fp = models.IntegerField(null=True)



    def __str__(self):
        return self.label


class Annotation(models.Model):
    document = models.ForeignKey(Document, models.CASCADE)
    label = models.ForeignKey(Label, models.CASCADE)
    accept = models.BooleanField()

    class Meta:
        unique_together = [("document", "label")]