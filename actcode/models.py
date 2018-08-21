import spacy
from django.db import models
from spacy.language import Language


class Project(models.Model):
    name = models.TextField()
    base_model = models.TextField()

    last_model = models.DateTimeField(null=True)
    model_task = models.TextField(null=True)
    model_location = models.TextField(null=True)

    def get_model(self) -> Language:
        model_name = self.base_model
        return spacy.load(model_name)


class Document(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    text = models.TextField()
    gold = models.BooleanField()
    reference = models.TextField(null=True)
    tokens = models.BinaryField(null=True)


class Label(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    label = models.TextField()

    last_eval = models.DateTimeField(null=True)
    eval_task = models.TextField(null=True)
    tp = models.IntegerField(null=True)
    fn = models.IntegerField(null=True)
    fp = models.IntegerField(null=True)

    @property
    def precision(self):
        if (not self.tp) or (self.fp is None):
            return self.tp # zero or None...
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        if (not self.tp) or (self.fn is None):
            return self.tp # zero or None...
        return self.tp / (self.tp + self.fn)

    @property
    def fscore(self):
        pr = self.precision
        re = self.recall
        if (not pr) or (not re):
            return pr
        return (2 * pr * re) / (pr + re)

    def __str__(self):
        return self.label


class Annotation(models.Model):
    document = models.ForeignKey(Document, models.CASCADE)
    label = models.ForeignKey(Label, models.CASCADE)
    accept = models.BooleanField()