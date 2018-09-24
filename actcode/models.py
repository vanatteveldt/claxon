import datetime

from django.db import models


class Project(models.Model):
    name = models.TextField()
    base_model = models.TextField()
    model_evaluation = models.TextField(null=True)

class Document(models.Model):
    text = models.TextField()
    gold = models.BooleanField()
    reference = models.TextField(null=True)
    title = models.TextField()


class Label(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    label = models.TextField()

    def __str__(self):
        return self.label

class Session(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    label = models.ForeignKey(Label, models.CASCADE)
    train = models.BooleanField()
    start_date = models.DateTimeField(default=datetime.datetime.now)
    description = models.TextField()
    query = models.TextField(null=True)
    state = models.TextField(null=True)
    populate_task_id = models.TextField(null=True)

class Annotation(models.Model):
    session = models.ForeignKey(Session, models.CASCADE)
    document = models.ForeignKey(Document, models.CASCADE)
    label = models.ForeignKey(Label, models.CASCADE)
    accept = models.BooleanField()

    class Meta:
        unique_together = [("document", "label")]