from django.db import models


class Project(models.Model):
    name = models.TextField()
    base_model = models.TextField()

    model_timestamp = models.DateTimeField(null=True)
    model_location = models.TextField(null=True)
    model_evaluation = models.TextField(null=True)

class Document(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    text = models.TextField()
    gold = models.BooleanField()
    reference = models.TextField(null=True)
    title = models.TextField()


class Label(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    label = models.TextField()

    def __str__(self):
        return self.label


class Annotation(models.Model):
    document = models.ForeignKey(Document, models.CASCADE)
    label = models.ForeignKey(Label, models.CASCADE)
    accept = models.BooleanField()

    class Meta:
        unique_together = [("document", "label")]