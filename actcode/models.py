from django.db import models

class Project(models.Model):
    name = models.TextField()
    base_model = models.TextField()

    def get_docs(self, model, label, uncertain=True):
        docs = list(self.document_set.all())
        texts = (d.text for d in docs)
        scores = (x.cats[label] for x in model.pipe(texts))

        if uncertain: key = lambda txt_score: abs(txt_score[1] - 0.5)
        else: key = lambda txt_score: -txt_score[1]

        return sorted(zip(docs, scores), key = key)


class Document(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    text = models.TextField()
    gold = models.BooleanField()

class Label(models.Model):
    project = models.ForeignKey(Project, models.CASCADE)
    label = models.TextField()

    def __str__(self):
        return self.label

class Annotation(models.Model):
    document = models.ForeignKey(Document, models.CASCADE)
    label = models.ForeignKey(Label, models.CASCADE)
    accept = models.BooleanField()