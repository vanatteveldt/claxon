import datetime
from random import sample

from django.utils import timezone
from numpy import argsort
from spacy.tokens.doc import Doc
from spacy.util import minibatch, compounding

from actcode.models import Label, Annotation, Document




class ActiveLearn:
    """
    Serializable class that remembers active learning state between HTTP sessions
    """
    N_SAMPLE = 250
    N_QUEUE = 10

    def __init__(self, label_id):
        self.label_id = label_id
        self.coded_docids = []
        self.todo_docids = []
        self.todo_scores = []

    def done(self, doc_id):
        """Signify that this document is done and can be removed fromt the queue"""
        if(doc_id != self.todo_docids[0]):
            raise ValueError("doc_id {doc_id} is not the first todo ({self.todo_docids})")
        self.coded_docids.append(doc_id)
        self.todo_docids.pop(0)
        self.todo_scores.pop(0)

    def get_doc(self):
        """Get the next document to annotate (and score)"""
        if not self.todo_docids:
            self._populate_todo(n=self.N_QUEUE)
        return Document.objects.get(pk=self.todo_docids[0]), self.todo_scores[0]

    def _populate_todo(self, n=5):
        """Populate the queue of documents to code"""
        print("Populating active learning todo, n=", n)
        label = Label.objects.get(pk=self.label_id)

        done = {a.document_id for a in Annotation.objects.filter(document__gold=False, label=label,
                                                                 document__project_id=label.project_id)}
        todo = list(Document.objects.filter(project=label.project, gold=False).exclude(pk__in=done).values_list("id", flat=True))
        if len(todo) > self.N_SAMPLE:
            todo = sample(todo, self.N_SAMPLE)

        model = self._get_model()
        tc = model.get_pipe("textcat")
        docs = list(Document.objects.filter(pk__in=todo).only("id", "tokens"))
        tokens = [Doc(model.vocab).from_disk(doc.tokens) for doc in docs]
        scores = [d.cats[label.label] for d in tc.pipe(tokens)]
        uncertainty = [abs(score - 0.5) for score in scores]
        index = list(argsort(uncertainty))[:n]

        self.todo_docids = [docs[i].id for i in index]
        self.todo_scores = [scores[i] for i in index]
        print("Done, |todo|=", len(self.todo_docids))

    def _get_model(self):
        """Get the classifier, updating if needed with coded documents"""
        label = Label.objects.get(pk=self.label_id)
        model = label.project.get_model()
        if self.coded_docids:
            print("Updating model with ", len(self.coded_docids), "docs")
            print(model.pipe_names[:-1])
            # we have some documents, so let's update the model
            with model.disable_pipes(*model.pipe_names[:-1]):
                optimizer = model.begin_training()
                losses = {}
                for batch in minibatch(self.coded_docids, size=compounding(4., 32., 1.001)):
                    ann = {a.document_id: a for a in Annotation.objects.filter(label=label, document__id__in=batch)}
                    texts = []
                    annotations = []
                    for doc in Document.objects.filter(pk__in=batch).only("id", "text"):
                        texts.append(doc.text)
                        annotations.append({'cats': {label.label: ann[doc.id].accept}})
                    print(annotations)
                    model.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        return model

    def to_dict(self):
        """Serialize"""
        return dict(label_id=self.label_id,
                    coded_docids=self.coded_docids, todo_docids=self.todo_docids,
                    todo_scores=self.todo_scores)

    @classmethod
    def from_dict(cls, state):
        """Deserialize"""
        al = cls(label_id=state["label_id"])
        al.todo_scores = state["todo_scores"]
        al.todo_docids = state["todo_docids"]
        al.coded_docids = state["coded_docids"]
        return al

def evaluate(label: Label):
    """Evaluate a label based on the project's model and gold annotations"""
    model = label.project.get_model()
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