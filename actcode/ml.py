import datetime
import os
from random import sample

import spacy
from django.conf import settings
from django.db import connection, reset_queries
from django.utils import timezone
from numpy import argsort
from spacy.tokens.doc import Doc
from spacy.util import minibatch, compounding
from tqdm import tqdm

from spacy.language import Language

from actcode.eval import Eval, combine
from actcode.models import Label, Annotation, Document, Project


class ActiveLearn:
    """
    Serializable class that remembers active learning state between HTTP sessions
    """
    N_SAMPLE = 1000
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
        docs = list(Document.objects.filter(pk__in=todo).only("id"))
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

def evaluate(project: Project, model):
    """Evaluate a label based on the project's model and gold annotations"""
    tc = model.get_pipe("textcat")
    labels = {l.id: l.label for l in project.label_set.all()}
    eval = {label: Eval(label) for label in labels.values()}

    gold = {}  # doc.id : {label: T/F, ..}
    for a in Annotation.objects.filter(document__gold=True, document__project=project):
        gold.setdefault(a.document_id, {})[labels[a.label_id]] = a.accept

    docs = list(gold.keys())
    #print("Classifying {} annotated documents".format(len(docs)))

    tokens = [get_tokens(model, project.id, doc) for doc in docs]
    for doc, result in zip(docs, tc.pipe(tokens)):
        for label, accept in gold[doc].items():
            predict = result.cats[label] > .5
            eval[label].add(accept, predict)

    return eval.values()




def retrain(project: Project, iterations=10):
    model = get_model(project)
    evals = evaluate(project, model)
    if iterations:
        print("Before", combine(evals).eval_str())
        annotations = list(Annotation.objects.filter(document__project_id=project.id, document__gold=False).values_list("id", flat=True))
        print("Retraining model from ", len(annotations), "annotations")
        labels = {l.id: l.label for l in project.label_set.all()}
        with model.disable_pipes(*model.pipe_names[:-1]):
            optimizer = model.begin_training()
            for i in range(iterations):
                losses = {}
                for batch in tqdm(list(minibatch(annotations, size=compounding(4., 32., 1.001)))):
                    ann = list(Annotation.objects.filter(pk__in=batch).select_related("document").only("id", "accept", "label_id", "document__text"))
                    reset_queries()

                    batch_texts = [a.document.text for a in ann]
                    batch_annotations = [{'cats': {labels[a.label_id]: a.accept}} for a in ann]

                    model.update(batch_texts, batch_annotations, sgd=optimizer, drop=0.2, losses=losses)
                evals = evaluate(project, model)
                for eval in sorted(evals, key=lambda e: e.f):
                    print(eval.eval_str())
                print(combine(evals).eval_str())

        print()
    for eval in sorted(evals, key=lambda e: e.f):
        print(eval.eval_str())
    print(combine(evals).eval_str())



def get_tokens(model: Language, project_id: int, doc_id: int):
    fn = os.path.join(settings.TOKEN_DIR, "project_{}".format(project_id), str(doc_id))
    if not os.path.exists(fn):
        raise ValueError("Document {self.id} has not been preprocessed ({fn} does not exist)".format(**locals()))
    return Doc(model.vocab).from_disk(fn)

def get_model(project: Project) -> Language:
    model_name = project.base_model
    model_name = 'nl_core_news_sm'
    model = spacy.load(model_name)
    if 'textcat' not in model.pipe_names:
        textcat = model.create_pipe('textcat')
        model.add_pipe(textcat, last=True)
        for label in project.label_set.values_list("label", flat=True):
            textcat.add_label(label)
    return model
