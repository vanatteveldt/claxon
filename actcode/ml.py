import datetime
import json
import logging
import os
from random import sample
from functools import lru_cache

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
from actcode.models import Label, Annotation, Document, Project, Session


class ActiveLearn:
    """
    Serializable class that remembers active learning state between HTTP sessions
    """
    N_SAMPLE = 1000
    N_QUEUE = 10

    def __init__(self, label_id, session_id, query=None):
        self.label_id = label_id
        self.session_id = session_id
        self.query = query
        self.todo_docids = []
        self.scores = {}

    def get_todo(self):
        done = set(Annotation.objects.filter(session_id=self.session_id).values_list("document_id", flat=True))
        return [id for id in self.todo_docids if id not in done]

    def get_doc(self):
        """Get the next document to annotate (and score)"""
        todo = self.get_todo()
        if not todo:
            self._populate_todo(n=self.N_QUEUE)
            if not self.todo_docids:
                raise ValueError("Done! Sorry for not handling this properly")
            return self.get_doc()
        return Document.objects.get(pk=todo[0])

    def _populate_todo(self, n=5):
        """Populate the queue of documents to code"""
        logging.info("Populating active learning todo, n={n}".format(**locals()))
        label = Label.objects.get(pk=self.label_id)

        done = {a.document_id for a in Annotation.objects.filter(document__gold=False, label=label,
                                                                 document__project_id=label.project_id)}
        todo = Document.objects.filter(project=label.project, gold=False).exclude(pk__in=done)
        if self.query:
            todo = todo.filter(text__icontains=self.query)
        todo = list(todo.values_list("id", flat=True))
        logging.debug("{} documents in todo".format(len(todo)))
        if len(todo) > self.N_SAMPLE:
            todo = sample(todo, self.N_SAMPLE)

        model = self._get_updated_model()
        tc = model.get_pipe("textcat")
        docids = list(Document.objects.filter(pk__in=todo).values_list("id", flat=True))
        tokens = [get_tokens(model, label.project_id, doc_id) for doc_id in docids]
        scores = [d.cats[label.label] for d in tc.pipe(tokens)]
        uncertainty = [abs(score - 0.5) for score in scores]
        index = list(argsort(uncertainty))[:n]

        self.todo_docids = [docids[i] for i in index]
        self.scores.update({docids[i]: scores[i] for i in index})
        logging.info("Populating done, |todo|={}".format(len(self.todo_docids)))

    def _get_updated_model(self):
        """Get the classifier, updating if needed with coded documents"""
        label = Label.objects.get(pk=self.label_id)
        model = get_model(label.project)
        ann = list(Annotation.objects.filter(session_id=self.session_id)
                       .select_related("document").only("accept", "document__text"))
        if ann:
            logging.info("Updating model with {} annotations".format(len(ann)))
            # we have some documents, so let's update the model
            with model.disable_pipes(*model.pipe_names[:-1]):
                optimizer = model.begin_training()
                losses = {}
                for batch in minibatch(ann, size=compounding(4., 32., 1.001)):
                    texts = [a.document.text for a in ann]
                    annotations = [{'cats': {label.label: a.accept}} for a in ann]
                    model.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        return model

    def to_dict(self):
        """Serialize"""
        return dict(label_id=self.label_id,
                    session_id=self.session_id,
                    todo_docids=self.todo_docids,
                    scores=self.scores,
                    query=self.query)

    @classmethod
    def from_dict(cls, state):
        """Deserialize"""
        al = cls(label_id=state["label_id"], session_id=state["session_id"], query=state["query"])
        al.scores = {int(id): s for (id, s) in state["scores"].items()}
        al.todo_docids = state["todo_docids"]
        return al

def evaluate(project: Project, model, annotations=None):
    """Evaluate a label based on the project's model and gold annotations"""
    tc = model.get_pipe("textcat")
    labels = {l.id: l.label for l in project.label_set.all()}
    eval = {label: Eval(label) for label in labels.values()}

    gold = {}  # doc.id : {label: T/F, ..}
    if annotations is None:
        annotations = Annotation.objects.filter(document__gold=True, document__project=project)
    for a in annotations:
        gold.setdefault(a.document_id, {})[labels[a.label_id]] = a.accept

    docs = list(gold.keys())
    tokens = [get_tokens(model, project.id, doc) for doc in docs]
    for doc, result in zip(docs, tc.pipe(tokens)):
        for label, accept in gold[doc].items():
            predict = result.cats[label] > .5
            eval[label].add(accept, predict)
    return eval.values()


def retrain(project: Project, iterations=10):
    model = get_base_model(project)
    annotations = list(Annotation.objects.filter(document__project_id=project.id, document__gold=False))

    train_eval = sample(annotations, 500) if len(annotations) > 500 else annotations

    logging.info("Retraining model using {} annotations".format(len(annotations)))
    labels = {l.id: l.label for l in project.label_set.all()}
    with model.disable_pipes(*model.pipe_names[:-1]):
        optimizer = model.begin_training()
        for i in range(iterations):
            losses = {}
            for batch in tqdm(list(minibatch(annotations, size=compounding(4., 32., 1.001)))):
                texts = dict(Document.objects.filter(pk__in={a.document_id for a in batch}).values_list("id", "text"))

                batch_texts = [texts[a.document_id] for a in batch]
                batch_annotations = [{'cats': {labels[a.label_id]: a.accept}} for a in batch]

                model.update(batch_texts, batch_annotations, sgd=optimizer, drop=0.2, losses=losses)
            evals = evaluate(project, model)
            evals_t = evaluate(project, model, train_eval)
            logging.info("It.{:2}: {} (train: {})".format(i, combine(evals).eval_str(label=""), combine(evals_t).eval_str(label="")))

    output_dir = os.path.join(settings.MODEL_DIR, "project_{}_{}".format(project.id, datetime.datetime.now().isoformat()))
    logging.debug("Saving model to {}...".format(output_dir))

    model.to_disk(output_dir)
    project.last_model = datetime.datetime.now()
    project.model_location = output_dir
    project.model_evaluation = json.dumps([e.to_dict() for e in evals])
    project.save()
    logging.info("Saved model to {}".format(output_dir))

@lru_cache(maxsize=2048)
def get_tokens(model: Language, project_id: int, doc_id: int):
    fn = os.path.join(settings.TOKEN_DIR, "project_{}".format(project_id), str(doc_id))
    if not os.path.exists(fn):
        raise ValueError("Document {doc_id} has not been preprocessed ({fn} does not exist)".format(**locals()))
    return Doc(model.vocab).from_disk(fn)


def get_model(project: Project) -> Language:
    model_name = project.model_location or project.base_model
    return _get_model(model_name, project)

def get_base_model(project: Project) -> Language:
    return _get_model(project.base_model, project)

def _get_model(model_name: str, project: Project) -> Language:
    logging.info("Loading model from {}".format( model_name))
    model = spacy.load(model_name)
    if 'textcat' not in model.pipe_names:
        logging.info(".. adding textcat pipe")
        textcat = model.create_pipe('textcat')
        model.add_pipe(textcat, last=True)
        for label in project.label_set.values_list("label", flat=True):
            textcat.add_label(label)
    return model
