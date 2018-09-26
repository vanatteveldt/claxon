import json
import logging
import os
import shutil
from collections import OrderedDict
from functools import lru_cache
from random import sample
from typing import Sequence

import spacy
from django.conf import settings
from huey.contrib import djhuey
from numpy import argsort
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.util import minibatch, compounding
from tqdm import tqdm

from actcode import tasks
from actcode.eval import Eval, combine
from actcode.models import Annotation, Document, Project, Session


class ActiveLearn:
    """
    Serializable class that remembers active learning state between HTTP sessions
    """

    def __init__(self, session):
        self.session = session
        state = json.loads(session.state) if session.state else {}
        self.todo_docids = state.get('docids', [])
        self.scores = {int(id): s for (id, s) in state.get('scores', {}).items()}
        self.latest_annotation_id = state.get('latest_annotation')

    def _save_state(self, model):
        state = dict(docids=self.todo_docids, scores=self.scores)
        try:
            max_ann = self.session.annotation_set.latest("pk")
            state["latest_annotation"] = max_ann.id
        except Annotation.DoesNotExist:
            pass

        if model:
            save_model(model, session=self.session)
        self.session.state = json.dumps(state)
        self.session.save()

    def get_todo(self):
        done = set(self.session.annotation_set.values_list("document_id", flat=True))
        return [id for id in self.todo_docids if id not in done]

    def get_doc(self, trigger=None):
        """Get the next document to annotate (and score)"""
        todo = self.get_todo()
        if not todo:
            self._populate_todo(n=settings.N_QUEUE)
            return self.get_doc()
        if trigger and len(todo) <= trigger and not self.session.populate_task_id:
            logging.info("Triggering background populate")
            res = tasks.populate_todo(self.session.id, n=settings.N_QUEUE)
            self.session.populate_task_id = res.task.task_id
            self.session.save()
        return Document.objects.get(pk=todo[0])

    def _populate_todo(self, n):
        task_id = self.session.populate_task_id
        if task_id:
            logging.info("Waiting for populate task {task_id}".format(**locals()))
            scores = djhuey.HUEY.result(task_id, blocking=True)
        else:
            logging.info("Doing synchronous populate")
            scores = self.do_populate_todo(n)
        if not scores:
            raise Exception("This session is done! Wouter is working on a nicer screen (with a cake)")

    def do_populate_todo(self, n):
        logging.info("Populating active learning todo, n={n}".format(**locals()))
        model, is_updated = self._get_updated_model()
        scores = get_todo(self.session, model, n)
        self.todo_docids = list(scores.keys())
        self.scores.update(scores)

        if not scores:
            logging.info("Session {self.session.id} done!".format(**locals()))
            delete_session_model(self.session)
            model = None

        logging.debug("Saving state")
        self._save_state(model=(model if is_updated else None))
        logging.info("Populating done, |todo|={}".format(len(self.todo_docids)))
        return scores

    def _get_updated_model(self):
        """Get the classifier, updating if needed with coded documents since last update"""
        annotations = self.session.annotation_set.all()
        model = get_session_model(self.session)
        if model:
            annotations = annotations.filter(pk__gt=self.latest_annotation_id)
            logging.debug("Got model from session state, latest annotation={}".format(self.latest_annotation_id))
        else:
            model = get_model(self.session.project)
            logging.debug("Got model from project")

        annnotations = list(annotations)
        if annotations:
            logging.info("Updating model with {} annotations".format(len(annnotations)))
            with model.disable_pipes(*model.pipe_names[:-1]):
                optimizer = model.begin_training()
                losses = {}
                for batch in minibatch(annotations, size=compounding(4., 32., 1.001)):
                    batch_tokens = [get_tokens(model,  a.document_id) for a in batch]
                    batch_annnotations = [{'cats': {self.session.label.label: a.accept}} for a in batch]
                    model.update(batch_tokens, batch_annnotations, sgd=optimizer, drop=0.2, losses=losses)
            is_updated = True
        else:
            is_updated = False
        return model, is_updated


def get_todo(session: Session, model: Language, n=10) -> OrderedDict:
    """Populate the queue of documents to code"""
    done = {a.document_id for a in Annotation.objects.filter(document__gold=False, label=session.label)}
    todo = Document.objects.filter(gold=False).exclude(pk__in=done)
    if session.query:
        todo = todo.filter(text__icontains=session.query)
    todo = list(todo.values_list("id", flat=True))
    logging.debug("{ntodo} documents in todo (query: {q}, done={ndone})"
                  .format(ntodo=len(todo), ndone=len(done), q=session.query))
    if len(todo) > settings.N_SAMPLE:
        todo = sample(todo, settings.N_SAMPLE)

    tc = model.get_pipe("textcat")
    tokens = [get_tokens(model, doc_id) for doc_id in todo]
    scores = [d.cats[session.label.label] for d in tc.pipe(tokens)]
    uncertainty = [abs(score - 0.5) for score in scores]
    index = list(argsort(uncertainty))[:n]

    return OrderedDict((todo[i], scores[i]) for i in index)


def train(project: Project, annotations: Sequence[Annotation], iterations=10, callback=None):
    model = get_base_model(project)

    labels = {l.id: l.label for l in project.label_set.all()}
    tc = model.get_pipe("textcat")
    with model.disable_pipes(*model.pipe_names[:-1]):
        optimizer = model.begin_training()
        for i in range(iterations):
            losses = {}
            for batch in tqdm(list(minibatch(annotations, size=compounding(4., 32., 1.001)))):
                tokens = [get_tokens(model, a.document_id) for a in batch]
                batch_annotations = [{'cats': {labels[a.label_id]: a.accept}} for a in batch]
                model.update(tokens, batch_annotations, sgd=optimizer, drop=0.2, losses=losses)
            if callback:
                callback(i, model)
    return model

def retrain(project: Project, iterations=10):
    annotations = list(Annotation.objects.filter(session__project_id=project.id, document__gold=False))
    train_eval = sample(annotations, 500) if len(annotations) > 500 else annotations
    logging.info("Retraining model using {} annotations".format(len(annotations)))
    def log_performance(i, model):
        evals = evaluate(project, model)
        evals_t = evaluate(project, model, train_eval)
        logging.info("It.{:2}: {} (train: {})".format(i, combine(evals).eval_str(label=""), combine(evals_t).eval_str(label="")))
    model = train(project, annotations, iterations, callback=log_performance)
    evals = evaluate(project, model)

    save_model(model, project)
    project.model_evaluation = json.dumps([e.to_dict() for e in evals])
    project.save()
    logging.info("Done retraining!")


def get_predictions(project: Project, model: Language=None, annotations=None):
    """Evaluate a label based on the project's model and gold annotations"""
    if model is None:
        model = get_model(project)
    tc = model.get_pipe("textcat")
    labels = {l.id: l.label for l in project.label_set.all()}

    gold = {}  # doc.id : {label: T/F, ..}
    if annotations is None:
        annotations = Annotation.objects.filter(document__gold=True, session__project=project)
    for a in annotations:
        gold.setdefault(a.document_id, {})[labels[a.label_id]] = a.accept

    docs = list(gold.keys())
    tokens = [get_tokens(model, doc) for doc in docs]
    for doc, result in zip(docs, tc.pipe(tokens)):
        for label, accept in gold[doc].items():
            predict = result.cats[label] > .5
            yield doc, label, accept, predict, result.cats[label]


def evaluate(project: Project, model: Language=None, annotations=None):
    """Evaluate a label based on the project's model and gold annotations"""
    labels = {l.id: l.label for l in project.label_set.all()}
    eval = {label: Eval(label) for label in labels.values()}
    for _doc, label, accept, predict, _score in get_predictions(project, model, annotations):
        eval[label].add(accept, predict)
    return eval.values()


def get_model_dir(project: Project=None, session: Session=None):
    if project:
        f = "project_{}".format(project.id)
    elif session:
        f = "session_{}".format(session.id)
    else:
        raise ValueError("Specify project or session")
    return os.path.join(settings.MODEL_DIR, f)


def save_model(model: Language, project=None, session=None):
    output_dir=get_model_dir(project, session)
    logging.debug("Saving model to {}...".format(output_dir))
    model.to_disk(output_dir)
    return output_dir


def get_model(project: Project) -> Language:
    """Get the retrained model for this project, or the base model if not found"""
    model_dir = get_model_dir(project)
    if not os.path.exists(model_dir):
        model_dir = project.base_model
    return _get_model(model_dir, project)


def get_session_model(session: Session) -> Language:
    """Get the model for this session, or None if not found"""
    model_dir = get_model_dir(session=session)
    if os.path.exists(model_dir):
        return _get_model(model_dir, session.project)
    else:
        logging.debug("Model not found: {model_dir}".format(**locals()))


def delete_session_model(session: Session):
    model_dir = get_model_dir(session=session)
    logging.debug("Deleting model {model_dir}".format(**locals()))
    shutil.rmtree(model_dir)


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


@lru_cache(maxsize=2048)
def get_tokens(model: Language, doc_id: int):
    fn = os.path.join(settings.TOKEN_DIR, str(doc_id))
    if not os.path.exists(fn):
        raise ValueError("Document {doc_id} has not been preprocessed ({fn} does not exist)".format(**locals()))
    return Doc(model.vocab).from_disk(fn)