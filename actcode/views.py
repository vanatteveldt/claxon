import json
import logging
import re

from django import forms
from django.core.exceptions import ValidationError
from django.db.models import Count
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import TemplateView, FormView

from actcode.eval import Eval
from actcode.ml import ActiveLearn
from actcode.models import Project, Annotation, Label, Document, Session


class ProjectListView(TemplateView):
    template_name = "projectlist.html"

    def get_context_data(self, **kwargs):
        kwargs = super().get_context_data(**kwargs)
        kwargs['projects'] = Project.objects.all()
        return kwargs


class ProjectView(TemplateView):
    template_name = "project.html"

    @property
    def project(self):
        return Project.objects.get(pk=self.kwargs['project'])

    def get_context_data(self, **kwargs):
        kwargs = super().get_context_data(**kwargs)


        labelstats = {l.id: dict(label=l) for l in self.project.label_set.all()}
        for a in Annotation.objects.filter(document__project=self.project.id).values("document__gold", "label",).annotate(n=Count('id')):
            key = 'ngold' if a['document__gold'] else 'ntrain'
            labelstats[a['label']][key] = a['n']

        if self.project.model_evaluation:
            labels = {l['label'].label: id for (id, l) in labelstats.items()}
            for eval in [Eval.from_dict(x) for x in json.loads(self.project.model_evaluation)]:
                d = labelstats[labels[eval.label]]
                d["f1"] = eval.f
                d["pr"] = eval.pr
                d["re"] = eval.re

        kwargs['ngold_total'] = self.project.document_set.filter(gold=True).count()
        kwargs['ntrain_total'] = self.project.document_set.filter(gold=False).count()
        kwargs['labelstats'] = sorted(labelstats.values(), key=lambda l:l['label'].label)

        return kwargs


class CodeView(TemplateView):
    template_name = "code.html"

    @property
    def project(self):
        return Project.objects.get(pk=self.kwargs['project'])

    @property
    def label(self):
        return Label.objects.get(pk=self.kwargs['label'])

    def get(self, request, *args, **kwargs):
        if 'accept' in self.request.GET:
            doc = Document.objects.get(pk=int(self.request.GET['doc']))
            try:
                a = Annotation.objects.get(document=doc, label_id=kwargs['label'])
            except Annotation.DoesNotExist:
                Annotation.objects.create(document=doc, label_id=kwargs['label'], accept=self.request.GET['accept'])
            else:
                a.accept = self.request.GET['accept']
                a.save()

        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        kwargs = super().get_context_data(**kwargs)

        done = list(Annotation.objects.filter(document__gold=True, label=self.label, document__project_id=self.project.id)
                    .select_related("document").only("document_id", "accept", "document__text").order_by("-id"))

        total = len(Document.objects.filter(gold=True, project_id=self.project.id))
        all_done = total <= len(done)
        percent = 100 * len(done) // total
        percent_w = 10 + 90 * len(done) // total
        if not all_done:
            if 'next' in self.request.GET:
                docid = int(self.request.GET['next'])
                doc = self.project.document_set.get(gold=True, pk=docid)
            else:
                doc = self.project.document_set.filter(gold=True).exclude(pk__in={a.document_id for a in done})[0]
            text = doc.text.replace("\n", "<br/>")
            base_url = reverse("actcode:code-gold", kwargs=dict(project=self.project.id, label=self.label.id))
            accept_url = "{base_url}?doc={doc.id}&accept=1".format(**locals())
            reject_url = "{base_url}?doc={doc.id}&accept=0".format(**locals())

        kwargs.update(**locals())
        return kwargs


_AC_CACHE_KEY = "actcode_active_state_{label}"
class ActiveCodeView(TemplateView):
    template_name = "code.html"

    def get(self, request, *args, **kwargs):
        self.project = Project.objects.get(pk=self.kwargs['project'])
        self.label = Label.objects.get(pk=self.kwargs['label'])

        CACHE_KEY = _AC_CACHE_KEY.format(label=self.label.id)

        state = self.request.session.get(CACHE_KEY)
        if state:
            self.state = ActiveLearn.from_dict(state)
        else:
            session = Session.objects.create(project=self.project, train=True, description="Online active learning")
            self.state = ActiveLearn(self.label.id, session.id)

        if 'accept' in self.request.GET:
            doc = Document.objects.get(pk=int(self.request.GET['doc']))
            try:
                a = Annotation.objects.get(document=doc, label_id=kwargs['label'])
            except Annotation.DoesNotExist:
                Annotation.objects.create(document=doc, label_id=kwargs['label'], accept=self.request.GET['accept'],
                                          session_id=self.state.session_id)
            else:
                a.accept = self.request.GET['accept']
                a.save()

        result = super().get(request, *args, **kwargs)
        self.request.session[CACHE_KEY] = self.state.to_dict()
        return result


    def get_context_data(self, **kwargs):
        kwargs = super().get_context_data(**kwargs)

        done = Annotation.objects.filter(session_id=self.state.session_id)

        if 'next' in self.request.GET:
            docid = int(self.request.GET['next'])
            doc = self.project.document_set.get(pk=docid)
        else:
            doc = self.state.get_doc()
        score = self.state.scores.get(doc.id)
        ntodo = len(self.state.get_todo())
        text = doc.text.replace("\n", "<br/>")
        if self.state.query:
            text = re.sub("({})".format(self.state.query), "<b>\\1</b>", text, flags=re.I)
        base_url = reverse("actcode:code-learn",  kwargs=dict(project=self.project.id, label=self.label.id))
        accept_url = "{base_url}?doc={doc.id}&accept=1".format(**locals())
        reject_url = "{base_url}?doc={doc.id}&accept=0".format(**locals())
        state = self.state
        kwargs.update(**locals())
        return kwargs


class FilterForm(forms.Form):
    query = forms.CharField()


class FilterView(FormView):
    template_name = "filter.html"
    form_class = FilterForm

    def form_valid(self, form):
        q = form.cleaned_data['query']
        pid = self.kwargs['project']
        lid = self.kwargs['label']
        done = set(Annotation.objects.filter(document__project__id=pid, document__gold=False, label=lid)
                   .order_by("document_id").distinct().values_list("document_id",flat=True))
        self.n = Document.objects.filter(gold=False, project_id=self.kwargs['project']).exclude(pk__in=done).filter(text__icontains=q).count()
        if 'check' in self.request.POST:
            return self.form_invalid(form)
        if self.n == 0:
            form.add_error(field=None, error="No documents found for query")
            return self.form_invalid(form)

        session = Session.objects.create(project_id=pid, train=True,
                                         description="Online active learning with query: {}".format(q))
        state = ActiveLearn(lid, session.id, query=q)
        CACHE_KEY = _AC_CACHE_KEY.format(label=lid)
        self.request.session[CACHE_KEY] = state.to_dict()
        return redirect('actcode:code-learn', label=lid, project=pid)

    def get_context_data(self, **kwargs):
        kwargs['n'] = getattr(self, "n", None)
        return super().get_context_data(**kwargs)