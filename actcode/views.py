import json
import logging

from django.db.models import Count
from django.urls import reverse
from django.views.generic import TemplateView

from actcode.eval import Eval
from actcode.ml import ActiveLearn
from actcode.models import Project, Annotation, Label, Document


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
            self.state = ActiveLearn(self.label.id)

        if 'accept' in self.request.GET:
            doc = Document.objects.get(pk=int(self.request.GET['doc']))
            Annotation.objects.create(document=doc, label_id=kwargs['label'], accept=self.request.GET['accept'])
            self.state.done(doc.id)

        result = super().get(request, *args, **kwargs)
        self.request.session[CACHE_KEY] = self.state.to_dict()
        return result


    def get_context_data(self, **kwargs):
        kwargs = super().get_context_data(**kwargs)
        doc, score = self.state.get_doc()
        text = doc.text.replace("\n", "<br/>")
        base_url = reverse("actcode:code-learn",  kwargs=dict(project=self.project.id, label=self.label.id))
        accept_url = "{base_url}?doc={doc.id}&accept=1".format(**locals())
        reject_url = "{base_url}?doc={doc.id}&accept=0".format(**locals())
        state = self.state
        kwargs.update(**locals())
        return kwargs


