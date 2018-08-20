from django.db.models import Count
from django.forms import ModelForm
from django.shortcuts import render

# Create your views here.
from django.views.generic import TemplateView, FormView

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

        labelstats = {l.id: dict(id=l.id, label=l.label) for l in self.project.label_set.all()}

        for a in Annotation.objects.filter(document__project=self.project.id).values("document__gold", "label",).annotate(n=Count('id')):
            key = 'ngold' if a['document__gold'] else 'ntrain'
            labelstats[a['label']][key] = a['n']


        kwargs['labelstats'] = sorted(labelstats.values(), key=lambda l:l['label'])
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
            Annotation.objects.create(document=doc, label_id=kwargs['label'], accept=self.request.GET['accept'])

        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        kwargs = super().get_context_data(**kwargs)
        done = {a.document_id for a in Annotation.objects.filter(document__gold=True, document__project_id=self.project.id)}
        total = len(Document.objects.filter(gold=True, project_id=self.project.id))
        if total > len(done):
            doc = self.project.document_set.filter(gold=True).exclude(pk__in=done)[0]
        percent = 100 * len(done) // total
        percent_w = 10 + 90 * len(done) // total

        kwargs.update(**locals())
        return kwargs

