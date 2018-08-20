import jsonlines
from django.core.management.base import BaseCommand
from actcode.models import Project, Label, Document


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument("name")
        parser.add_argument("base_model")
        parser.add_argument("documents")
        parser.add_argument("labels")


    def handle(self, *args, **options):
        p = Project.objects.create(name=options['name'], base_model=options['base_model'])

        labels = [l.strip() for l in options['labels'].split(",")]
        labels = [Label(project=p, label=l) for l in labels]
        Label.objects.bulk_create(labels)

        docs = [Document(project=p, text=doc['text']) for doc in jsonlines.open(options['documents'])]
        Document.objects.bulk_create(docs)

        print("Created project {p.id}:{p.name} with {} documents and labels {}".
              format(p.document_set.count(), p.label_set.all(), **locals()))
