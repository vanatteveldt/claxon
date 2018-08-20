from django.core.management.base import BaseCommand
from actcode.ml import evaluate
from actcode.models import Project, Label, Document


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument("project_id")
        parser.add_argument("label")


    def handle(self, *args, **options):
        l = Label.objects.get(project_id=options['project_id'], label=options['label'])
        evaluate(l)