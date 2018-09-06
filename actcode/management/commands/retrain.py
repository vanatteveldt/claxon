from django.core.management.base import BaseCommand
from actcode.ml import retrain, evaluate
from actcode.models import Project


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument("project_id")
        parser.add_argument('--iterations', type=int, nargs='?', default=10)

    def handle(self, *args, **options):
        p = Project.objects.get(id=options['project_id'])
        retrain(p, iterations = options['iterations'])