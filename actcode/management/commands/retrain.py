import json

from django.core.management.base import BaseCommand

from actcode.eval import combine
from actcode.ml import retrain, evaluate
from actcode.models import Project


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument("project_id")
        parser.add_argument('--iterations', type=int, nargs='?', default=10)

    def handle(self, *args, **options):
        p = Project.objects.get(id=options['project_id'])
        n =  options['iterations']
        if n > 0:
            retrain(p, iterations = n)
        else:
            # evaluate only
            evals = evaluate(p)
            for e in evals:
                print(e.eval_str(fill_label=14))
            print(combine(evals).eval_str(fill_label=14))
            p.model_evaluation = json.dumps([e.to_dict() for e in evals])
            p.save()
