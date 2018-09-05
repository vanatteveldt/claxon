import os

from tqdm import tqdm
from django.core.management.base import BaseCommand
from actcode.models import Project, Document
from django.conf import settings

_CHUNK_SIZE = 10


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument("project_id")

    def handle(self, *args, **options):
        project = Project.objects.get(pk=options['project_id'])
        dir = os.path.join(settings.TOKEN_DIR, "project_{}".format(project.id))
        os.makedirs(dir, exist_ok=True)
        docids = set(project.document_set.filter(gold=False).values_list("id", flat=True))
        preprocessed = {int(x) for x in os.listdir(dir)}
        todo = list(docids - preprocessed)
        if not todo:
            print("All documents already preprocessed")
            return
        print("Preprocessing {} documents".format(len(todo)))
        m = project.get_model()
        chunks = [todo[i:i+_CHUNK_SIZE] for i in range(0, len(todo), _CHUNK_SIZE)]
        for chunk in tqdm(chunks):
            docs = list(Document.objects.filter(pk__in=chunk))
            texts = [doc.text for doc in docs]
            results = m.pipe(texts, disable="textcat")
            for doc, tokens in zip(docs, results):
                fn = os.path.join(dir, str(doc.id))
                tokens.to_disk(fn)

