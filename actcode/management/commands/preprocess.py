from tqdm import tqdm
from django.core.management.base import BaseCommand
from actcode.models import Project, Document


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument("project_id")

    def handle(self, *args, **options):
        project = Project.objects.get(pk=options['project_id'])
        docids = list(project.document_set.filter(gold=False, tokens__isnull=True).values_list("id", flat=True))
        if not docids:
            print("All documents already preprocessed")
            return
        print("Preprocessing {} documents".format(len(docids)))
        m = project.get_model()

        _CHUNK_SIZE=10
        chunks = [docids[i:i+_CHUNK_SIZE] for i in range(0, len(docids), _CHUNK_SIZE)]
        for chunk in tqdm(chunks):
            docs = Document.objects.filter(pk__in=chunk)
            texts = [doc.text for doc in docs]
            results = m.pipe(texts, disable="textcat")
            for doc, tokens in zip(docs, results):
                doc.tokens = tokens.to_bytes()
                doc.save()
