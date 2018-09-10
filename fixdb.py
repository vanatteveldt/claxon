import django;
from django.db.models import Count
from django.db import transaction

django.setup()

from actcode.models import Document, Annotation

tofix = list(Document.objects.filter(title=""))
print("Fixing",len(tofix), " titles")
with transaction.atomic():
    for i, doc in enumerate(tofix):
        title = doc.text.strip().split("\n")[0]
        doc.title = title
        doc.save()
        if not i % 1000: print(i)

dupes = list(Annotation.objects.values('document_id', 'label_id').annotate(n = Count('id')).filter(n__gte=2))
print("Fixing", len(dupes), "duplicate annotations")
with transaction.atomic():
    for i, dupe in enumerate(dupes):
        ids = list(Annotation.objects.filter(document_id = dupe['document_id'], label_id=dupe['label_id']).values_list('pk', flat=True).order_by('id'))
        remove = ids[:-1]
        Annotation.objects.filter(pk__in=remove).delete()
        if not i % 1000: print(i)
