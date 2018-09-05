import django;
from django.db.models import Count

django.setup()

from actcode.models import Document, Annotation

tofix = list(Document.objects.filter(title=""))
print("Fixing",len(tofix), " titles")
for doc in tofix:
    title = doc.text.strip().split("\n")[0]
    doc.title = title
    doc.save()

dupes = list(Annotation.objects.values('document_id', 'label_id').annotate(n = Count('id')).filter(n__gte=2))
print("Fixing", len(dupes), "duplicate annotations")
for dupe in dupes:
    ids = list(Annotation.objects.filter(document_id = dupe['document_id'], label_id=dupe['label_id']).values_list('pk', flat=True).order_by('id'))
    remove = ids[:-1]
    Annotation.objects.filter(pk__in=remove).delete()

