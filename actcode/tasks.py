from huey.contrib.djhuey import db_task
import logging

import actcode.ml
from actcode.models import Session


@db_task()
def populate_todo(session_id, n):
    s = Session.objects.get(pk=session_id)
    print("Populating TODO for session", s)
    actcode.ml.ActiveLearn(s).do_populate_todo(n)
    s.populate_task_id = None   # more efficient to do 1 save, but meh
    s.save()
    print("DONE")
