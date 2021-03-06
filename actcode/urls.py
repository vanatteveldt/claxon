from django.conf.urls import url

from actcode.views import ProjectView, CodeView, ProjectListView, ActiveCodeView, FilterView, ActiveCodeStartView, \
    PredictionsView

app_name = 'reviews'
urlpatterns = [
    url("^$", ProjectListView.as_view(), name="index"),
    url("^project/(?P<project>[0-9]+)/$", ProjectView.as_view(), name="project"),
    url("^project/(?P<project>[0-9]+)/code-gold/(?P<label>[0-9]+)/$", CodeView.as_view(), name="code-gold"),
    url("^project/(?P<project>[0-9]+)/learn-start/(?P<label>[0-9]+)/$", ActiveCodeStartView.as_view(), name="code-start"),
    url("^project/(?P<project>[0-9]+)/learn/(?P<session>[0-9]+)/$", ActiveCodeView.as_view(), name="code-learn"),
    url("^project/(?P<project>[0-9]+)/filter/(?P<label>[0-9]+)/$", FilterView.as_view(), name="code-filter"),
    url("^project/(?P<project>[0-9]+)/predictions/$", PredictionsView.as_view(), name="predictions"),
]