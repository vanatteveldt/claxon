# Generated by Django 2.1 on 2018-08-20 10:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('actcode', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='label',
            name='eval_task',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='label',
            name='fn',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='label',
            name='fp',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='label',
            name='last_eval',
            field=models.DateTimeField(null=True),
        ),
        migrations.AddField(
            model_name='label',
            name='tp',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='project',
            name='last_model',
            field=models.DateTimeField(null=True),
        ),
        migrations.AddField(
            model_name='project',
            name='model_location',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='project',
            name='model_task',
            field=models.TextField(null=True),
        ),
    ]