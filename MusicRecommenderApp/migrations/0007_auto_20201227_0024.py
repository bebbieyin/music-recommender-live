# Generated by Django 3.1.3 on 2020-12-26 16:24

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('MusicRecommenderApp', '0006_auto_20201226_2031'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='userstracks',
            unique_together={('track_id', 'user_id')},
        ),
    ]
