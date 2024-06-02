from django.db import models

# Create your models here.


class Patient(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    health = models.CharField(max_length=100)
    disease = models.CharField(max_length=100)
    video_file = models.FileField(upload_to='videos/', null=True)