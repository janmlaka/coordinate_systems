from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

# Create your models here.

class TextFile(models.Model):
    content = models.TextField()


class YourModel(models.Model):
    D48_file = models.FileField(upload_to='uploads/')
    D96_file = models.FileField(upload_to='uploads/')

class Coordinates(models.Model):
    coosys = models.CharField(max_length=50)
    x_coo = models.FloatField(validators=[MinValueValidator(1), MaxValueValidator(10)])
    y_coo = models.FloatField(validators=[MinValueValidator(1), MaxValueValidator(10)])
    z_coo = models.FloatField(validators=[MinValueValidator(1), MaxValueValidator(10)])

    def __str__(self) -> str:
        return f"{self.coosys} {self.x_coo} {self.y_coo} {self.z_coo}"