#forms.py
from django import forms

class CoordinateFileForm(forms.Form):
    D48 = forms.FileField()
    D96 = forms.FileField()
