from django import forms


class KosForm(forms.Form):
    jarak_km = forms.FloatField(label="Jarak ke kampus (km)", min_value=0, max_value=20)
    rating = forms.FloatField(label="Rating kos (1-5)", min_value=1, max_value=5)
    ac = forms.ChoiceField(label="AC", choices=[("ya", "Ya"), ("tidak", "Tidak")])