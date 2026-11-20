from django import forms


class KosForm(forms.Form):
    jarak_km = forms.FloatField(
        label="Jarak ke kampus (km)",
        min_value=0.6,
        max_value=5,
        widget=forms.NumberInput(attrs={"step": "0.1", "min": "0.6", "max": "5"}),
    )
    rating = forms.FloatField(
        label="Rating kos (1-5)",
        min_value=3.1,
        max_value=4.9,
        widget=forms.NumberInput(attrs={"step": "0.1", "min": "3.1", "max": "4.9"}),
    )
    ac = forms.ChoiceField(
        label="AC",
        choices=[("ya", "Ya"), ("tidak", "Tidak")],
    )