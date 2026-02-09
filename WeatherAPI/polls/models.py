from django.db import models

class StationBiasSample(models.Model):
    station_id = models.CharField(max_length=16, db_index=True)
    date = models.DateField(db_index=True)

    forecast_high_f = models.FloatField()
    observed_high_f = models.FloatField()
    error_f = models.FloatField()  # observed - forecast

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("station_id", "date")
        indexes = [
            models.Index(fields=["station_id", "date"]),
        ]
