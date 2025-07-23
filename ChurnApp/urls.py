from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
		     path("Visualization.html", views.Visualization, name="Visualization"),
		     path("VisualizationAction", views.VisualizationAction, name="VisualizationAction"),
		     path("Predict.html", views.Predict, name="Predict"),
		     path("PredictAction", views.PredictAction, name="PredictAction"),		       	     
		    ]