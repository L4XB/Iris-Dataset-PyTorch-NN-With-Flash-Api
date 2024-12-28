from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from .iris_model import Model



class IrisClassificationView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Model()
        model_path = os.path.join(os.path.dirname(__file__), '../../model/iris_model.pt')
        self.model.load_state_dict(torch.load(model_path, weights_only= True))
        self.model.eval()
        
    def post(self, request):
        sepal_length = request.data.get("sepal_length")
        sepal_width = request.data.get("sepal_width")
        petal_length = request.data.get("petal_length")
        petal_width = request.data.get("petal_width")

        if sepal_length is None or sepal_width is None or petal_length is None or petal_width is None:
            return Response({"error": "Missing required parameters"}, status=status.HTTP_400_BAD_REQUEST)
        
        features = torch.tensor([[sepal_length, sepal_width, petal_length, petal_width]], dtype=torch.float32)
        
        
        with torch.no_grad():
            prediction = self.model(features)
            predicted_class = torch.argmax(prediction, dim=1).item()

        class_names = ["Setosa", "Versicolor", "Virginica"]
        predicted_class_name = class_names[predicted_class]

        return Response({"predicted_class": predicted_class_name}, status=status.HTTP_200_OK)




