from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.src.model import Model


class IrisClassificationView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = Model()
        self.model.load_state_dict(torch.load('model/iris_model.pt'))
        self.model.eval()
        
    def post(self, request):
        sepal_length = request.data.get("sepal_length")
        sepal_width = request.data.get("sepal_width")
        petal_length = request.data.get("petal_length")
        petal_width = request.data.get("petal_width")

        if sepal_length is None or sepal_width is None or petal_length is None or petal_width is None:
            return Response({"error": "Missing required parameters"}, status=status.HTTP_400_BAD_REQUEST)
        
        
        
        return Response({"message" : "Endpoint Works finde"}, status= status.HTTP_200_OK)
