from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from .serializers import LinearRegressionSerializer
from .linearregression import LinearRegression

@api_view(["POST"])
def linear_regression(request):
    serializer = LinearRegressionSerializer(data = request.data)
    if serializer.is_valid():
        params = serializer.validated_data
        model = LinearRegression(data = params["data"], noise = params["noise"], random_state = params["random_state"], test_size = params["test_size"])
        model.input_data()
        model.split_data()
        history = model.train_model(
            nooflayers = params["nooflayers"],
            noofneurons = params["noofneurons"],
            activation = params["activation"],
            optimizer = params["optimizer"],
            loss = params["loss"],
            metrics = params["metrics"],
            epochs = params["epochs"],
            validation_split = params["validation_split"],
            batch_size = params["batch_size"]
        )
        return Response({
            "raw_data": model.plot_raw_data(),
            "accuracy": model.plot_accuracy(history),
            "loss": model.plot_loss(history),
        })

    return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)
        
