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
        linear_reg = LinearRegression(data = params["data"], noise = params["noise"], random_state = params["random_state"], test_size = params["test_size"])
        history, model = linear_reg.train_model(
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
        loss, accuracy = linear_reg.evaluate_model(model)
        y_pred, y_true = linear_reg.predict_model(model, params["sample_number"])
        return Response({
            "raw_data": linear_reg.plot_raw_data(),
            "accuracy_data" : history.history["accuracy"],
            "loss_data" : history.history["loss"],
            "accuracy_plot" : linear_reg.plot_accuracy(history),
            "loss_plot" : linear_reg.plot_loss(history),
            "decision_boundary" : linear_reg.plot_decision_boundary(model),
            "test_loss" : loss,
            "test_accuracy" : accuracy,
            "y_pred" : y_pred,
            "y_true" : y_true
        })

    return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)
        
