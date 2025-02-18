from rest_framework import serializers

class LinearRegressionSerializer(serializers.Serializer):
    data = serializers.IntegerField(default=10000)
    noise = serializers.FloatField(default=0.2)
    random_state = serializers.IntegerField(default=42)
    test_size = serializers.FloatField(default=0.2)
    nooflayers = serializers.IntegerField(default=2)
    noofneurons = serializers.ListField(child=serializers.IntegerField(), default=[10, 10])
    activation = serializers.ListField(child=serializers.CharField(), default=["relu", "relu"])
    optimizer = serializers.CharField(default="adam")
    loss = serializers.CharField(default="binary_crossentropy")
    metrics = serializers.ListField(child=serializers.CharField(), default=["accuracy"])
    epochs = serializers.IntegerField(default=10)
    validation_split = serializers.FloatField(default=0.2)
    batch_size = serializers.IntegerField(default=32)
    sample_number = serializers.IntegerField(default=0)
