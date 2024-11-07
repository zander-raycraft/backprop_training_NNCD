import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from model import FeedForwardModel
from model_test import calculate_accuracy

# Define the two hidden layer structures to compare
HIDDEN_LAYERS_MODEL_1 = [4, 8, 4]
HIDDEN_LAYERS_MODEL_2 = [6, 6, 6]

"""
    @about: Calculate the accuracy of the model on the test data.

    @params: model -> FeedForwardModel, the neural network model to be tested.
    @params: test_data -> list of tuples, each tuple is (inputs, target_outputs).
    @return: float, accuracy as a percentage.
"""
def calculate_accuracy(model, test_data):
    correct_predictions = 0
    total_predictions = len(test_data)

    for inputs, target_outputs in test_data:
        predicted_outputs = model.forward(inputs)
        predicted_outputs = [round(output) for output in predicted_outputs]  # Round to nearest int (0 or 1)

        if predicted_outputs == target_outputs:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


"""
    @about: Initialize and train a FeedForwardModel with specified hidden layers.

    @params: hidden_layers -> list<int>, hidden layer structure for the model.
    @params: training_data -> list<tuple>, dataset to train on.
    @params: epochs -> int, number of epochs for training.
    @return: FeedForwardModel, the trained model.
"""
def train_model(hidden_layers, training_data, epochs=30, learning_rate=0.1):
    input_size = 6
    output_size = 4
    model = FeedForwardModel(size_input=input_size, size_output=output_size, size_hidden=hidden_layers,
                             learning_rate=learning_rate)

    model.train(training_data, epochs)
    return model


def main():
    data_file = os.path.join(os.path.dirname(__file__), "../data/dataSet_1.txt")
    model_1_training_data = FeedForwardModel(size_hidden=[1]).load_data(data_file)  # Load data once for both models

    # Train Model 1 with HIDDEN_LAYERS_MODEL_1 structure
    print("\nTraining Model 1 with hidden layers:", HIDDEN_LAYERS_MODEL_1)
    model_1 = train_model(HIDDEN_LAYERS_MODEL_1, model_1_training_data, epochs=30)

    # Train Model 2 with HIDDEN_LAYERS_MODEL_2 structure
    print("\nTraining Model 2 with hidden layers:", HIDDEN_LAYERS_MODEL_2)
    model_2 = train_model(HIDDEN_LAYERS_MODEL_2, model_1_training_data, epochs=30)

    print("\nEvaluating Models on Training Data...")
    accuracy_1 = calculate_accuracy(model_1, model_1_training_data)
    accuracy_2 = calculate_accuracy(model_2, model_1_training_data)
    print(f"Model 1 Accuracy with hidden layers {HIDDEN_LAYERS_MODEL_1}: {accuracy_1:.2f}%")
    print(f"Model 2 Accuracy with hidden layers {HIDDEN_LAYERS_MODEL_2}: {accuracy_2:.2f}%")

    # Comparison Output
    better_model = "Model 1" if accuracy_1 > accuracy_2 else "Model 2"
    print(f"\n{better_model} performed better on the training dataset.")

if __name__ == "__main__":
    main()
