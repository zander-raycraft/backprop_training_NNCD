import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from model import FeedForwardModel

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


def main():
    input_size = 8
    output_size = 4
    hidden_sizes = [4, 4, 4]
    learning_rate = 0.2


    epochs = 20

    # Initialize the model
    model = FeedForwardModel(size_input=input_size, size_output=output_size, size_hidden=hidden_sizes,
                             learning_rate=learning_rate)
    data_file = os.path.join(os.path.dirname(__file__), "../data/dataSet_2.txt")
    training_data = model.load_data(data_file)

    # Train the model
    print("Starting training...")
    model.train(training_data, epochs)
    print("Training complete.")

    # Test the model and calculate accuracy
    print("Testing model...")
    accuracy = calculate_accuracy(model, training_data)
    print(f"Model accuracy on training data: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
