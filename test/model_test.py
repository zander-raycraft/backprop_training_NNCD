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


"""
    @about: Get two binary inputs from the user and format them for the model.

    @params: max_bit_length -> int, fixed bit length for binary inputs.
    @return: list of integers, formatted inputs for the model.
"""
def get_user_input(max_bit_length):
    binary1 = input(f"Enter the first {max_bit_length}-bit binary number: ")
    binary2 = input(f"Enter the second {max_bit_length}-bit binary number: ")
    if len(binary1) != max_bit_length or len(binary2) != max_bit_length:
        print(f"Error: Both inputs must be {max_bit_length} bits long.")
        return None

    # Convert binary strings to a list of integers
    binary1_bits = [int(bit) for bit in binary1]
    binary2_bits = [int(bit) for bit in binary2]

    return binary1_bits + binary2_bits

"""
    @about: Print the weights of each layer in the model, including input, hidden, and output layers.

    @params: model -> FeedForwardModel, the neural network model.
    @params: label -> str, label to indicate whether weights are before or after training.
"""
def print_weights(model, label):
    print(f"\n{label} Weights:")

    # Hidden layer weights
    for i, hidden_layer in enumerate(model.hidden_layers):
        print(f"\nHidden Layer {i + 1} Weights:")
        for j, weight_vector in enumerate(hidden_layer.get_weights()):
            formatted_vector = ", ".join(f"{weight:.4f}" for weight in weight_vector)
            print(f"  Node {j + 1}: [{formatted_vector}]")

    # Output layer weights
    print("\nOutput Layer Weights:")
    for i, weight_vector in enumerate(model.output_layer.get_weights()):
        formatted_vector = ", ".join(f"{weight:.4f}" for weight in weight_vector)
        print(f"  Node {i + 1}: [{formatted_vector}]")

def main():

    # PARAMS
    input_size = 6
    output_size = 4
    hidden_sizes = [4, 4]
    learning_rate = 0.1
    epochs = 30
    max_bit_length = 3

    model = FeedForwardModel(size_input=input_size, size_output=output_size, size_hidden=hidden_sizes,
                             learning_rate=learning_rate)
    data_file = os.path.join(os.path.dirname(__file__), "../data/dataSet_1.txt")
    training_data = model.load_data(data_file)

    # Print initial weights before training
    print_weights(model, "Initial")

    # Train the model
    print("Starting training...")
    model.train(training_data, epochs)
    print("Training complete.")
    print_weights(model, "Final")

    # Test the model and calculate accuracy
    print("Testing model...")
    accuracy = calculate_accuracy(model, training_data)
    print(f"Model accuracy on training data: {accuracy:.2f}%")

    # while True:
    #     print("\n--- Binary Addition Test ---")
    #     user_input = get_user_input(max_bit_length)
    #     if user_input is None:
    #         continue
    #     predicted_output = model.forward(user_input)
    #     predicted_output = [round(output) for output in predicted_output]
    #     predicted_binary = ''.join(str(bit) for bit in predicted_output)
    #     print(f"Model's binary addition result: {predicted_binary}")
    #
    #     if input("Try another addition? (y/n): ").lower() != 'y':
    #         break

if __name__ == "__main__":
    main()
