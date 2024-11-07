import random
import os

MAX_INSTANCES = 50
MAX_BIT_LENGTH = 4

def generate_data(num_cases=MAX_INSTANCES, max_bit_length=MAX_BIT_LENGTH):
    """
        @about: Generate unique test cases for binary addition with a fixed bit length.

        @params: num_cases -> int, number of test cases to generate.
        @params: max_bit_length -> int, fixed bit length for the binary numbers to be added.
        @return: list of tuples, each tuple is (binary1, binary2, result) for binary addition.
    """
    data = []
    seen_pairs = set()

    while len(data) < num_cases:
        # Generate two random binary nums
        num1 = random.randint(0, 2 ** max_bit_length - 1)
        num2 = random.randint(0, 2 ** max_bit_length - 1)

        # convert to bin string and pad
        binary1 = format(num1, f'0{max_bit_length}b')
        binary2 = format(num2, f'0{max_bit_length}b')
        pair = tuple(sorted((binary1, binary2)))

        # Skip duplicate pairs
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        # Calculate the binary result and pad it
        sum_result = num1 + num2
        result_binary = format(sum_result, f'0{max_bit_length}b')[-max_bit_length:]
        data.append((binary1, binary2, result_binary))

    return data

def get_unique_filename(base_name="dataSet", extension=".txt"):
    """
        @about: Generate a unique filename by incrementing the suffix number.

        @params: base_name -> str, base name for the file.
        @params: extension -> str, file extension.
        @return: str, a unique filename with an incremented number suffix.
    """
    counter = 1
    while True:
        filename = f"{base_name}_{counter}{extension}"
        if not os.path.exists(filename):
            return filename
        counter += 1

def save_data(data):
    """
        @about: Save binary addition data to a unique text file in the specified format.

        @params: data -> list of tuples, binary addition cases.
    """
    filename = get_unique_filename()
    with open(filename, 'w') as file:
        for binary1, binary2, result in data:
            file.write(f"{binary1} {binary2} {result}\n")
    print(f"Data saved to {filename}")

# Generate and save data
data = generate_data(MAX_INSTANCES, max_bit_length=MAX_BIT_LENGTH)
save_data(data)
