# Define the file path
file_path = '/home/dunpt1504/Documents/School_project/Quality_assessment/IOU_scores/test_dataset_qs_EAD2019/Lesions.txt'

# Initialize variables
max_value = float('-inf')
max_line = ""

# Open the file and process each line
with open(file_path, 'r') as file:
    for line in file:
        # Split the line to extract the value
        parts = line.strip('[]\n').split(', ')
        uuid = parts[0].strip("'")
        value = float(parts[1])
        
        # Update max_value and max_line if a larger value is found
        if value > max_value:
            max_value = value
            max_line = line.strip()

# Print the result
print(f"Line with max value: {max_line}")
print(f"Max value: {max_value}")
