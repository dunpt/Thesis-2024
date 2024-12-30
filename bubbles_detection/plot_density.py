import pandas as pd
import matplotlib.pyplot as plt

# Define file paths and labels
file_paths = [
    "/home/dunpt1504/Documents/School_project/Quality_assessment/IOU_scores/test_dataset_qs_400/Benign.txt",
    "/home/dunpt1504/Documents/School_project/Quality_assessment/IOU_scores/test_dataset_qs_400/Lesions.txt",
    "/home/dunpt1504/Documents/School_project/Quality_assessment/IOU_scores/test_dataset_qs_400/Uncertain.txt"
]
labels = ["Benign", "Lesions", "Uncertain"]

# Initialize a list to store values
data = []

# Read and sort values from each file
for file_path in file_paths:
    with open(file_path, "r") as file:
        # Read and evaluate each line, then sort by the value (second element)
        file_data = sorted([eval(line.strip())[1] for line in file if line.strip()])
        data.append(file_data)

# Plot each dataset as a scatter plot, using a unique x-position for each label
plt.figure(figsize=(10, 6))

for i, values in enumerate(data):
    # Set x positions for each file
    x_positions = [i + 1] * len(values)  # Unique x-position for each file
    plt.scatter(x_positions, values, label=labels[i], alpha=0.3)

# Set axis labels and title
plt.xticks([1, 2, 3], labels)
plt.xlabel("File Type")
plt.ylabel("Score")
plt.title("Score Distribution by File Type")

# Save the plot as an image
image_path = "test_dataset_qs_400.png"  # Update the path as needed
plt.savefig(image_path)
plt.show()
