import os
import csv

# Define the path to your dataset's parent directory
dataset_path = '/home/amit/SkinCareAI/acne_test_dataset/test'
dataset1_path= '/home/amit/SkinCareAI/acne_test_dataset/train'
# Define the name for the output CSV file
output_csv_file = 'image_data.csv'
output_csv_file1 = 'image_data1.csv'
# Initialize a list to hold all the data
data = []

# Walk through the directories to find image files
for root, dirs, files in os.walk(dataset_path):
    # The label is the name of the directory one level below the root
    # e.g., for /home/amit/SkinCareAI/acne_test_dataset/test/acne, the label is 'acne'
    label = os.path.basename(root)

    for file in files:
        # Check for common image file extensions
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Create the full path to the image
            image_path = os.path.join(root, file)

            # Append the image path and label to our data list
            data.append([image_path, label])
# Walk through the directories to find image files
for root, dirs, files in os.walk(dataset1_path):
    # The label is the name of the directory one level below the root
    # e.g., for /home/amit/SkinCareAI/acne_test_dataset/test/acne, the label is 'acne'
    label = os.path.basename(root)

    for file in files:
        # Check for common image file extensions
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Create the full path to the image
            image_path = os.path.join(root, file)

            # Append the image path and label to our data list
            data.append([image_path, label])

# Write the data to a CSV file
with open(output_csv_file1, 'w', newline='') as csvfile:
    # Define the column headers
    fieldnames = ['image_path', 'label']
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(fieldnames)

    # Write the data rows
    writer.writerows(data)
with open(output_csv_file, 'w', newline='') as csvfile:
    # Define the column headers
    fieldnames = ['image_path', 'label']
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(fieldnames)

    # Write the data rows
    writer.writerows(data)


print(f"Successfully created '{output_csv_file}' with {len(data)} images.")
print(f"Successfully created '{output_csv_file1}' with {len(data)} images.")