import kagglehub

# Download latest version
path = kagglehub.dataset_download("imtkaggleteam/acne-computer-vision")

print("Path to dataset files:", path)