from PIL import Image
import os
input_folder = "wild_1"  # Replace with your input folder containing images
output_folder = "wild_128"  # Replace with your output folder for resized images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Loop through each file in the input folder
for file in files:
    if file.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
        # Open the image
        img_path = os.path.join(input_folder, file)
        img = Image.open(img_path)

        # Resize the image to 128x128
        resized_img = img.resize((128, 128), Image.BICUBIC)

        # Save the resized image to the output folder
        output_path = os.path.join(output_folder, f"resized_{file}")
        resized_img.save(output_path)

        print(f"Resized {file} and saved to {output_path}")
