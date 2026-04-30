import os
from PIL import Image

def resize_images(input_folder = "", target_width=400, output_folder="resized"):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through images in current directory
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            print(f"Processing {filename}...")

            # Load image
            img = Image.open(f"{input_folder}/{filename}")
            w, h = img.size

            # Compute new height: target_width * h / w
            new_h = int(target_width * h / w)

            # Resize
            img_resized = img.resize((target_width, new_h), Image.LANCZOS)

            # Save into new folder
            out_path = os.path.join(output_folder, filename)
            img_resized.save(out_path, quality=90)

    print("Done!")

if __name__ == "__main__":
    dataset = "room"
    inputFolder = f"data/{dataset}/images"
    outputFolder = f"data/{dataset}/images_400"
    resize_images(input_folder = inputFolder, output_folder=outputFolder)
