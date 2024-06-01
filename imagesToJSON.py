from imageEmbedding import ImagetoEmbedding
ite = ImagetoEmbedding()
from api import ImageFunctions
from io import BytesIO
import base64, json
from PIL import Image
from os import listdir
from os.path import isfile, join

images_folder = r"path_to_dataset_images"

original_images = [rf"{images_folder}\{f}" for f in listdir(images_folder) if isfile(join(images_folder, f))]

for original_image in original_images[:200]:
    images = ImageFunctions.analyze_and_crop_images(original_image)

    try:
        with open("alternate.json", 'r') as file:
            imageData = json.load(file)
    except FileNotFoundError:
        imageData = []

    for image in images:
        filename = image
        image_embedding = ite.generateEmbedding(filename)
        with open(filename, 'rb') as imgfile:
            base64_bytes = base64.b64encode(imgfile.read())
            base64_encoded = base64_bytes.decode()
            data={
                'id': len(imageData) + 1,
                # 'name':(filename.split("/")[-1]),
                # 'image': base64_encoded,
                'embedding':image_embedding
                }
            comp_data = {
                'id': len(imageData) + 1,
                'name':(filename.split("/")[-1]),
                'image': base64_encoded,
            }

            with open("embeddings.json", "a") as file:
                json.dump(data, file)
                file.write("\n")
        # Append new data to existing JSON data
        imageData.append(comp_data)
    # Write the updated data back to the JSON file
    with open("alternate.json", 'w') as file:
        json.dump(imageData, file)
