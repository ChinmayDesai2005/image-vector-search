from google.cloud import vision
from PIL import Image

class CloudVision:
    def text_detection(path_to_content):

        client = vision.ImageAnnotatorClient()

        with open(path_to_content, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )
        
        if texts:
            return texts[0].description
        return None
    
    def detect_landmarks(path_to_content):

        client = vision.ImageAnnotatorClient()

        with open(path_to_content, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.landmark_detection(image=image)
        landmarks = response.landmark_annotations

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        if landmarks:
            return landmarks[0].description
        return None
    
    def localize_objects(path):
        client = vision.ImageAnnotatorClient()

        with open(path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        objects = client.object_localization(image=image).localized_object_annotations

        return objects

class ImageFunctions:
    def analyze_and_crop_images(path_to_image):
        images = []
        original = Image.open(path_to_image)

        #analyze using CloudVision
        objects = CloudVision.localize_objects(path_to_image)
        print(f"Number of objects found: {len(objects)}")

        object_names = []
        #Crop and save images
        for object_ in objects:
            # print(object_)
            vertices = []
            object_names.append(object_.name)
            for vertex in object_.bounding_poly.normalized_vertices[::2]:
                vertices.append(vertex.x * original.width)
                vertices.append(vertex.y * original.height)
            temp = original.crop(tuple(vertices))
            temp.save(f"objects/{object_.name}.{original.format}")
            images.append(f"objects/{object_.name}.{original.format}")

        return list(set(images))
    
    def decode_from_base64(image_name, encoded):
        import io, base64
        from PIL import Image

        img = Image.open(io.BytesIO(base64.decodebytes(bytes(encoded, "utf-8"))))
        img.save(f'static/results/{image_name}')
        return f'static/results/{image_name}'
    

# print(ImageFunctions.analyze_and_crop_images(r"static\uploads\0e6871ee-7246-48ad-a356-a7fdde9878d2.jpg"))