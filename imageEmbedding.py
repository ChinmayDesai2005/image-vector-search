import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

class ImagetoEmbedding:
    def __init__(self):
        vertexai.init(project="trainingmlteam", location="us-central1")
    
    def generateEmbedding(self, path_to_image):
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
        image = Image.load_from_file(path_to_image)

        embeddings = model.get_embeddings(
            image=image)
        return embeddings.image_embedding



# ite = ImagetoEmbedding()
# print(ite.generateEmbedding(r"C:\Users\Chinmay_D\Desktop\Clothing_images\0ef80b69-2c18-435d-b091-91f401e01ed4.jpg"))