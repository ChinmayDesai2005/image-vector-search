from google.cloud import aiplatform
from api import ImageFunctions
import json

aiplatform.init(project="trainingmlteam", location="asia-south1")

class VectorSearch():
    def search_image(query_embeds):

        with open('alternate.json', 'r') as file:
            cloth_names = json.load(file)

        my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name="projects/trainingmlteam/locations/asia-south1/indexEndpoints/INDEX_ENDPOINT_ID"
            )

            # Query the index endpoint for the nearest neighbors.
        resp = my_index_endpoint.find_neighbors(
            deployed_index_id="DEPLOYED_INDEX_ID",
            queries=query_embeds,
            num_neighbors=5,
        )

        print(resp)
        result_images = []
        for res in resp[0]:
            for names in cloth_names:
                if res.id == str(names['id']):
                    print(names['name'])
                    result_images.append(ImageFunctions.decode_from_base64(f"{res.id}_{names['name']}", names['image']))
        
        return result_images
