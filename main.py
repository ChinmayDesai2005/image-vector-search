import os 
from flask import *
from werkzeug.utils import secure_filename
from api import CloudVision, ImageFunctions
from PIL import Image
from imageEmbedding import ImagetoEmbedding
from vectorSearch import VectorSearch

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def main():
    return render_template("index.html")

@app.route('/success', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        f = request.files['file'] 
        if f:
            choice = request.form.get('tools')
            print(choice)
            path_to_content = f"{UPLOAD_FOLDER}/{f.filename}"
            f.save(f'static/{path_to_content}')
            text = None
            match choice:
                case "text_detection":
                    text = CloudVision.text_detection(path_to_content = f'static/{path_to_content}')
                case "landmark_detection":
                    text = CloudVision.detect_landmarks(path_to_content = f'static/{path_to_content}')
                case "object_detection":
                    text = ImageFunctions.analyze_and_crop_images(f'static/{path_to_content}')
            if not text:
                text = "No result. Please check if you are using the correct tool!"
            return render_template("Acknowledgement.html", name = f.filename, path_to_content = path_to_content, text = text)

@app.route('/vector-search')
def vector_search():
    return render_template('vector1.html')

@app.route('/vector_results', methods=['POST'])
def vector_results():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            f.save(f"static/file.jpg")
            objects = ImageFunctions.analyze_and_crop_images(f"static/file.jpg")
            query_embeds = []
            ite = ImagetoEmbedding()
            for object_ in objects:
                query_embeds.append(ite.generateEmbedding(object_))
            result_images = VectorSearch.search_image(query_embeds)
            return render_template('vector_results.html', image_urls=result_images, original_image="static/file.jpg")


if __name__ == '__main__':   
    app.run(debug=True)