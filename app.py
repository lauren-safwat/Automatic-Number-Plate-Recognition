# Imports
from fileinput import filename
from re import U
from flask import Flask, render_template, request
import os 
from deeplearning import car_plate_recognition

# Create the webserver interface gateway
app = Flask(__name__)

# Buid Paths
Base_Path   = os.getcwd()
Upload_Path = os.path.join(Base_Path, 'static/upload/')  


## This is the home route 
@app.route('/')
def home():
    return render_template('home.html')


## This is the detect page route | Accepting both get and post requests
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    # Recive the file from the button and save to static/upload 
    if request.method == 'POST':
        # Just to Recive file from the submit button
        uploaded_image = request.files['image_name']
        file_name      = uploaded_image.filename

        # Get path to be saved then save it.
        path_to_save   = os.path.join(Upload_Path, file_name)
        uploaded_image.save(path_to_save)
        
        # Call OCR Function
        results = car_plate_recognition(path_to_save, file_name)

        nBoxes = len(results)
        img_paths = ['box_{}_{}'.format(str(i+1), file_name) for i in range(nBoxes)]

        return render_template('detect.html',
                               upload = True,
                               nBoxes = nBoxes,
                               uploaded_img = file_name,
                               img_paths = img_paths,
                               ocr_text = results)

    
    return render_template('detect.html', upload=False)


# Run the app
if __name__ == "__main__":
    app.run(debug=True) 