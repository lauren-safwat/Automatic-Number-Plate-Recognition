# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pytesseract as pt
import pickle
import re
import easyocr
import os
from pylab import rcParams



# Part One | Object Detection Model
# ------------------------------- #

# Define a list of colors for visualization
classes =['licence']
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

# TFLite Model Path
model_path = './static/models/TFLite_Model/model.tflite'

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()



def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image


def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Retur the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    # Feed the input image to the model
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all outputs from the model
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def car_plate_detection(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    boxes = []
    
    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # Append the bounding box coordinates to the list
        boxes.append(list(obj['bounding_box']))

        # Find the class index of the current object
        class_id = int(obj['class_id'])

        # Draw the bounding box and label on the image
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Make adjustments to make the label visible for all objects
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
        cv2.putText(original_image_np, label, (xmin, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Return the final image and the bounding boxes coordinates
    original_uint8 = original_image_np.astype(np.uint8)
    # original_uint8 = cv2.cvtColor(original_uint8, cv2.COLOR_BGR2RGB)
    return original_uint8, boxes



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #



# Part Two | Optical Charachter Recognition
# --------------------------------------- #

# A function to filter the extra recognized text
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


def apply_OCR(plate, filename, idx, region_threshold):
    # Converting the RGB plate image to gray scale image
    plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    
    ## resize image & perform gaussian blur to smoothen image
    # plate = cv2.resize(plate, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
    # plate = cv2.GaussianBlur(plate, (1,1), 0)

    # histogram equalization
    equ = cv2.equalizeHist(plate)

    # Gaussian blur
    blur = cv2.GaussianBlur(equ, (5, 5), 1)

    # manual thresholding || this threshold might vary!
    th2 = 60 
    equ[equ>=th2] = 255
    equ[equ<th2]  = 0

    rcParams['figure.figsize'] = 8, 16
    
    # Saving the car plate image
    cv2.imwrite('./static/roi/box_{}_{}'.format(str(idx), filename), plate)

    # Applying EasyOCR
    reader = easyocr.Reader(['en'])
    plate_num = reader.readtext(equ, paragraph="False", detail=0)

    # Applying PyTesseract | Deployment Version
    # plate_num = pt.image_to_string(equ, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6 --oem 3')
    # plate_num = re.sub('[\W_]+', '', plate_num)

    return plate_num[-1] if plate_num else 'No text'



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #



def car_plate_recognition(img_path, filename, detection_threshold = 0.5, region_threshold = 0.6):
    # Run inference and draw detection result on the local copy of the original file
    detection_result_image, boxes = car_plate_detection(
        img_path,
        interpreter,
        threshold = detection_threshold
    )
    detection_result_image = cv2.cvtColor(detection_result_image, cv2.COLOR_RGB2BGR)


    # Save the image with detections
    cv2.imwrite('./static/predict/{}'.format(filename), detection_result_image)

    # Read the image and get its dimensions
    image = cv2.imread(img_path)
    width = image.shape[1]
    height = image.shape[0]

    results = []

    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        # Bounding box coordinates
        dimensions = [height, width, height, width]
        roi = [a*b for a,b in zip(box, dimensions)]

        # Cropping the detected car plate
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        
        # Apply OCR to get the car plate number
        plate_number = apply_OCR(region, filename, idx+1, region_threshold)

        # Append the plate number to results
        results.append(plate_number)
    
    return results
