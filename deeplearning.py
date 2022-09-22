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


# from tf.keras.preprocessing.image import load_img, img_to_array


## Load Model
# model           = tf.keras.models.load_model('../Tensorflow/workspace/models/my_ssd_mobnet/export/saved_model/.')


# tf.config.experimental.set_visible_devices([], 'GPU')

################################ Start Of Lauren's Function ###################################

# Part One | Object Detection Model
# ------------------------------- #

# # Required Paths
# config_path = '../Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config'
# ckpt_path = '../Tensorflow/workspace/models/my_ssd_mobnet/ckpt-11'
# label_map_path = '../Tensorflow/workspace/annotations/label_map.pbtxt'

# # Load pipeline config and build a detection model
# configs = config_util.get_configs_from_pipeline_file(config_path)
# detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# # Restore checkpoint
# ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# ckpt.restore(ckpt_path).expect_partial()

# # Get label_map
# category_index = label_map_util.create_category_index_from_labelmap(label_map_path)

# @tf.function
# def detect_fn(image):
#     image, shapes = detection_model.preprocess(image)
#     prediction_dict = detection_model.predict(image, shapes)
#     detections = detection_model.postprocess(prediction_dict, shapes)
#     return detections


# def car_plate_detection(img_path, filename, detection_threshold):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     image_np = np.array(img)

#     input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
#     detections = detect_fn(input_tensor)

#     num_detections = int(detections.pop('num_detections'))
#     detections = {key: value[0, :num_detections].numpy()
#                 for key, value in detections.items()}
#     detections['num_detections'] = num_detections

#     # detection_classes should be ints.
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

#     label_id_offset = 1
#     image_np_with_detections = image_np.copy()

#     viz_utils.visualize_boxes_and_labels_on_image_array(
#                 image_np_with_detections,
#                 detections['detection_boxes'],
#                 detections['detection_classes']+label_id_offset,
#                 detections['detection_scores'],
#                 category_index,
#                 use_normalized_coordinates=True,
#                 max_boxes_to_draw=5,
#                 min_score_thresh=detection_threshold,
#                 agnostic_mode=False)

#     cv2.imwrite('./static/predict/{}'.format(filename), image_np_with_detections)
    
#     return detections


# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #


# # Part Two | Optical Charachter Recognition
# # --------------------------------------- #

# # A function to filter the extra recognized text
# def filter_text(region, ocr_result, region_threshold):
#     rectangle_size = region.shape[0]*region.shape[1]
    
#     plate = [] 
#     for result in ocr_result:
#         length = np.sum(np.subtract(result[0][1], result[0][0]))
#         height = np.sum(np.subtract(result[0][2], result[0][1]))
        
#         if length*height / rectangle_size > region_threshold:
#             plate.append(result[1])
#     return plate


# def OCR(img_path, filename, detection_threshold = 0.5, region_threshold = 0.6):
#     # Pass Image to object detection function
#     # image = np.array(tf.keras.preprocessing.image.load_img(path))
#     image = cv2.imread(img_path)
#     detections = car_plate_detection(img_path, filename, detection_threshold)

#     # Filter the detections with confidence score above the detection threshold
#     scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
#     boxes = detections['detection_boxes'][:len(scores)]
#     classes = detections['detection_classes'][:len(scores)]

#     # Get the image dimensions
#     width = image.shape[1]
#     height = image.shape[0]

#     results = []

#     # Apply ROI filtering and OCR
#     for idx, box in enumerate(boxes):
#         # Bounding box coordinates
#         roi = box*[height, width, height, width]

#         # Cropping the detected car plate
#         region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]

#         # Image Preprocessing

#         ## Converting the RGB plate image to gray scale image
#         region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

#         ## Applying gaussian blur to smoothen the plate image
#         region = cv2.GaussianBlur(region, (5,5), 0)

#         ## Resizing the image
#         # region = cv2.resize(region, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)

#         # Saving the car plate image
#         cv2.imwrite('./static/roi/box_{}_{}'.format(str(idx+1), filename), region)

#         # Applying OCR
#         reader = easyocr.Reader(['en'])
#         plate_number = reader.readtext(region, paragraph="False", detail=0)
#         # plate_number = filter_text(region, ocr_result, region_threshold)

#         # Append the plate number to results
#         results.append(plate_number[-1])
        
#         plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
#         plt.show()
#         print(plate_number)
    
#     return results

    





    # # grayscale region within bounding box
    # gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    
    # # resize image & perform gaussian blur to smoothen image
    # gray = cv2.resize(gray, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # # create rectangular kernel for dilation | Make regions more clear by applying dialation
    # rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    # # find contours of regions of interest within license plate
    # try:
    #     contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # except:
    #     ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # sort contours left-to-right
    # sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # # create copy of gray image
    # im2 = gray.copy()
    # # create blank string to hold license plate number
    # plate_num = ""
    # # loop through contours and find individual letters and numbers in license plate
    # for cnt in sorted_contours:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     height, width = im2.shape
    #     # if height of box is not tall enough relative to total height then skip
    #     if height / float(h) > 6: continue

    #     ratio = h / float(w)
    #     # if height to width ratio is less than 1.5 skip
    #     if ratio < 1.5: continue

    #     # if width is not wide enough relative to total width then skip
    #     if width / float(w) > 15: continue

    #     area = h * w
    #     # if area is less than 100 pixels skip
    #     if area < 100: continue

    #     # draw the rectangle
    #     rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
    #     # grab character region of image
    #     roi = thresh[y-5:y+h+5, x-5:x+w+5]
    #     # perfrom bitwise not to flip image to black text on white background
    #     roi = cv2.bitwise_not(roi)
    #     # perform another blur on character region
    #     roi = cv2.medianBlur(roi, 5)
   
    #     cv2.imwrite('./static/roi/{}'.format(filename), im2)
    #     text = pt.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    #     # clean tesseract text by removing any unwanted blank spaces
    #     clean_text = re.sub('[\W_]+', '', text)
    #     plate_num += clean_text

    # if plate_num != None:
    #     print("License Plate #: ", plate_num)
    
    # roi = box

    # roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR) 
    # cv2.imwrite('./static/roi/{}'.format(filename), roi_bgr)
    # text = pt.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6 --oem 3')
    # text = re.sub('[\W_]+', '', text)
    # print(text)
    # return result[-1] if result else "No Text"

################################ End Of Lauren's Function ###################################





################################## Start Of Yasser's Function ###################################

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
    
    ## Saving the car plate image
    cv2.imwrite('./static/roi/box_{}_{}'.format(str(idx), filename), plate)

    # Applying EasyOCR
    reader = easyocr.Reader(['en'])
    ocr_result = reader.readtext(equ, paragraph="False", detail=0)

    # ## Applying PyTesseract | Deployment Version
    # ocr_result = pt.image_to_string(equ, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6 --oem 3')


    print(ocr_result)

    # Resizing the image
    # region = cv2.resize(region, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)

    # Saving the car plate image
    # cv2.imwrite('./static/roi/box_{}_{}'.format(str(idx), filename), plate)

    # #### Applying OCR
    # reader = easyocr.Reader(['en'])
    # ocr_result = reader.readtext(plate, paragraph="False", detail=0)

    # print(ocr_result)

    # plate_number = filter_text(region, ocr_result, region_threshold)

    return ocr_result[-1] if ocr_result else 'No text'
    
    # # ## Deployment Version
    # return ocr_result



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


# Part One | Object Detection Model 
# def object_detection(path, filename):
#     # read image
#     image = tf.keras.preprocessing.image.load_img(path) # PIL object
#     image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
#     image1 = tf.keras.preprocessing.image.load_img(path,target_size=(224,224))
    
#     # data preprocessing
#     image_arr_224 = tf.keras.preprocessing.image.img_to_array(image1)/255.0  # convert into array and get the normalized output
#     h,w,d = image.shape
#     test_arr = image_arr_224.reshape(1,224,224,3)
    
#     # make predictions | get back coords
#     coords = model.predict(test_arr)
    
#     # denormalize the values
#     denorm = np.array([w,w,h,h])
#     coords = coords * denorm
#     coords = coords.astype(np.int32)
    
#     # draw bounding on top the image
#     xmin, xmax,ymin,ymax = coords[0]
#     pt1 =(xmin,ymin)
#     pt2 =(xmax,ymax)
#     print(pt1, pt2)
#     cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    
#     # Convert to bgr format | Savve the Image 
#     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
#     cv2.imwrite('./static/predict/{}'.format(filename), image_bgr)

#     return coords


# # Part Two | Optical Charachter Recognition Model 
# def OCR(path, filename):
#     ## Pass Image to object detection function
#     img = np.array(tf.keras.preprocessing.image.load_img(path))
#     cods = object_detection(path, filename)
#     # cods = object_detection(path)
#     print(cods)

#     ## Get the 4 coords
#     xmin ,xmax,ymin,ymax = cods[0]
    
#     # get the subimage that makes up the bounded region and take an additional 15 pixels on each side
#     box = img[int(ymin)-15:int(ymax)+15, int(xmin)-15:int(xmax)+15]
    
#     gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)

#     # resize image & perform gaussian blur to smoothen image
#     gray = cv2.resize(blur, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)

    
#     cv2.imwrite('./static/roi/{}'.format(filename), gray)

#     reader = easyocr.Reader(['en'])
#     result = reader.readtext(gray, paragraph="False", detail=0)
    
#     print(result)



#     # # grayscale region within bounding box
#     # gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    
#     # # resize image & perform gaussian blur to smoothen image
#     # gray = cv2.resize(gray, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
#     # blur = cv2.GaussianBlur(gray, (5,5), 0)
#     # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
#     # # create rectangular kernel for dilation | Make regions more clear by applying dialation
#     # rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#     # dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

#     # # find contours of regions of interest within license plate
#     # try:
#     #     contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # except:
#     #     ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # # sort contours left-to-right
#     # sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
#     # # create copy of gray image
#     # im2 = gray.copy()
#     # # create blank string to hold license plate number
#     # plate_num = ""
#     # # loop through contours and find individual letters and numbers in license plate
#     # for cnt in sorted_contours:
#     #     x,y,w,h = cv2.boundingRect(cnt)
#     #     height, width = im2.shape
#     #     # if height of box is not tall enough relative to total height then skip
#     #     if height / float(h) > 6: continue

#     #     ratio = h / float(w)
#     #     # if height to width ratio is less than 1.5 skip
#     #     if ratio < 1.5: continue

#     #     # if width is not wide enough relative to total width then skip
#     #     if width / float(w) > 15: continue

#     #     area = h * w
#     #     # if area is less than 100 pixels skip
#     #     if area < 100: continue

#     #     # draw the rectangle
#     #     rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
#     #     # grab character region of image
#     #     roi = thresh[y-5:y+h+5, x-5:x+w+5]
#     #     # perfrom bitwise not to flip image to black text on white background
#     #     roi = cv2.bitwise_not(roi)
#     #     # perform another blur on character region
#     #     roi = cv2.medianBlur(roi, 5)
   
#     #     cv2.imwrite('./static/roi/{}'.format(filename), im2)
#     #     text = pt.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
#     #     # clean tesseract text by removing any unwanted blank spaces
#     #     clean_text = re.sub('[\W_]+', '', text)
#     #     plate_num += clean_text

#     # if plate_num != None:
#     #     print("License Plate #: ", plate_num)
    
#     # roi = box

#     # roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR) 
#     # cv2.imwrite('./static/roi/{}'.format(filename), roi_bgr)
#     # text = pt.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6 --oem 3')
#     # text = re.sub('[\W_]+', '', text)
#     # print(text)
#     return result[-1] if result else "No Text"