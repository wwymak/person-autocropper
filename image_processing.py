import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
from collections import defaultdict
from io import StringIO
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from operator import itemgetter
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder for images to crop",type=str)
args = parser.parse_args()
imagefolder = args.folder

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './object_detection/' + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './object_detection/'  + 'data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def image_processing_pipeline(imagefolder):
    input_images_path = glob.glob(imagefolder + '/*.jpg')
    def image_generator():
        for x in range(0, len(input_images_path)+1, 5):
            yield input_images_path[ x : x+ 5]
    def parse_boxes_on_image_array(image,boxes,
                                  classes,
                                  scores,
                                  category_index,
                                  use_normalized_coordinates=False,
                                  max_boxes_to_draw=20,
                                  min_score_thresh=.5,
                                  agnostic_mode=False,
                                  line_thickness=4):
        """
            modification of the function from the tf example:
            Overlay labeled boxes on an image with formatted scores and label names.
            This function groups boxes that correspond to the same location
            and creates a display string for each detection and overlays these
            on the image. Note that this function modifies the image in place, and returns
            that same image.
            Args:
            image: uint8 numpy array with shape (img_height, img_width, 3)
            boxes: a numpy array of shape [N, 4]
            classes: a numpy array of shape [N]. Note that class indices are 1-based,
              and match the keys in the label map.
            scores: a numpy array of shape [N] or None.  If scores=None, then
              this function assumes that the boxes to be plotted are groundtruth
              boxes and plot all boxes as black with no classes or scores.
            category_index: a dict containing category dictionaries (each holding
              category index `id` and category name `name`) keyed by category indices.

            max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
              all boxes.
            min_score_thresh: minimum score threshold for a box to be visualized
            agnostic_mode: boolean (default: False) controlling whether to evaluate in
              class-agnostic mode or not.  This mode will display scores but ignore
              classes.
            line_thickness: integer (default: 4) controlling line width of the boxes.
            Returns:
                uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes,
                boxes list for each image with info about the boxes corners
        """
        # Create a display string (and color) for every box location, group any boxes
        # that correspond to the same location.
        STANDARD_COLORS = vis_util.STANDARD_COLORS
        output_boxes_classes_scores = [];

        box_to_display_str_map = defaultdict(list)
        box_to_color_map = defaultdict(str)
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                data_obj = {}

                box = tuple(boxes[i].tolist())
                data_obj['box'] = box
                if scores is None:
                    box_to_color_map[box] = 'black'
                else:

                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    data_obj['classname'] = class_name
                    data_obj['score'] = scores[i]
                    display_str = '{}: {}%'.format(
                        class_name,
                        int(100*scores[i]))

                    box_to_display_str_map[box].append(display_str)

                    box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]
                    output_boxes_classes_scores.append(data_obj)


      # Draw all boxes onto image.
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            vis_util.draw_bounding_box_on_image_array(
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates)

        return image, output_boxes_classes_scores

    def detection_pipeline(input_images_path):
        output_data = []
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                for image_path in input_images_path:
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                     # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                  # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                      [detection_boxes, detection_scores, detection_classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})

                  # Visualization of the results of a detection.
                    img, boxesdata = parse_boxes_on_image_array(
                      image_np,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,
                      line_thickness=8)
                    output_data.append({
                        'img_path': image_path,
                        'boxes': boxesdata
                    })
        return output_data

    def calc_img_variance(imagearr):
        gray = cv2.cvtColor(imagearr, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def cropping(box, image, filename):
        ul = box[1] * image.size[0]
        ut = box[0]* image.size[1]
        br = box[3] * image.size[0]
        bb = box[2]* image.size[1]
        cropbox_width = np.abs(br - ul)
        cropbox_height = np.abs(bb - ut)

        #adjust for tall images-- just take the top since most likely to include head:
        if cropbox_width / cropbox_height < 1.5 :
            new_height = 2 * cropbox_width / 3
            bb = ut + new_height
        #adjst for wide images-- take the center:
        elif cropbox_width / cropbox_height > 1.5 :
            new_width = 1.5 * cropbox_height
            adjustment = (cropbox_width - new_width) * 0.5
            ul = ul + adjustment
            br = br - adjustment
        cropbox = (ul, ut, br, bb)
        out = image.crop(cropbox)
        out.save(filename)
        return out

    def cropping_pipeline(boxes_detected_data):
        for dataobj in boxes_detected_data:
            imgpath = dataobj['img_path']
            boxesArr = dataobj['boxes']
            image = Image.open(imgpath)
            blurriness = []

            for idx, data in enumerate(boxesArr):
                if(data['classname'] == 'person'):
                    fname = imgpath.replace('.jpg', '').replace(imagefolder, 'output_images') + '_' + str(idx) + '_cropped.jpg'
                    cropped = cropping(data['box'], image, fname)
                    laplacian = calc_img_variance(np.array(cropped))
                    blurriness.append({
                        'img': cropped,
                        'blur': laplacian
                    })
            if len(blurriness) > 0:
                most_clear_img = max(blurriness, key=lambda d: d['blur'])['img']
                fname_most_clear = imgpath.replace('.jpg', '').replace(imagefolder, 'output_images_best') + '_best_cropped.jpg'
                most_clear_img.save(fname_most_clear)

    i = 0
    for image_paths_arr in image_generator():
        if(len(image_paths_arr) ==0):
            break
        box_data = detection_pipeline(image_paths_arr)
        cropping_pipeline(box_data)
        i = i+1
        print('done {} images'.format(i * 5) )

image_processing_pipeline(imagefolder)
