import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

from PIL import Image
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

"""if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')
"""

class objectdetect():
    def __init__(self,num1,num2):
        self.start = num1
        self.end = num2

    def setobjectdetection(self):
# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
        MODEL_NAME = 'output_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('saerondata', 'brakepad_label_map.pbtxt')

        NUM_CLASSES = 4

        self.t = 0

# ## Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)




# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.


        PATH_TO_TEST_IMAGES_DIR = 'brakepad_new'
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(int(self.start), int(self.end)) ]

# Size, in inches, of the output images.
        self.IMAGE_SIZE = (12, 8)

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
              for image_path in TEST_IMAGE_PATHS:
                #image = Image.open(image_path)
                image1 = cv2.imread(image_path,cv2.IMREAD_COLOR)
                self.cvimage = cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)


        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        #image_np = load_image_into_numpy_array(image)
                height, width, channels = self.cvimage.shape


        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #image_np_expanded = np.expand_dims(image_np, axis=0)
                image_np_ex1 = np.reshape(self.cvimage, (1,height,width,3))

        # Actual detection.
                (boxes, self.scores, self.classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_ex1}) #image_np_expanded

        # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    self.cvimage,#image_np
                    np.squeeze(boxes),
                    np.squeeze(self.classes).astype(np.int32),
                    np.squeeze(self.scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=5)

                self.t = self.t + 1

                self.showobject()
                #plt.figure(figsize=IMAGE_SIZE)
                #plt.imshow(cvimage)
                #plt.show()

                #t = t + 1
                #cvimage_new = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
                #cv2.imwrite("C:\\Users\\Damin\\Pictures\\objectdetection_brakepad\\%d.jpg" % t, cvimage_new)

    def showobject(self):
        #plt.figure(figsize=self.IMAGE_SIZE)
        #plt.imshow(self.cvimage)
        #plt.show()
        self.cvimage_new = cv2.cvtColor(self.cvimage, cv2.COLOR_BGR2RGB)
        cv2.imwrite("C:\\Users\\Damin\\Pictures\\objectdetection_brakepad\\%d.jpg" % self.t, self.cvimage_new)


    def cvimage(self):
        return self.cvimage_new

    def score(self):
        return self.scores



if __name__ == "__main__":
    play = objectdetect(1,4)
    play.setobjectdetection()
    play.showobject()
    print("this is module file.")
