# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Damin\Documents\pyqt5\test\saeronfullword_1image.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QGuiApplication
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QGraphicsScene, QPushButton)
from PyQt5.QtGui import QGuiApplication

import sys
#import object_detection_brakepadtest2_new as odtest2
from utils import visualization_utils as vis_utils
import numpy as np
import cv2, time
import threading
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visual_test_ver1_prototype as vis_util


class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        self.i = 1
        self.t = 1
        self.image_number = 0

    #def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1149, 990)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.SystemStat = QtWidgets.QLabel(self.centralwidget)
        self.SystemStat.setGeometry(QtCore.QRect(0, 0, 341, 111))
        self.SystemStat.setObjectName("SystemStat")
        self.Time = QtWidgets.QLabel(self.centralwidget)
        self.Time.setGeometry(QtCore.QRect(340, 0, 791, 80))
        self.Time.setObjectName("Time")
        self.Volume_1_CheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.Volume_1_CheckBox.setGeometry(QtCore.QRect(340, 80, 91, 39))
        self.Volume_1_CheckBox.setChecked(False)
        self.Volume_1_CheckBox.setObjectName("Volume_1_CheckBox")
        self.Volume_1_PushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.Volume_1_PushButton_2.setGeometry(QtCore.QRect(500, 80, 67, 39))
        self.Volume_1_PushButton_2.setObjectName("Volume_1_PushButton_2")
        self.Volume_1_ComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.Volume_1_ComboBox.setGeometry(QtCore.QRect(570, 80, 118, 39))
        self.Volume_1_ComboBox.setObjectName("Volume_1_ComboBox")
        self.Volume_1_PushButton = QtWidgets.QPushButton(self.centralwidget)
        self.Volume_1_PushButton.setGeometry(QtCore.QRect(690, 80, 50, 39))
        self.Volume_1_PushButton.setObjectName("Volume_1_PushButton")
        self.Volume_1_Label = QtWidgets.QLabel(self.centralwidget)
        self.Volume_1_Label.setGeometry(QtCore.QRect(440, 80, 61, 39))
        self.Volume_1_Label.setObjectName("Volume_1_Label")
        self.Volume_2_Label = QtWidgets.QLabel(self.centralwidget)
        self.Volume_2_Label.setGeometry(QtCore.QRect(840, 80, 61, 39))
        self.Volume_2_Label.setObjectName("Volume_2_Label")
        self.Volume_2_PushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.Volume_2_PushButton_2.setGeometry(QtCore.QRect(900, 80, 67, 39))
        self.Volume_2_PushButton_2.setObjectName("Volume_2_PushButton_2")
        self.Volume_2_ComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.Volume_2_ComboBox.setGeometry(QtCore.QRect(970, 80, 118, 39))
        self.Volume_2_ComboBox.setObjectName("Volume_2_ComboBox")
        self.Volume_2_CheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.Volume_2_CheckBox.setGeometry(QtCore.QRect(740, 80, 101, 39))
        self.Volume_2_CheckBox.setObjectName("Volume_2_CheckBox")
        self.Volume_2_PushButton = QtWidgets.QPushButton(self.centralwidget)
        self.Volume_2_PushButton.setGeometry(QtCore.QRect(1090, 80, 50, 39))
        self.Volume_2_PushButton.setObjectName("Volume_2_PushButton")
        self.info1 = QtWidgets.QLabel(self.centralwidget)
        self.info1.setGeometry(QtCore.QRect(10, 528, 1131, 401))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.info1.sizePolicy().hasHeightForWidth())
        self.info1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.info1.setFont(font)
        self.info1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.info1.setFrameShape(QtWidgets.QFrame.Box)
        self.info1.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.info1.setObjectName("info1")
        self.cam1 = QtWidgets.QLabel(self.centralwidget)
        self.cam1.setGeometry(QtCore.QRect(10,160,560,341))
        self.cam1.setFrameShape(QtWidgets.QFrame.Box)
        self.cam1.setScaledContents(False)
        self.cam1.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.cam1.setObjectName("cam1")
        MainWindow.setCentralWidget(self.centralwidget)
        self.cam2 = QtWidgets.QLabel(self.centralwidget)
        self.cam2.setGeometry(QtCore.QRect(580,160,560,341))
        self.cam2.setFrameShape(QtWidgets.QFrame.Box)
        self.cam2.setScaledContents(False)
        self.cam2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.cam2.setObjectName("cam1")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1149, 21))
        self.menubar.setObjectName("menubar")
        self.menumenu = QtWidgets.QMenu(self.menubar)
        self.menumenu.setObjectName("menumenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionclose = QtWidgets.QAction(MainWindow)
        self.actionclose.setObjectName("actionclose")
        self.actionTextimage = QtWidgets.QAction(MainWindow)
        self.actionTextimage.setObjectName("actionTextimage")
        self.menumenu.addAction(self.actionclose)
        self.menumenu.addAction(self.actionTextimage)
        self.menubar.addAction(self.menumenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.SystemStat.setText(_translate("MainWindow", "TextLabel"))
        self.Time.setText(_translate("MainWindow", "TextLabel"))
        self.Volume_1_CheckBox.setText(_translate("MainWindow", "Internal HDD"))
        self.Volume_1_PushButton_2.setText(_translate("MainWindow", "SaveHere"))
        self.Volume_1_PushButton.setText(_translate("MainWindow", "Refresh"))
        self.Volume_1_Label.setText(_translate("MainWindow", "Volume"))
        self.Volume_2_Label.setText(_translate("MainWindow", "Volume"))
        self.Volume_2_PushButton_2.setText(_translate("MainWindow", "SaveHere"))
        self.Volume_2_CheckBox.setText(_translate("MainWindow", "External HDD"))
        self.Volume_2_PushButton.setText(_translate("MainWindow", "Refresh"))
        self.info1.setText(_translate("MainWindow", "TextLabel"))
        self.cam1.setText(_translate("MainWindow", "Image_original"))
        self.cam2.setText(_translate("MainWindow", "Image_original"))
        self.menumenu.setTitle(_translate("MainWindow", "menu"))
        self.actionclose.setText(_translate("MainWindow", "close"))
        self.actionTextimage.setText(_translate("MainWindow", "Textimage"))

    def nextobject(self):
        self.image_number += 1
        self.original_image = cv2.imread(
            'C:\\Users\\Damin\\Desktop\\Brakepad_test1\\images\\brakepad_{0}.png'.format(self.image_number))
        #self.nextimage_C = self.original_image[235:235 + 91, 235:235 + 271]
        #self.nextimage_L = self.original_image[295:295+ 85, 97: 97 + 82]
        self.nextimage_C = cv2.imread("C:\\Users\\Damin\\Desktop\\Brakepad_test1\\images\\{0}.png".format(self.image_number))
        self.nextimage_L = cv2.imread("C:\\Users\\Damin\\Desktop\\Brakepad_test2\\images\\{0}.png".format(self.image_number))
        print("object_{0} 실행".format(self.image_number))

    def btn(self, MainWindow):
        self.btn = QPushButton("Next object", MainWindow)
        self.btn.move(40, 200)
        self.btn.clicked.connect(self.nextobject)

    def btn2(self, MainWindow):
        #objectdetectTF2 = odtest2.objectdetect(1,2)
        self.btn2 = QPushButton("Detect", MainWindow)
        self.btn2.move(40, 230)
        #self.btn2.clicked.connect(objectdetectTF2.setobjectdetection)
        self.btn2.clicked.connect(self.objectdetect)

    def objectdetect(self):
        with open('word_Center.txt', 'r', encoding='utf8') as f:
            self.word_Center = [str(num) for num in f.read().split()]
        with open('word_Left.txt', 'r', encoding='utf8') as f:
            self.word_Left = [str(num) for num in f.read().split()]

        # Multithreading
        p1 = threading.Thread(target = self.DetectLeft)
        p2 = threading.Thread(target = self.DetectCenter)
        p1.start()
        time.sleep(0.3)
        p2.start()

        self.i += 1

    def DetectLeft(self):
        x_offset2 = 97
        y_offset2 = 295
        print("------------------------\n"+"{0}번째 Left 검사 시작".format(self.image_number)+"\n------------------------")
        # Model preparation
        # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
        # What model to download.
        MODEL_NAME = 'output_inference_graph.pb'
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'brakepadtest2_label_map.pbtxt')
        NUM_CLASSES = 8
        self.t = 0

        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.
        # Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        #self.TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.png'.format(i)) for i in
                                 #range(int(self.i), int(self.i + 1))]

        # Size, in inches, of the output images.
        self.IMAGE_SIZE = (12, 8)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

        with detection_graph.as_default():
            # with tf.Session(graph=detection_graph) as sess:
            with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                beforeimage = np.zeros((75,64,3))

                while True:
                    if np.any(self.nextimage_L != beforeimage):
                        self.cvimage_L = cv2.cvtColor(self.nextimage_L, cv2.COLOR_RGB2BGR)

                        # the array based representation of the image will be used later in order to prepare the
                        # result image with boxes and labels on it.
                        height, width, channels = self.cvimage_L.shape

                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_ex1 = np.reshape(self.cvimage_L, (1, height, width, 3))

                        # Actual detection.
                        (boxes, self.scores, self.classes, num) = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_np_ex1})

                        # Visualization of the results of a detection.
                        self.Result_Left = vis_util.visualutil.visualize_boxes_and_labels_on_image_array(
                            self.cvimage_L,
                            np.squeeze(boxes),
                            np.squeeze(self.classes).astype(np.int32),
                            np.squeeze(self.scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=2)

                        beforeimage = self.nextimage_L
                        lasttest = time.time()
                    elif np.all(self.nextimage_L == beforeimage):
                        pass
                        if (time.time() - lasttest) > 10:
                            break
                print("------------------------\n"+"{0}번째 Left 검사 끝".format(self.image_number)+"\n------------------------")

    def DetectCenter(self):
        x_offset1 = 235
        y_offset1 = 235
        x_offset2 = 97
        y_offset2 = 295
        print("------------------------\n"+"{0}번째 Center 검사 시작".format(self.image_number)+"\n------------------------")
        # Model preparation
        # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
        # What model to download.
        MODEL_NAME = 'C:\\Users\\Damin\\Desktop\\Brakepad_test1\\object_detection\\output_inference_graph.pb'
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = 'C:\\Users\\Damin\\Desktop\\Brakepad_test1\\object_detection\\data\\brakepadtest1_label_map.pbtxt'
        NUM_CLASSES = 14
        self.t = 0

        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.
        # Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        #PATH_TO_TEST_IMAGES_DIR = "C:\\Users\\Damin\\Desktop\\Brakepad_test1\\images"
        #self.TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.png'.format(i)) for i in
                                #range(int(self.i), int(self.i + 1))]

        # Size, in inches, of the output images.
        self.IMAGE_SIZE = (12, 8)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        with detection_graph.as_default():
            # with tf.Session(graph=detection_graph) as sess:
            with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                beforeimage = np.zeros((75,64,3))

                while True:
                    if np.any(self.nextimage_C != beforeimage):
                        self.cvimage_C = cv2.cvtColor(self.nextimage_C, cv2.COLOR_RGB2BGR)

                        # the array based representation of the image will be used later in order to prepare the
                        # result image with boxes and labels on it.
                        height, width, channels = self.cvimage_C.shape

                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_ex1 = np.reshape(self.cvimage_C, (1, height, width, 3))

                        # Actual detection.
                        (boxes, self.scores, self.classes, num) = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_np_ex1})

                        # Visualization of the results of a detection.
                        self.Result_Center = vis_util.visualutil.visualize_boxes_and_labels_on_image_array(
                            self.cvimage_C,
                            np.squeeze(boxes),
                            np.squeeze(self.classes).astype(np.int32),
                            np.squeeze(self.scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=2)
                        self.original_image[y_offset2:y_offset2 + self.cvimage_L.shape[0],
                                            x_offset2:x_offset2 + self.cvimage_L.shape[1]] = self.cvimage_L

                        self.original_image[y_offset1:y_offset1 + self.cvimage_C.shape[0],
                                            x_offset1:x_offset1 + self.cvimage_C.shape[1]] = self.cvimage_C

                        qimage = QImage(self.original_image, self.original_image.shape[1],
                                        self.original_image.shape[0], self.original_image.shape[1] * 3,
                                        QImage.Format_RGB888)

                        self.cam1.setPixmap(QtGui.QPixmap(qimage))
                        self.cam1.show()
                        self.info1.setText("Center word:" + self.Result_Center + "\nLeft word:" + self.Result_Left)
                        now = time.localtime()

                        if (self.Result_Left in self.word_Left) and (self.Result_Center in self.word_Center):
                           title = "%04d.%02d.%02d-%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday,
                                                                       now.tm_hour, now.tm_min,
                                                                       now.tm_sec) + "-OK"
                        else:
                           title = "%04d.%02d.%02d-%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday,
                                                                       now.tm_hour, now.tm_min,
                                                                       now.tm_sec) + "-NG"

                        cv2.imwrite('C:\\Users\\Damin\\Pictures\\objectdetection_brakepadtest\\%s.png' % title,
                                    self.original_image)
                        QApplication.processEvents()

                        beforeimage = self.nextimage_C
                        lasttest = time.time()
                    elif np.all(self.nextimage_C == beforeimage):
                        QApplication.processEvents()
                        pass
                        if (time.time() - lasttest) > 10:
                            break
                print("------------------------\n"+"{0}번째 Center 검사 끝".format(self.image_number) + "\n------------------------")

    def myShow(self, qimage):
        original_image = cv2.imread(
            'C:\\Users\\Damin\\Desktop\\Brakepad_test1\\images\\brakepad_1.png')
        qimage = QtGui.QImage(original_image, original_image.shape[1],
                              original_image.shape[0], original_image.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.cam1.setPixmap(QtGui.QPixmap(qimage))
        self.cam1.show()
        self.info1.setText("Center word:")
        QApplication.processEvents()
        #return self.cam1.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    ui.btn(MainWindow)
    ui.btn2(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())