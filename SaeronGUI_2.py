# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Damin\Documents\pyqt5\test\saeron_gui_modified.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QGraphicsScene)

import object_detection_brakepad_modify as od
from utils import visualization_utils as vis_utils
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1907, 1037)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.SystemStat = QtWidgets.QLabel(self.centralwidget)
        self.SystemStat.setGeometry(QtCore.QRect(0, 0, 937, 119))
        self.SystemStat.setObjectName("SystemStat")
        self.Time = QtWidgets.QLabel(self.centralwidget)
        self.Time.setGeometry(QtCore.QRect(930, 0, 940, 80))
        self.Time.setObjectName("Time")
        self.Volume_1_CheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.Volume_1_CheckBox.setGeometry(QtCore.QRect(930, 80, 118, 39))
        self.Volume_1_CheckBox.setChecked(False)
        self.Volume_1_CheckBox.setObjectName("Volume_1_CheckBox")
        self.Volume_1_PushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.Volume_1_PushButton_2.setGeometry(QtCore.QRect(1165, 80, 67, 39))
        self.Volume_1_PushButton_2.setObjectName("Volume_1_PushButton_2")
        self.Volume_1_ComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.Volume_1_ComboBox.setGeometry(QtCore.QRect(1232, 80, 118, 39))
        self.Volume_1_ComboBox.setObjectName("Volume_1_ComboBox")
        self.Volume_1_PushButton = QtWidgets.QPushButton(self.centralwidget)
        self.Volume_1_PushButton.setGeometry(QtCore.QRect(1350, 80, 50, 39))
        self.Volume_1_PushButton.setObjectName("Volume_1_PushButton")
        self.Volume_1_Label = QtWidgets.QLabel(self.centralwidget)
        self.Volume_1_Label.setGeometry(QtCore.QRect(1048, 80, 117, 39))
        self.Volume_1_Label.setObjectName("Volume_1_Label")
        self.Volume_2_Label = QtWidgets.QLabel(self.centralwidget)
        self.Volume_2_Label.setGeometry(QtCore.QRect(1518, 80, 117, 39))
        self.Volume_2_Label.setObjectName("Volume_2_Label")
        self.Volume_2_PushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.Volume_2_PushButton_2.setGeometry(QtCore.QRect(1635, 80, 67, 39))
        self.Volume_2_PushButton_2.setObjectName("Volume_2_PushButton_2")
        self.Volume_2_ComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.Volume_2_ComboBox.setGeometry(QtCore.QRect(1702, 80, 118, 39))
        self.Volume_2_ComboBox.setObjectName("Volume_2_ComboBox")
        self.Volume_2_CheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.Volume_2_CheckBox.setGeometry(QtCore.QRect(1400, 80, 118, 39))
        self.Volume_2_CheckBox.setObjectName("Volume_2_CheckBox")
        self.Volume_2_PushButton = QtWidgets.QPushButton(self.centralwidget)
        self.Volume_2_PushButton.setGeometry(QtCore.QRect(1820, 80, 50, 39))
        self.Volume_2_PushButton.setObjectName("Volume_2_PushButton")
        self.info1 = QtWidgets.QLabel(self.centralwidget)
        self.info1.setGeometry(QtCore.QRect(10, 528, 622, 401))
        self.info1.setFrameShape(QtWidgets.QFrame.Box)
        self.info1.setObjectName("info1")
        self.info2 = QtWidgets.QLabel(self.centralwidget)
        self.info2.setGeometry(QtCore.QRect(638, 528, 623, 401))
        self.info2.setFrameShape(QtWidgets.QFrame.Box)
        self.info2.setObjectName("info2")
        self.info3 = QtWidgets.QLabel(self.centralwidget)
        self.info3.setGeometry(QtCore.QRect(1270, 530, 622, 401))
        self.info3.setFrameShape(QtWidgets.QFrame.Box)
        self.info3.setObjectName("info3")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 120, 1881, 401))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cam2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cam2.sizePolicy().hasHeightForWidth())
        self.cam2.setSizePolicy(sizePolicy)
        self.cam2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.cam2.setWordWrap(False)
        self.cam2.setObjectName("cam2")
        self.horizontalLayout.addWidget(self.cam2)
        self.cam1 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.cam1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.cam1.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.cam1.setObjectName("cam1")
        self.horizontalLayout.addWidget(self.cam1)
        self.cam3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.cam3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.cam3.setScaledContents(False)
        self.cam3.setObjectName("cam3")
        self.horizontalLayout.addWidget(self.cam3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1907, 21))
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
        self.info2.setText(_translate("MainWindow", "TextLabel"))
        self.info3.setText(_translate("MainWindow", "TextLabel"))
        self.cam2.setText(_translate("MainWindow", "Image"))
        self.cam1.setText(_translate("MainWindow", "Image"))
        self.cam3.setText(_translate("MainWindow", "Imgae"))
        self.menumenu.setTitle(_translate("MainWindow", "menu"))
        self.actionclose.setText(_translate("MainWindow", "close"))
        self.actionTextimage.setText(_translate("MainWindow", "Textimage"))

        self.objectdetect(self)

    def objectdetect(self, MainWindow):
        objectdetectTF = od.objectdetect(2, 3)
        objectdetectTF.setobjectdetection()
        height, width, channel = objectdetectTF.cvimage.shape
        bytesPerLine = 3 * width
        qImg1 = QImage(objectdetectTF.cvimage.data, width, height, bytesPerLine, QImage.Format_RGB888)

        self.cam1.setPixmap(QtGui.QPixmap(qImg1))
        self.cam1.show()

        #_translate = QtCore.QCoreApplication.translate
        #self.info1.setText(_translate("MainWindow", vis_utils.classscore()))
        #self.info1.show()



        objectdetectTF = od.objectdetect(3,4)
        objectdetectTF.setobjectdetection()
        height, width, channel = objectdetectTF.cvimage.shape
        bytesPerLine = 3 * width
        qImg2 = QImage(objectdetectTF.cvimage.data, width, height, bytesPerLine, QImage.Format_RGB888)

        self.cam2.setPixmap(QtGui.QPixmap(qImg2))
        self.cam2.show()

        objectdetectTF = od.objectdetect(7,8)
        objectdetectTF.setobjectdetection()
        height, width, channel = objectdetectTF.cvimage.shape
        bytesPerLine = 3 * width
        qImg3 = QImage(objectdetectTF.cvimage.data, width, height, bytesPerLine, QImage.Format_RGB888)

        self.cam3.setPixmap(QtGui.QPixmap(qImg3))
        self.cam3.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

