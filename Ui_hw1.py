# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\ostrich\HW1\hw1.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1125, 539)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(290, 10, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(540, 10, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(290, 200, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(30, 50, 151, 231))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setObjectName("frame")
        self.btn1_2 = QtWidgets.QPushButton(self.frame)
        self.btn1_2.setGeometry(QtCore.QRect(10, 70, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_2.setFont(font)
        self.btn1_2.setObjectName("btn1_2")
        self.btn1_1 = QtWidgets.QPushButton(self.frame)
        self.btn1_1.setGeometry(QtCore.QRect(10, 10, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_1.setFont(font)
        self.btn1_1.setObjectName("btn1_1")
        self.btn1_4 = QtWidgets.QPushButton(self.frame)
        self.btn1_4.setGeometry(QtCore.QRect(10, 180, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_4.setFont(font)
        self.btn1_4.setObjectName("btn1_4")
        self.btn1_3 = QtWidgets.QPushButton(self.frame)
        self.btn1_3.setGeometry(QtCore.QRect(10, 130, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_3.setFont(font)
        self.btn1_3.setObjectName("btn1_3")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(520, 50, 261, 321))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_3.setObjectName("frame_3")
        self.frame_5 = QtWidgets.QFrame(self.frame_3)
        self.frame_5.setGeometry(QtCore.QRect(10, 30, 241, 231))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_5.setObjectName("frame_5")
        self.btn3_1 = QtWidgets.QPushButton(self.frame_5)
        self.btn3_1.setGeometry(QtCore.QRect(20, 180, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn3_1.setFont(font)
        self.btn3_1.setObjectName("btn3_1")
        self.label_9 = QtWidgets.QLabel(self.frame_5)
        self.label_9.setGeometry(QtCore.QRect(10, 0, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setGeometry(QtCore.QRect(10, 30, 221, 141))
        self.frame_6.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_6.setObjectName("frame_6")
        self.label_10 = QtWidgets.QLabel(self.frame_6)
        self.label_10.setGeometry(QtCore.QRect(10, 10, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.frame_6)
        self.label_11.setGeometry(QtCore.QRect(10, 40, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.frame_6)
        self.label_12.setGeometry(QtCore.QRect(10, 70, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.frame_6)
        self.label_13.setGeometry(QtCore.QRect(10, 100, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.edtAngle = QtWidgets.QLineEdit(self.frame_6)
        self.edtAngle.setGeometry(QtCore.QRect(60, 10, 113, 20))
        self.edtAngle.setObjectName("edtAngle")
        self.edtScale = QtWidgets.QLineEdit(self.frame_6)
        self.edtScale.setGeometry(QtCore.QRect(60, 40, 113, 20))
        self.edtScale.setObjectName("edtScale")
        self.edtTx = QtWidgets.QLineEdit(self.frame_6)
        self.edtTx.setGeometry(QtCore.QRect(60, 70, 113, 20))
        self.edtTx.setObjectName("edtTx")
        self.edtTy = QtWidgets.QLineEdit(self.frame_6)
        self.edtTy.setGeometry(QtCore.QRect(60, 100, 113, 20))
        self.edtTy.setObjectName("edtTy")
        self.label_14 = QtWidgets.QLabel(self.frame_6)
        self.label_14.setGeometry(QtCore.QRect(180, 10, 21, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.frame_6)
        self.label_15.setGeometry(QtCore.QRect(180, 70, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.frame_6)
        self.label_16.setGeometry(QtCore.QRect(180, 100, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.btn3_2 = QtWidgets.QPushButton(self.frame_3)
        self.btn3_2.setGeometry(QtCore.QRect(30, 270, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn3_2.setFont(font)
        self.btn3_2.setObjectName("btn3_2")
        self.label_8 = QtWidgets.QLabel(self.frame_3)
        self.label_8.setGeometry(QtCore.QRect(10, 10, 211, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(290, 40, 151, 121))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setObjectName("frame_2")
        self.btn2_2 = QtWidgets.QPushButton(self.frame_2)
        self.btn2_2.setGeometry(QtCore.QRect(10, 70, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn2_2.setFont(font)
        self.btn2_2.setObjectName("btn2_2")
        self.btn2_1 = QtWidgets.QPushButton(self.frame_2)
        self.btn2_1.setGeometry(QtCore.QRect(10, 10, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn2_1.setFont(font)
        self.btn2_1.setObjectName("btn2_1")
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(290, 260, 151, 231))
        self.frame_4.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_4.setObjectName("frame_4")
        self.btn4_2 = QtWidgets.QPushButton(self.frame_4)
        self.btn4_2.setGeometry(QtCore.QRect(10, 70, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_2.setFont(font)
        self.btn4_2.setObjectName("btn4_2")
        self.btn4_1 = QtWidgets.QPushButton(self.frame_4)
        self.btn4_1.setGeometry(QtCore.QRect(10, 10, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_1.setFont(font)
        self.btn4_1.setObjectName("btn4_1")
        self.btn4_4 = QtWidgets.QPushButton(self.frame_4)
        self.btn4_4.setGeometry(QtCore.QRect(10, 180, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_4.setFont(font)
        self.btn4_4.setObjectName("btn4_4")
        self.btn4_3 = QtWidgets.QPushButton(self.frame_4)
        self.btn4_3.setGeometry(QtCore.QRect(10, 130, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_3.setFont(font)
        self.btn4_3.setObjectName("btn4_3")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(900, 0, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.frame_7 = QtWidgets.QFrame(self.centralwidget)
        self.frame_7.setGeometry(QtCore.QRect(870, 40, 201, 391))
        self.frame_7.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_7.setObjectName("frame_7")
        self.btn5_2 = QtWidgets.QPushButton(self.frame_7)
        self.btn5_2.setGeometry(QtCore.QRect(10, 70, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_2.setFont(font)
        self.btn5_2.setObjectName("btn5_2")
        self.btn5_1 = QtWidgets.QPushButton(self.frame_7)
        self.btn5_1.setGeometry(QtCore.QRect(10, 10, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_1.setFont(font)
        self.btn5_1.setObjectName("btn5_1")
        self.btn5_4 = QtWidgets.QPushButton(self.frame_7)
        self.btn5_4.setGeometry(QtCore.QRect(10, 200, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_4.setFont(font)
        self.btn5_4.setObjectName("btn5_4")
        self.btn5_3 = QtWidgets.QPushButton(self.frame_7)
        self.btn5_3.setGeometry(QtCore.QRect(10, 130, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_3.setFont(font)
        self.btn5_3.setObjectName("btn5_3")
        self.btn5_5 = QtWidgets.QPushButton(self.frame_7)
        self.btn5_5.setGeometry(QtCore.QRect(10, 300, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_5.setFont(font)
        self.btn5_5.setObjectName("btn5_5")
        self.label_6 = QtWidgets.QLabel(self.frame_7)
        self.label_6.setGeometry(QtCore.QRect(10, 250, 114, 33))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.index = QtWidgets.QLineEdit(self.frame_7)
        self.index.setGeometry(QtCore.QRect(120, 258, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.index.setFont(font)
        self.index.setObjectName("index")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1125, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "1. Image Processing"))
        self.label_2.setText(_translate("MainWindow", "2. Adaptive Threshold"))
        self.label_3.setText(_translate("MainWindow", "3. Image Transformation"))
        self.label_4.setText(_translate("MainWindow", "4. Convolution"))
        self.btn1_2.setText(_translate("MainWindow", "1.2 Color Conversion"))
        self.btn1_1.setText(_translate("MainWindow", "1.1 Load Image"))
        self.btn1_4.setText(_translate("MainWindow", "1.4 Blending"))
        self.btn1_3.setText(_translate("MainWindow", "1.3 Image Flipping"))
        self.btn3_1.setText(_translate("MainWindow", "3.1 Rotation, Scaling, Translation"))
        self.label_9.setText(_translate("MainWindow", "Parameters"))
        self.label_10.setText(_translate("MainWindow", "Angle:"))
        self.label_11.setText(_translate("MainWindow", "Scale:"))
        self.label_12.setText(_translate("MainWindow", "Tx:"))
        self.label_13.setText(_translate("MainWindow", "Ty:"))
        self.label_14.setText(_translate("MainWindow", "deg"))
        self.label_15.setText(_translate("MainWindow", "pixel"))
        self.label_16.setText(_translate("MainWindow", "pixel"))
        self.btn3_2.setText(_translate("MainWindow", "3.2 Perspective Transform"))
        self.label_8.setText(_translate("MainWindow", "3.1 Rot, scale, Transformation"))
        self.btn2_2.setText(_translate("MainWindow", "2.2 Local Threshold"))
        self.btn2_1.setText(_translate("MainWindow", "2.1 Global Threshold"))
        self.btn4_2.setText(_translate("MainWindow", "4.2 Sobel X"))
        self.btn4_1.setText(_translate("MainWindow", "4.1 Gaussian"))
        self.btn4_4.setText(_translate("MainWindow", "4.4 Magnitude"))
        self.btn4_3.setText(_translate("MainWindow", "4.3 Sobel Y"))
        self.label_5.setText(_translate("MainWindow", "5. LeNet5"))
        self.btn5_2.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.btn5_1.setText(_translate("MainWindow", "5.1 Show Train Images"))
        self.btn5_4.setText(_translate("MainWindow", "5.4 Show Training Result"))
        self.btn5_3.setText(_translate("MainWindow", "5.3 Train 1 Epoch"))
        self.btn5_5.setText(_translate("MainWindow", "5.5 Inference"))
        self.label_6.setText(_translate("MainWindow", "Test Image Index:"))
