# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
from Ui_hw1 import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtGui
import numpy as np
from scipy import signal,ndimage
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random


class MainWindow(QMainWindow, Ui_MainWindow):
    src1 = None
    src2 = None
    mousePosition = []
    gaussian=None
    Ix=None
    Iy=None
    G=None
    mnist_data_path = 'MNIST_data/'
    learning_rate = 0.001
    batch_size = 100
    num_epochs=1
    model_path = './model/model.ckpt' 
    def __init__(self, parent=None):

        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn4_3.clicked.connect(self.on_btn4_3_click)
        self.btn4_4.clicked.connect(self.on_btn4_4_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)
        self.btn5_3.clicked.connect(self.on_btn5_3_click)
        self.btn5_4.clicked.connect(self.on_btn5_4_click)
        self.btn5_5.clicked.connect(self.on_btn5_5_click)

    # button for problem 1.1
    def on_btn1_1_click(self):

        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
        img = cv2.imread(fileName[0])
        print("Height: " + str(img.shape[0]))
        print("Width: " + str(img.shape[1]))
        # 顯示圖片
        cv2.imshow('My Image', img)
        self.wait_key()

    def on_btn1_2_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
        img = cv2.imread(fileName[0])
        cv2.imshow('BGR', img)

        b, g, r = cv2.split(img)  # 拆分通道
        im_rbg = cv2.merge([g, r, b])  # 合併通道
        # 顯示圖片
        cv2.imshow('RBG', im_rbg)
        self.wait_key()

    def on_btn1_3_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
        img = cv2.imread(fileName[0])
        cv2.imshow('Original', img)
        h_flip = cv2.flip(img, 1)
        cv2.imshow('Flip', h_flip)
        self.wait_key()

    def on_btn1_4_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
        self.src1 = cv2.imread(fileName[0])
        self.src2 = cv2.flip(self.src1, 1)
        cv2.namedWindow("BLENDING")
        cv2.createTrackbar("BLEND", "BLENDING", 0, 100, self.on_trackbar)
        self.on_trackbar(0)
        self.wait_key()

    def on_trackbar(self, val):
        alpha = val / 100
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(self.src1, alpha, self.src2, beta, 0.0)
        cv2.imshow("BLENDING", dst)

    def on_btn2_1_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
        img = cv2.imread(fileName[0], cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Original Image', img)
        ret, global_thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow('Threshold Image', global_thresh)
        self.wait_key()

    def on_btn2_2_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
        img = cv2.imread(fileName[0], cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Original Image', img)
        local_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -1)
        cv2.imshow('Adaptive Threshold Image', local_thresh)
        self.wait_key()

    def on_btn3_1_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
        img = cv2.imread(fileName[0])
        h, w = img.shape[:2]
        cv2.imshow('Original Image', img)
        x = int(self.edtTx.text())
        y = int(self.edtTy.text())
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(img, M, (w, h))
        M = cv2.getRotationMatrix2D((130 + x, 125 + y), int(self.edtAngle.text()), float(self.edtScale.text()))
        img_warp = cv2.warpAffine(shifted, M, (w, h))
        cv2.imshow('Rotation + Scale + Translation Image', img_warp)
        self.wait_key()

    def on_btn3_2_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
        self.src1 = cv2.imread(fileName[0])
        cv2.namedWindow('Original')
        cv2.imshow('Original', self.src1)
        cv2.setMouseCallback('Original', self.mouse_click)
        self.wait_key()

    def mouse_click(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.mousePosition) < 4:
            self.mousePosition.append([x, y])
            if (len(self.mousePosition) == 4):
                M = cv2.getPerspectiveTransform(np.float32(self.mousePosition),
                                                np.float32([[20, 20], [450, 20], [450, 450], [20, 450]]))
                dst = cv2.warpPerspective(self.src1, M, (450, 450))
                M = np.float32([[1, 0, -20], [0, 1, -20]])
                shifted = cv2.warpAffine(dst, M, (430, 430))
                cv2.imshow('Perspective Result Image', shifted)
                self.mousePosition = []

    def on_btn4_1_click(self):
        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')
        img = cv2.imread(fileName[0])
        cv2.imshow('Original Image', img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 灰階
        x, y = np.mgrid[-1:2, -1:2]
        sigma=1
        normal = 1 / (2.0 * np.pi * sigma**2)
        gaussian_kernel = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        self.gaussian = signal.convolve2d(gray, gaussian_kernel, mode='same', boundary='fill', fillvalue=0)
        # self.gaussian=np.uint8(np.absolute(self.gaussian))
        plt.ion()
        plt.figure(num='Gaussian')
        plt.imshow(self.gaussian, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.show()
        self.wait_key()

    def on_btn4_2_click(self):
        Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
        self.Ix = ndimage.filters.convolve(self.gaussian, Kx)
        self.Ix = self.Ix / self.Ix.max() * 255
        plt.ion()
        plt.figure(num='Sobel X')
        plt.imshow(np.uint8(np.absolute(self.Ix)), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.show()
        

    def on_btn4_3_click(self):
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        self.Iy = ndimage.filters.convolve(self.gaussian, Ky)
        self.Iy = self.Iy / self.Iy.max() * 255
        self.Iy=np.uint8(np.absolute(self.Iy))
        plt.ion()
        plt.figure(num='Sobel Y')
        plt.imshow(np.uint8(np.absolute(self.Iy)), cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.show()

    def on_btn4_4_click(self):
        self.G = np.hypot( self.Ix,self.Iy)
        self.G = self.G /self.G.max() * 255
        plt.ion()
        plt.figure(num='Magnitude')
        plt.imshow(self.G, cmap=plt.get_cmap('gray'))
        plt.axis('off')
        plt.show()
    def on_btn5_1_click(self):
        mnist = input_data.read_data_sets(self.mnist_data_path, one_hot=True)  # using one-hot for output
        plt.ion()
        plt.figure(num='Image')
        for num in range (0,10):
            i=random.randint(0,55000)
            image = mnist.train.images[i]
            image = np.array(image, dtype='float')
            pixels = image.reshape((28, 28))
            plt.subplot(2,5,num+1)
            plt.axis('off')
            plt.title("Label:" +str(np.argmax(mnist.train.labels[i, :] )))
            plt.imshow(pixels, cmap='gray')
        plt.show()
    
    def on_btn5_2_click(self):
        print("hyperparameters:")
        print("batch size:",self.batch_size)
        print("learning rate:",self.learning_rate)
        print("optimizer: "+"Adam")
    def on_btn5_3_click(self):
        self.train()
    def on_btn5_4_click(self):
        img = cv2.imread("images/50epoch.png")
        # 顯示圖片
        cv2.imshow('50 epochs', img)
        self.wait_key()
    def on_btn5_5_click(self):
          
        saver = tf.train.import_meta_graph("./model/model.ckpt.meta") 
        with tf.Session() as sess:  
            
            #----- load previous model-----#
            saver.restore(sess, self.model_path)
            
            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")
            Y = graph.get_tensor_by_name("Y:0")
            Y_conv = graph.get_tensor_by_name("Softmax:0")
            
            mnist = input_data.read_data_sets(self.mnist_data_path, one_hot=True)  # using one-hot for output
            pred=sess.run(Y_conv[int(self.index.text()),:], feed_dict = {X: mnist.test.images, Y: mnist.test.labels})
            image = mnist.test.images[int(self.index.text())]
            image = np.array(image, dtype='float')
            image = image.reshape((28, 28))
            plt.ion()
            plt.figure(num='Test Image Index'+ self.index.text())
            plt.subplot(2,1,1)
            plt.axis('off')
            plt.imshow(image, cmap='gray')
            plt.subplot(2,1,2)
            plt.bar(['0','1','2','3','4','5','6','7','8','9'], pred)
            plt.show()
            
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)
    def lenet_5_forward_propagation(self,X):
        """
        @note: construction of leNet-5 forward computation graph:
            CONV1 -> MAXPOOL1 -> CONV2 -> MAXPOOL2 -> FC3 -> FC4 -> SOFTMAX
            
        @param X: input dataset placeholder, of shape (number of examples (m), input size)
        
        @return: A_l, the output of the softmax layer, of shape (number of examples, output size)
        """
        
        # reshape imput as [number of examples (m), weight, height, channel]
        X_ = tf.reshape(X, [-1, 28, 28, 1])  # num_channel = 1 (gray image)
        
        ### CONV1 (f = 5*5*1, n_f = 6, s = 1, p = 'same')
        W_conv1 = self.weight_variable(shape = [5, 5, 1, 6])
        b_conv1 = self.bias_variable(shape = [6])
        # shape of A_conv1 ~ [m,28,28,6]
        A_conv1 = tf.nn.relu(tf.nn.conv2d(X_, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1)
        
        ### MAXPOOL1 (f = 2*2*1, s = 2, p = 'same')
        # shape of A_pool1 ~ [m,14,14,6]
        A_pool1 = tf.nn.max_pool(A_conv1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
        
        ### CONV2 (f = 5*5*1, n_f = 16, s = 1, p = 'same')
        W_conv2 = self.weight_variable(shape = [5, 5, 6, 16])
        b_conv2 = self.bias_variable(shape = [16])    
        # shape of A_conv2 ~ [m,10,10,16]
        A_conv2 = tf.nn.relu(tf.nn.conv2d(A_pool1, W_conv2, strides = [1, 1, 1, 1], padding = 'VALID') + b_conv2)    
        
        ### MAXPOOL2 (f = 2*2*1, s = 2, p = 'same')  
        # shape of A_pool2~ [m,5,5,16]
        A_pool2 = tf.nn.max_pool(A_conv2, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

        ### FC3 (n = 120)
        # flatten the volumn to vector
        A_pool2_flat = tf.reshape(A_pool2, [-1, 5*5*16])
        
        W_fc3 = self.weight_variable([5*5*16, 120])
        b_fc3 = self.bias_variable([120])
        # shape of A_fc3 ~ [m,120]
        A_fc3 = tf.nn.relu(tf.matmul(A_pool2_flat, W_fc3) + b_fc3)
            
        ### FC4 (n = 84)
        W_fc4 = self.weight_variable([120, 84])
        b_fc4 = self.bias_variable([84])
        # shape of A_fc4 ~ [m, 84]
        A_fc4 = tf.nn.relu(tf.matmul(A_fc3, W_fc4) + b_fc4)

        # Softmax (n = 10)
        W_l = self.weight_variable([84, 10])
        b_l = self.bias_variable([10])
        # shape of A_l ~ [m,10]
        A_l=tf.nn.softmax(tf.matmul(A_fc4, W_l) + b_l)

        return A_l
    def train(self):
        mnist = input_data.read_data_sets(self.mnist_data_path, one_hot=True)  # using one-hot for output
        X_train, Y_train = mnist.train.images, mnist.train.labels
        X_valid, Y_valid = mnist.validation.images, mnist.validation.labels
        X_test,  Y_test  = mnist.test.images, mnist.test.labels

        #--- get the shape of data ---#
        m_train, n_x = X_train.shape
        _, n_y       = Y_train.shape
        m_valid, _   = X_valid.shape
        m_test, _    = X_test.shape

        #--- build the model ---#
        X = tf.placeholder(tf.float32, [None, n_x], name="X")
        Y = tf.placeholder(tf.float32, [None, n_y], name="Y")

        Y_conv = self.lenet_5_forward_propagation(X)

        # cost function
        cost = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(Y_conv, 1e-8,1.0)))

        # accuracy
        correct_prediction = tf.equal(tf.argmax(Y_conv,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # optimizer (using Adam)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        # initial the graph and session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()  
        sess.run(init)
        loss=[]
        iteration=[]

        # iterations
        for i in range( int(self.num_epochs*m_train/self.batch_size) ):
            # print(i)
            batch_xs, batch_ys = mnist.train.next_batch(self.batch_size, shuffle = True)  # using mini-batch
            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
            loss.append(minibatch_cost*100)
            iteration.append(i)
        plt.ion()
        plt.figure(num='1 epoch')
        plt.title("epoch [0/50]")

        plt.xlabel("iteration")
        plt.ylabel("loss %")
        plt.plot(iteration,loss)
        plt.show()
        sess.close()
    def wait_key(self):
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

