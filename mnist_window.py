import tensorflow as tf 
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import random
import numpy as np


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/number.png'))
        self.setWindowTitle('手写数字识别')
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        test_images = test_images.reshape(10000, 28, 28, 1)
        self.test_images = test_images/255.0
        self.model = tf.keras.models.load_model('models/mnist_cnn.h5')
        self.resize(800, 600)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("测试样本")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        self.to_predict = self.test_images[0]
        img_init = self.to_predict*255
        img_init = cv2.resize(img_init, (400, 400))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap('images/target.png'))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        # left_layout.setAlignment(Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 加载测试样本 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 识别手写字体 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        label_result = QLabel(' 识 别 结 果 ')
        self.result = QLabel("7")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        # right_layout.addSpacing(5)
        right_widget.setLayout(right_layout)

        # 关于页面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用手写数字识别系统')
        about_title.setFont(QFont('楷体', 18))  
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/wxpayx.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()
        label_super.setText("<a href='https://www.jianshu.com/u/9bc6de048aa5'>我的个人主页</a>")
        label_super.setFont(QFont('楷体', 12))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        # git_img = QMovie('images/')
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, '主页面')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    def change_img(self):
        index = random.randint(0, 9999)
        self.to_predict = self.test_images[index]
        img = self.to_predict*255
        img = cv2.resize(img, (400, 400))
        cv2.imwrite('images/target.png', img)
        self.img_label.setPixmap(QPixmap('images/target.png'))

    def predict_img(self):
        one_hot = self.model.predict(self.to_predict.reshape(1, 28, 28, 1))
        result = np.argmax(one_hot)
        self.result.setText(str(result))
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())



