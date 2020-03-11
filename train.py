import tensorflow as tf 

# 定义回调函数，如果模型的准确率大于0.99则停止训练
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
# 加载手写字体识别数据集, 其中训练集为60000张28*28的灰度图像，测试集为10000张
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 卷积神经网络调整
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
# 归一化操作，将数值映射到0-1之前的数字，方便运算
train_images = train_images/255.0
test_images = test_images/255.0

# 传统模型
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28,28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')]
# )

# 卷积神经网络模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    #Add another convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #Now flatten the output. After this you'll just have the same DNN structure as the non convolutional version
    tf.keras.layers.Flatten(),
    #The same 128 dense layers, and 10 output layers as in the pre-convolution example:
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

# 模型训练
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])
loss, acc = model.evaluate(test_images, test_labels)
print('测试结果：准确率为{}'.format(acc))

# 保存模型
model.save('models/mnist_cnn.h5')
print('模型保存成功！')

