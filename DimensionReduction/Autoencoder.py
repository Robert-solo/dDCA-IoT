
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense

# 加载 MNIST 数据集
(X_train, _), (X_test, _) = mnist.load_data()

# 数据预处理
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# 构建自编码器模型
input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img)  # 编码层，将输入压缩到32维
decoded = Dense(784, activation='sigmoid')(encoded)  # 解码层，将编码结果解压缩到784维

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 训练自编码器模型
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# 使用训练好的自编码器模型进行数据降维
encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(X_test)

# 可视化降维后的结果
n = 10  # 可视化的数字数量
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(8, 4))  # 将32维的编码结果reshape为8x4的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
