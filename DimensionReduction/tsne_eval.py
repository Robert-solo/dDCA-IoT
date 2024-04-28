# import matplotlib.pyplot as plt
# from time import time
# from sklearn.datasets import load_digits
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import numpy as np
#
# # 加载Digits数据集
# # digits = load_digits()
# # data = digits.data
# # target = digits.target
#
# # 更换成其他类型的数据集
# data= np.loadtxt(r"../data/mnist2500_X.txt")
# target = np.loadtxt(r"../data/mnist2500_labels.txt")
# # Y = TSNE(X, 2, 50, 20.0)
# print(data.shape) # 查看数据维度（64）
#
# # 可视化降维后的数据
# def show_pic(new_data, target, label):
#     plt.title(label)
#     plt.scatter(new_data[:, 0], new_data[:, 1], c=target)
#     plt.show()
# # 使用不同的算法将数据降为2维
# # 1、PCA降维可视化
# pca = PCA(n_components=2).fit(data)
# pca_data = pca.transform(data) # 降维转换
# show_pic(pca_data, target, 'PCA')
#
# # 2、LDA降维可视化
# lda = LinearDiscriminantAnalysis(n_components=2).fit(data, target)
# lda_data = lda.transform(data) # 降维转换
# show_pic(lda_data, target, 'LDA')
#
# # 3、TSNE降维可视化（计算复杂度高，较慢）
# print('  t-SNE降维中，请耐心等待......')
# start = time() # 开始时间
# tsne = TSNE(n_components=2, init='pca', random_state=0)
# tsne_data = tsne.fit_transform(data)
# end = time() # 完成时间
# show_pic(tsne_data, target, 't-SNE (cost {:.1f} seconds)'.format(end-start))
#
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
#
#
# # 评估三种算法降维后（均降成2维）的分类准确性（基于KNN算法）
# def eval_model(new_data, target, label):
#     X_train, X_test, y_train, y_test = train_test_split(new_data, target, random_state=0)  # 拆分数据集
#     model = KNeighborsClassifier(3).fit(X_train, y_train)
#     score = model.score(X_test, y_test)  # 在测试集上评估模型成绩
#     print(label, score)  # 打印模型成绩
#
#
# print('数据集由64维降成2维后，使用K-NN分类准确性对比：')
#
# eval_model(pca_data, target, 'PCA accuracy:')
# eval_model(lda_data, target, 'PDA accuracy:')
# eval_model(tsne_data, target, 't-SNE accuracy:')

## 优化后的代码版本
import matplotlib.pyplot as plt
from time import time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense

def load_data():
    # 更换成其他类型的数据集
    data = np.loadtxt(r"../data/mnist2500_X.txt")
    target = np.loadtxt(r"../data/mnist2500_labels.txt")
    return data, target


def visualize_reduction(data, target, reduction_method, title):
    reduced_data = reduction_method.fit_transform(data, target)
    plt.title(title)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target)
    plt.show()

def evaluate_model(data, target, reduction_method, method_name):
    reduced_data = reduction_method.fit_transform(data, target)
    X_train, X_test, y_train, y_test = train_test_split(reduced_data, target, random_state=0)
    model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f'{method_name} accuracy: {score}')

def train_autoencoder_and_reduce_dimension(data):
    # 构建自编码器模型
    input_img = Input(shape=(784,))
    encoded = Dense(32, activation='relu')(input_img)  # 编码层，将输入压缩到32维
    decoded = Dense(784, activation='sigmoid')(encoded)  # 解码层，将编码结果解压缩到784维

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # 训练自编码器模型
    autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True)

    # 使用训练好的自编码器模型进行数据降维
    encoder = Model(input_img, encoded)
    encoded_imgs = encoder.predict(data)

    return encoded_imgs


def main():
    data, target = load_data()

    # PCA
    start_time1 = time()
    pca = PCA(n_components=2)
    visualize_reduction(data, target, pca, 'PCA')
    print(f'The PCA deduction time cost {time() - start_time1:.1f} seconds)')
    evaluate_model(data, target, pca, 'PCA')


    # LDA
    start_time2 = time()
    lda = LinearDiscriminantAnalysis(n_components=2)
    visualize_reduction(data, target, lda, 'LDA')
    print(f'The LDA deduction time cost {time() - start_time2:.1f} seconds)')
    evaluate_model(data, target, lda, 'LDA')


    # t-SNE
    # print('t-SNE reduction in progress, please wait...')
    start_time3 = time()
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    visualize_reduction(data, target, tsne, 't-SNE')
    print(f'The TSNE deduction time cost {time() - start_time3:.1f} seconds)')
    # visualize_reduction(data, target, tsne, 't-SNE')
    evaluate_model(data, target, tsne, 't-SNE')


    # 自编码器降维
    start_time4 = time()
    encoded_data = train_autoencoder_and_reduce_dimension(data)
    visualize_reduction(encoded_data, target, pca, 'Autoencoder')
    print(f'The Autoencoder deduction time cost {time() - start_time4:.1f} seconds)')
    evaluate_model(encoded_data, target, pca, 'Autoencoder')



if __name__ == "__main__":
    main()

