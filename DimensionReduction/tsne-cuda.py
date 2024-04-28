from tsnecuda import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def load_data():
    # 更换成其他类型的数据集
    data = np.loadtxt(r"../data/mnist2500_X.txt")
    target = np.loadtxt(r"../data/mnist2500_labels.txt")
    return data, target

def visualize_reduction(reduced_data, target, title):
    plt.title(title)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target)
    plt.show()

def evaluate_model(reduced_data, target, method_name):
    X_train, X_test, y_train, y_test = train_test_split(reduced_data, target, random_state=0)
    model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f'{method_name} accuracy: {score}')

def main():
    data, target = load_data()
    tsnecuda = TSNE(n_components=2, perplexity=15, learning_rate=10)
    tsnecuda_data = tsnecuda.fit_transform(data)
    visualize_reduction(tsnecuda_data, target, 't-SNE-CUDA')
    evaluate_model(tsnecuda_data, target, 't-SNE-CUDA')

if __name__ == "__main__":
    main()
