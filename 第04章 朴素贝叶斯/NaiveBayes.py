import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math
#3．朴素贝叶斯法利用贝叶斯定理与学到的联合概率模型进行分类预测。
#将输入 𝑥 分到后验概率最大的类 𝑦 。
#后验概率最大等价于0-1损失函数时的期望风险最小化。
#模型：高斯模型，多项式模型，伯努利模型

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_test[0], y_test[0])

#此处选用高斯朴素贝叶斯
class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
        # 然后返回由这些元组组成的列表。如果各个迭代器的元素个数不一致，
        # 则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
        # >>>a = [1,2,3]
        # >>> b = [4,5,6]
        # >>> c = [4,5,6,7,8]
        # >>> zipped = zip(a,b)     # 打包为元组的列表
        # [(1, 4), (2, 5), (3, 6)]
        #此处的zip(*train_data) 是将训练集的各个维度分别求和
        #然后求各个维度的均值和标准差
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label: self.summarize(value)
            for label, value in data.items()
        }
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(
            self.calculate_probabilities(X_test).items(),
            key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))

model = NaiveBayes()
model.fit(X_train, y_train)
print(model.predict([4.4,  3.2,  1.3,  0.2]))
print(model.score(X_test, y_test))

#scikit-learn实例
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.predict([[4.4,  3.2,  1.3,  0.2]]))

from sklearn.naive_bayes import BernoulliNB, MultinomialNB # 伯努利模型和多项式模型