from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve, f1_score
import pandas as pd
import numpy as np

"""
training  输入训练集数据和参数，训练后得到模型  
serving 输入训练完成的模型、应用数据和应用参数，得到应用结果
evaluating  输入应用结果、真实结果和评估参数，得到评估指标和结果

点击训练 = training(训练集)+serving(训练集)
点击评估 = serving(验证集) + evaluating(训练集/验证集)
"""


class AdaBoost:
    def __init__(self):
        pass

    def training(self, x_train, y_train, n_estimators: int = 10, **kwargs):
        """

        :param x_train: 训练集特征,df
        :param y_train: 训练集标签，df
        :param n_estimators: 超参
        :param kwargs: 训练完成的模型
        :return:
        """

        self.model = AdaBoostClassifier(n_estimators=n_estimators)
        self.model = self.model.fit(x_train, y_train)

        return self

    def serving(self, x_predict, **kwargs):
        """

        :param model: training函数返回的model
        :param x_predict: 待预测的数据， df
        :param kwargs: serving的参数
        :return: 预测结果，df
        """
        y_predict = pd.DataFrame()
        y_predict['probability'] = self.model.predict_proba(x_predict)[:, 1]
        y_predict['prediction'] = self.model.predict(x_predict)
        return y_predict


def evaluating(y_predict, y_true, **kwargs):
    """

    :param y_predict: serving输出的预测结果，df
    :param y_true: 目标，df
    :param kwargs: 评估参数
    :return: 重要的评估指标，dict格式；评估结果，df格式
    """

    f1score = f1_score(y_true, y_predict['prediction'])
    evaluate_dict = {'f1score': f1score}

    precision, recall, thresholds = precision_recall_curve(y_true, y_predict['probability'])
    evaluate_df = pd.DataFrame([precision, recall, thresholds]).T
    evaluate_df.columns = ['precision', 'recall', 'thresholds']

    return evaluate_dict, evaluate_df


# 示例

from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=2)
x_train, x_test, y_train, y_test = train_test_split(X, y)

# 训练

model = AdaBoost().training(x_train, y_train, n_estimators=7)
y_train_predict = AdaBoost().serving(x_train)
# 评估

y_test_predict = AdaBoost.serving(x_test, model)

evaluate_train_dict, evaluate_train_df = evaluating(y_train_predict, y_train)
evaluate_test_dict, evaluate_test_df = evaluating(y_test_predict, y_test)
