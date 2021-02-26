## 逻辑回归

[toc]

### 线性回归 to 逻辑回归

**本质**：逻辑回归的本质就是在线性回归的基础上做了一个非线性的映射（变换），使得算法具有非线性的属性。
Q1:为什么要加这个非线性的变换呢？
答：因为对于线性回归，预测的变量是连续型的变量，不适合于分类型的离散变量（eg y=0或者y=1）。原因在于线性回归的定义可能让y大于0或者小于1，现在我们需要让0<=y<=1。就用sigmoid函数做映射函数。sigmod函数在负无穷大时，趋向于0；正无穷大时，趋向于1。

为什么不采用分段函数而要采用sigmoid函数呢？因为sigmoid函数是连续的，阶梯函数是不连续且不可微的

### 决策边界

$$h_\theta(x) = \frac{1}{1+e^{-\theta^T x}}$$

- 当$$h_\theta(x)\geq0.5$$时，预测y=1
- 当$$h_\theta(x)<0.5$$时，预测y=0

等同于

- 当$$\theta^T\ge0$$时，预测y=1
- 当$$\theta^T<0$$时，预测y=0

假设对于2个特征变量的函数，$$h_\theta(x) = \frac{1}{1+e^-{(\theta_0+\theta_1x_1+\theta_2x_2)}}$$，最后求解为

$$\begin{cases} \ \theta_0=-3\\ \theta_1=1\\\theta_2=1\end{cases}$$

则当$$\theta_0+\theta_1x_1+\theta_2x_2\ge0$$时，y=1;当$$\theta_0+\theta_1x_1+\theta_2x_2<0$$时，y=0;那么决策边界就是$$\theta_0+\theta_1x_1+\theta_2x_2=0$$这条线；

另外一种情况就是可能 $$-1+x_1^2+x_2^2=0$$也是决策边界，代表一个圆；

### 损失函数

Q2: 为什么不用线性回归的损失函数作为逻辑回归的损失函数呢？

答：继续使用线性回归的损失函数，会导致代价函数变成一个非凸函数。这就导致会有很多局部最小值，用梯度下降法很难保证其收敛到全局最小值。

Q3：损失函数特点？

- 当真实类别y=1时，异常概率越大，损失越小
- 当真实类别y=0时，异常概率越小，损失越大

$$J= \begin{cases} -log(p)& \text{y=1}\\ -log(1-p)& \text{y=0} \end{cases}$$

通过控制系数的方法，将两个方程联系起来，可以得到，单个样本的损失函数为：

$$J = -log(p)-(1-p)log(1-p)$$

全部样本的损失可以取平均值

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^m y^ilog(p^i)+(1-y^i)log(1-p^i)$$

Q4:通过最大似然函数求解损失函数

### 代码实现

我们在线性回归的基础上，修改得到逻辑回归。主要内容为：

- 定义sigmoid方法，使用sigmoid方法生成逻辑回归模型
- 定义损失函数，并使用梯度下降法得到参数
- 将参数代入到逻辑回归模型中，得到概率
- 将概率转化为分类



    import numpy as np
    
    # 因为逻辑回归是分类问题，因此需要对评价指标进行更改
    
    from .metrics import accuracy_score
    
    class LogisticRegression:
    def __init__(self):
        """初始化Logistic Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None
    
    """
    定义sigmoid方法
    参数：线性模型t
    输出：sigmoid表达式
    """
    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))
    
    """
    fit方法，内部使用梯度下降法训练Logistic Regression模型
    参数：训练数据集X_train, y_train, 学习率, 迭代次数
    输出：训练好的模型
    """
    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
    
        """
        定义逻辑回归的损失函数
        参数：参数theta、构造好的矩阵X_b、标签y
        输出：损失函数表达式
        """
        def J(theta, X_b, y):
            # 定义逻辑回归的模型：y_hat
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                # 返回损失函数的表达式
                return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')
        """
        损失函数的导数计算
        参数：参数theta、构造好的矩阵X_b、标签y
        输出：计算的表达式
        """
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)
    
        """
        梯度下降的过程
        """
        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter += 1
            return theta
    
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        # 梯度下降的结果求出参数heta
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        # 第一个参数为截距
        self.intercept_ = self._theta[0]
        # 其他参数为各特征的系数
        self.coef_ = self._theta[1:]
        return self
    
    """
    逻辑回归是根据概率进行分类的，因此先预测概率
    参数：输入空间X_predict
    输出：结果概率向量
    """
    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
    
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        # 将梯度下降得到的参数theta带入逻辑回归的表达式中
        return self._sigmoid(X_b.dot(self._theta))
    
    """
    使用X_predict的结果概率向量，将其转换为分类
    参数：输入空间X_predict
    输出：分类结果
    """
    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        # 得到概率
        proba = self.predict_proba(X_predict)
        # 判断概率是否大于0.5，然后将布尔表达式得到的向量，强转为int类型，即为0-1向量
        return np.array(proba >= 0.5, dtype='int')
    
    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
    
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    
    def __repr__(self):
        return "LogisticRegression()"
下面我们使用Iris数据集，来调用上面实现的逻辑回归。

数据展示

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y<2,:2]
y = y[y<2]
plt.scatter(X[y==0,0], X[y==0,1], color="red")
plt.scatter(X[y==1,0], X[y==1,1], color="blue")
plt.show()


```

```
from myAlgorithm.model_selection import train_test_split
from myAlgorithm.LogisticRegression import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 查看训练数据集分类准确度

log_reg.score(X_test, y_test)
```

