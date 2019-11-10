

- ```
    class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=None, **kwargs)
                                           
    ```

- n_neighbors：聚类个数。
- weights（权重）：决定算法如何分配权重，默认为uniform，表示远近权重都一样。参数选项如下：
     • 'uniform'：不管远近权重都一样；
      • 'distance'：权重和距离成反比，距离预测目标越近具有越高的权重。
      • 自定义函数：自定义一个函数，根据输入的坐标值返回对应的权重，达到自定义权重的目的。
- algorithm：算法的选择，默认值为auto。
  -  brute：暴力搜索，计算预测样本和**全部训练集样本**的距离，最后筛选出前 K 个最近的样本。不过当数据较小或比较稀疏时，无论选择哪个最后都会使用 'brute'
  - kd_tree ：KD 树是一种「二叉树」结构，就是把整个空间划分为特定的几个子空间，然后在合适的子空间中去搜索待预测的样本点。假设数据集样本数为 m，特征数为 n，则当**样本数量 m 大于 2 的 n 次方时，用 KD 树算法搜索效果会比较好**
  - ball_tree：对于一些分布不均匀的数据集，KD 树算法搜索效率并不好，为了优化就产生了球树这种算法。
  - ‘auto‘默认选项，自动选择合适的方法构建模型
- leaf_size：只有ball_tree和kd_tree才有必要。表示停止建子树的叶子节点数量的阈值。默认30。但如果数据量增多这个参数需要增大，否则速度过慢不说，还容易过拟合。
- p与metric：距离表示，当metric参数是"minkowski"的时候，p=1是manhattan_distance，p=2是euclidean_distance。默认为p=2
- metric：指定距离度量方法，一般都是使用欧式距离。
      • 'euclidean' ：欧式距离
      • 'manhattan'：曼哈顿距离
      • 'chebyshev'：切比雪夫距离
      • 'minkowski'： 闵可夫斯基距离，默认参数
- n_jobs：指定多少个CPU进行运算，默认是-1，也就是全部都算。