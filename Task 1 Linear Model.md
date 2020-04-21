# 1.机器学习基础概念
虽然任务里没有做这部分基础概念的要求，但是我觉得有些基础的还是提一下。并几乎所有的算法都有围绕以下知识去解读或者优化。  
## 监督学习（Supervised learning） 和无监督学习（Unsupervised learning）
西瓜书定义： 根据训练数据是否拥有标记信息来划分，有标记信息就是监督学习，否则为无监督学习。
举个栗子：分类、线性回归就是监督学习；聚类就是无监督学习。  
  
## 泛化（generalization）能力、偏差（Bias）、方差（Variance）
学得模型适用于新样本的能力，称为"泛化" 能力。  
**偏差**：描述的是预测值（估计值）的期望与真实值之间的差距。偏差越大，越偏离真实数据，如下图第二行所示。  
**方差**：描述的是预测值的变化范围，离散程度，也就是离其期望值的距离。方差越大，数据的分布越分散，如下图右列所示。  

<img src="https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E6%96%B9%E5%B7%AE%E5%92%8C%E5%81%8F%E5%B7%AE.jpg" width=356 height=356 />
   
## 过拟合（Underfit）、欠拟合（Overfitting）
**过拟合**：指在拟合一个模型时，使用过多参数，导致模型过于复杂，泛化能力太低。如下图第三个所示，它完美的适应了所有的训练数据，但是这种模型在新样本中的表现非常差。  
**欠拟合**：与过拟合相反，模型过于简单，不能够挖掘出规律，或者挖掘出的规律太简单而没有意义。如下图第一个所示，直接一条直线穿过。
![image](https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E6%8B%9F%E5%90%88%E8%AF%B4%E6%98%8E.jpg)
  
**如何识别欠拟合**：在训练数据上表现不好，当然在测试数据上一般也不可能好到哪去。  
**如何改善欠拟合问题**：  
1、将数据整理成适合该模型的，比如引入新的特征；  
2、尝试调整或者引入更多的参数；  
3、改用其他模型或用集成模型方法    
  
**如何识别过拟合**：在训练数据上表现很好，但是在测试数据上很差。  
**如何改善过拟合问题**：  
1、交叉检验，通过交叉检验得到较优的模型参数;  
2、特征选择，减少特征数或使用较少的特征组合，对于按区间离散化的特征，增大划分的区间;  
3、正则化，常用的有 L1、L2 正则。而且 L1正则还可以自动进行特征选择; 如果有正则项则可以考虑增大正则项参数 lambda;  
4、增加训练数据可以有限的避免过拟合;  
5、Bagging ,将多个弱学习器Bagging 一下效果会好很多，比如随机森林等.  
6、提早停止策略，不让模型过于复杂。比如剪枝  
  
## 参考资料
【机器学习防止欠拟合、过拟合方法】 https://zhuanlan.zhihu.com/p/29707029  
【方差和偏差】http://scott.fortmann-roe.com/docs/BiasVariance.html  

# 2.线性回归原理
线性回归模型形式：
![image](https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%85%AC%E5%BC%8F.png)
  
![image](https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%85%AC%E5%BC%8F-%E5%90%91%E9%87%8F%E8%A1%A8%E7%A4%BA.png)  
  
【机器学习算法系列（2）：线性回归】https://plushunter.github.io/2017/01/08/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E7%B3%BB%E5%88%97%EF%BC%882%EF%BC%89%EF%BC%9A%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92/  

# 3.线性回归损失函数、代价函数、目标函数
## 先解释下三个函数的含义：  
**Loss Function** 是定义在单个样本上的，算的是一个样本的误差。  
**Cost Function** 是定义在整个训练集上的，是所有样本误差的平均，也就是损失函数的平均。  
**Object Function** 定义为：Cost Function + 正则化项（*λJ(f)*）。  
（ 定义一个函数 J(f)，这个函数专门用来度量模型的复杂度，在机器学习中也叫正则化(regularization)。常用的有 *L1* ,*L2*范数。）  
  
## 线性回归的损失函数：
  
![image](https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png)
  
## 线性回归的代价函数：
  
![image](https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E4%BB%A3%E4%BB%B7%E5%87%BD%E6%95%B0.png)
  
## 线性回归的目标函数：
  
![image](https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0.png)
  
## 参考资料
【机器学习中的目标函数、损失函数、代价函数有什么区别？】https://www.zhihu.com/question/52398145  
【线性回归之代价函数除2m】https://blog.csdn.net/u010106759/article/details/50380442  
  
# 4.优化方法
最优化，就是： 
1. 构造一个合适的目标函数，使得这个目标函数取到极值的解就是你所要求的东西； 
2. 找到一个能让这个目标函数取到极值的解的方法。

求函数极值，也就是对函数求导。通常情况下目标函数可能有多个极值，如何找到适合的极值就是下面要介绍的一些方法。  

## **最小二乘法**：   
在线性回归中，最小二乘法就是试图找到一条直线，使所有样本到直线上的欧氏距离之和最小。一般可直接利用矩阵求解。如果遇到有多个解或者属性数量大于样本数而无法直接求解的时候，常见的做法是引入正则化项。  
  
## **梯度下降法**：  
梯度下降法的优化思想是用当前位置负梯度方向作为搜索方向，因为该方向为当前位置的最快下降方向，所以也被称为是”最速下降法“。最速下降法越接近目标值，步长越小，前进越慢。  
  
梯度下降法的缺点：  
（1）靠近极小值时收敛速度减慢；  
（2）直线搜索时可能会产生一些问题；  
（3）可能会“之字形”地下降。  


## **牛顿法**  
牛顿法是一种在实数域和复数域上近似求解方程的方法。方法使用函数 *f (x)* 的泰勒级数的前面几项来寻找方程 *f (x)* = 0的根。牛顿法最大的特点就在于它的收敛速度很快。  
由于牛顿法是基于当前位置的切线来确定下一次的位置，所以牛顿法又被很形象地称为是"切线法"。  
牛顿法的搜索路径（二维情况）如下图所示：  
  
![image](https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E7%89%9B%E9%A1%BF%E6%B3%95%E6%90%9C%E7%B4%A2%E8%B7%AF%E5%BE%84.gif)  
  
从本质上去看，牛顿法是二阶收敛，梯度下降是一阶收敛，所以牛顿法就更快。如果更通俗地说的话，比如你想找一条最短的路径走到一个盆地的最底部，梯度下降法每次只从你当前所处位置选一个坡度最大的方向走一步，牛顿法在选择方向时，不仅会考虑坡度是否够大，还会考虑你走了一步之后，坡度是否会变得更大。所以，可以说牛顿法比梯度下降法看得更远一点，能更快地走到最底部。（牛顿法目光更加长远，所以少走弯路；相对而言，梯度下降法只考虑了局部的最优，没有全局思想。）  
根据wiki上的解释，从几何上说，牛顿法就是用一个二次曲面去拟合你当前所处位置的局部曲面，而梯度下降法是用一个平面去拟合当前的局部曲面，通常情况下，二次曲面的拟合会比平面更好，所以牛顿法选择的下降路径会更符合真实的最优下降路径。  

优点：二阶收敛，收敛速度快；   
缺点：牛顿法是一种迭代算法，每一步都需要求解目标函数的Hessian矩阵的逆矩阵，计算比较复杂。  
  
## **拟牛顿法**  
拟牛顿法的本质思想是改善牛顿法每次需要求解复杂的Hessian矩阵的逆矩阵的缺陷，它使用正定矩阵来近似Hessian矩阵的逆，从而简化了运算的复杂度。拟牛顿法和最速下降法一样只要求每一步迭代时知道目标函数的梯度。通过测量梯度的变化，构造一个目标函数的模型使之足以产生超线性收敛性。这类方法大大优于最速下降法，尤其对于困难的问题。另外，因为拟牛顿法不需要二阶导数的信息，所以有时比牛顿法更为有效。  
   
 ## 参考资料  
【最小二乘法本质】https://www.zhihu.com/question/37031188  
【如何通俗地理解“最大似然估计法”】https://www.matongxue.com/madocs/447.html  
【最优化问题的简洁介绍是什么】https://www.zhihu.com/question/26341871  
【常见的几种最优化方法】https://www.cnblogs.com/maybe2030/p/4751804.html   
【牛顿法】https://zh.wikipedia.org/zh-hans/%E7%89%9B%E9%A1%BF%E6%B3%95
   
# 5.线性回归评估指标
均方误差（MSE）  
<img src="https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE.png" width=356 height=125 />
  
均方根误差（RMSE）  
<img src="https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AE.png" width=356 height=125 />
  
MAE（平均绝对误差）  
<img src="https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/%E5%B9%B3%E5%9D%87%E7%BB%9D%E5%AF%B9%E8%AF%AF%E5%B7%AE.png" width=356 height=125 />  
  
R Squared（R方）  
<img src="https://github.com/magicyang5000/Datawhale-Algorithm-Principle/blob/master/images/R%E6%96%B9.png" width=356 height=356 />
  
**一般用R方来评估模型效果。**  
  
## 参考资料
【回归评价指标MSE、RMSE、MAE、R-Squared】https://www.jianshu.com/p/9ee85fdad150  

# 6.Sklearn.LinearRegression函数参数详解
## 函数
sklearn.linear_model.LinearRegression(fit_intercept=True , normalize=False , copy_X=True , n_jobs=None) 
  
## **参数**
* **fit_intercept** : boolean, optional, default True  
  * 是否加入截距，截距也就是经常出现在式子中的b，默认是加入的，也可以设置为False,这时假设数据已经中心化  
  
* **normalize** : boolean, optional, default False  
  * 设置为True的话则对数据进行fit前会进行规范化处理，  
  * 如果fit_intercept是False的话，那么normalize参数会被忽略。  
  
* **copy_X** : boolean, optional, default True  
  * 默认复制X结果而不是直接修改X，X是训练样本数据  
  
* **n_jobs** : int, optional, default 1  
  * 也就是用来进行计算的任务数，默认只有一个，如果设置为-1那么便等于cpu的核心数  
  
## **属性(模型可见的参数)**
* **coef_** : array, shape (n_features, ) or (n_targets, n_features)  
  * 表示该线性回归的参数，如果有n个特征那么返回的参数也是n  
   
* **intercept_** : array  
  * 截距  
  
## **方法**
* **fit(X, y[, sample_weight])** Fit linear model.  
  * 拟合训练数据，可选参数为sample_weight，也就是每个样本的权重值，可能是你认为更为重要的特征，X为特征数据，y为label值  
  
* **get_params([deep])** Get parameters for this estimator. deep默认为True  
  * 返回一个字典，键为参数名，值为参数值，如下{‘copy_X’: True, ‘fit_intercept’: True, ‘n_jobs’: 1, ‘normalize’: True}  
    
* **predict(X)** Predict using the linear model  
  * 输入样本数据输出预测值  
    
* **score(X, y[, sample_weight])** Returns the coefficient of determination R^2 of the prediction.  
  * 就是R方值，这个系数的计算方式是这样的，1-（所有样本的残差的平方和/所有样本的类值减去类平均值的平方和），这个值可以是负数，最好的情况是1，说明所有数据都预测正确 
    
* set_params(**params) Set the parameters of this estimator.  
  * 设置估计器的参数，可以修改参数重新训练  
  
 ## 参考资料
【线性回归函数官方文档】https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html  
【线性回归函数中文解释】https://blog.csdn.net/u013019431/article/details/79962305  
