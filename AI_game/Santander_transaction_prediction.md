### 题目 Description

预测一个乘客在未来会不会交易，不计交易次数，只要有交易即为1，否则为0. kaggle原题[链接](https://www.kaggle.com/c/santander-customer-transaction-prediction)

Identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. 

You are provided with an anonymized dataset containing numeric feature variables, the binary `target` column, and a string `ID_code`column.

The task is to predict the value of `target` column in the test set.

### Data

提供了 $200k \times 202$ 的数据，200维的特征，通过EDA发现特征向量基本服从高斯分布，且各个变量之间是独立的，疑似经过 PCA 处理后的特征。target为0 和 1，其中处于1的占比为 0.1117，不平衡现象明显。

测试集为 $200k \times 201$，含有一行ID，与200维数据。目标为预测该用户的target（0/1）

### 1nd 解决方法

discussion[链接](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89003#latest-515846)

共使用600个特征，使用agumentation技巧扩充数据集，2.1NN/1LGBM融合。

magic特征为unique特征，作者称其通过EDA观察 LGB 树发现其只使用unique特征，从而注重挖掘数据的unique信息。并根据其出现次数构建了5个类别：

1. value在 target=1 的train data中出现多次
2. value在 target=0 的train data中出现多次
3. value在 target=0和target=1 的 train data中都出现多次
4. value在train data中只出现一次
5. value在train data和test data（仅包含真实数据） 中都是unique

由此可以构建200个特征（对每个原始特征transform），另外200个特征是数值特征，其将test+data中的unique数据替换为特征的均值。在没有处理掉fake时，作者用前200个特征用LGBM取得了0.910LB（大致前进0.09），增加后200个特征后取得0.914LB。处理掉fake之后，用LGBM取得0.921LB。

具体来说：作者使用10折交叉验证，并用不同的seed并在最后进行融合。采用augmentation技巧，即将同样target的行的特征进行shuffle后作为新的数据进行预测，对target=1的数据进行16次shuffle，对target=0的行进行4次shuffle，同时相应打上伪标签（test数据也被加入到里面进行训练，其中前2700个被设为1，后2000个作为0），用LGBM预测，0.92522 LB。

作者同时训练了一个NN模型，将 val 特征的原始value，unique类与数值特征先映射到embedding，可以构建200个embeddings，然后将200个embedding加权平均在输入到一串全连接网络中导出最终输出。加权的思想类似与attention，加权权重为前置的一个独立NN模块，这样保证最终每个特征用同样的方式训练。训练时对每个batch进行独立的augmentation，结果为0.92497PB，再把 test 加入训练后，0.92546PB。大致扫了下作者的源码，NN的全连接为32层，频繁使用小量的dropout（0.08）和 BN 层。NN的好处是可以提取到特征之间的关系，以及特征内部数据之间的group关系，同时NN需要训练的超参数更少。

### 2nd 解决方法

discussion[链接](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88939#latest-515844)

```
1. 移除fake测试集，保留100k的真实测试集，真实测试集的定义为其存在仅出现 1 次的特征，fake行是真实行拼凑起来的，会对结果造成影响。
2. 同时对训练集与测试集的各个特征进行标准化处理
3. 构建频率特征，按照小数点精度，构建4种频率特征
4. 将每一个原始特征与其频率特征取出来进行拼接，并增加类别特征（200个类别），构成新的训练集，在这种情况下相当于仅用原始一个特征来进行训练，而原始每个特征都是均等的（独立性假设）
5. 用 lightgbm 训练和预测
6. 将200个预测结果进行拼接，拼接公式见公式5
7. join真测试集的预测结果，将假测试集的结果置为0，进行提交
```

该参赛队伍[代码链接](https://github.com/KazukiOnodera/santander-customer-transaction-prediction/blob/master/py/990_2nd_place_solution_golf.py)

对 **#1** 解释： **magic**之一，在查看训练集与测试集的unique计数分布时，发现二者存在较大差距，所以测试集由真的测试集与假的测试集组成，假的测试集是防止用户进行LB Probe，真的测试集用来评分。假的测试集是真测试集拼凑的，所以特征不唯一，以此来排除它。运用该magic，大致可以从0.901提升到 LB 0.91，PB 0.907

对 **#4** 解释：用单个特征来进行预测，可以有更多的训练数据（$200k \times 200$ 行），训练时对单个特征的关注度也更高，个人理解该方式与比赛期间的augmentation技巧类似。

```
原始的单个特征为 value
通过 round+计数 扩展特征，最后再加上 特征类别（200个类别），形成的新的train数据为：
value, count_org, count_2, count_3, count_4, varnum ---> target
test数据集做同样处理
```

对 **#6** 解释：假设单个特征是依 target 独立的，也即满足下述假定：
$$
f(n)= \begin{cases} P({\bf x} \,|\, y=1)= \prod_i P(x_i  \,|\, y=1) \\ P({\bf x}  \,|\, y=0)= \prod_i P(x_i  \,|\, y=0) \end{cases} \tag 1
$$
我们的目标是由 $P(y=1  \,|\, x_i)$ 导出 $P(y=1  \,|\, {\bf x})$，其中 ${\bf x} = \{x_1, x_2,\dots,x_{200}\}$

$$
\zeta = \frac{P(y=1 \,|\, {\bf x})}{P(y=0 \,|\, {\bf x})} = \frac{P({\bf x} \,|\, y=1)\cdot P(y=1)}{P({\bf x} \,|\, y=0) \cdot P(y=0)}=\frac{P(y=1) \prod_i P(x_i \,|\, y=1)}{P(y=0) \prod_i P(x_i \,|\, y=0)} \tag 2
$$

按照训练集的统计，可设
$$
\lambda = \frac {P(y=1)} {P(y=0)} = \frac 1 9 \tag 3
$$
同时由朴素贝叶斯可知：
$$
\begin{align}
&P(x_i \,|\, y=1)  = \frac {P(y=1 \,|\, x_i) \cdot P(x_i)}{P(y=1)} \tag 4 \\
\zeta & =\frac 1 9 \cdot \prod_i \frac {P(y=1 \,|\, x_i) \cdot P(y=0)} {P(y=0 \,|\, x_i) \cdot P(y=1)} \\
 & =\frac 1 9 \cdot \prod_i \frac {9 \cdot P(y=1 \,|\, x_i)} {P(y=0 \,|\, x_i)} \tag 5
\end{align}
$$

### 3nd 解决方法

先取出测试集fake 数据，再把train和fake拼接在一起，然后做特征工程，其使用了unique计数、密度（count平滑后的结果）、前二者相除，故共800维特征。用LGBM训练，取得0.9225 LB，作者观察到单用一些特征训练的auc接近0.5，对最终的结果没有什么帮助，所以采用 Lasso 来自动削减该类特征。CV时优化了reg_alpha， max_bin， learning_rate， num_leaves 4个参数。

作者关键性的操作是用 CNN 进行二阶段的训练。将LGBM的预测结果作为一个特征加入到CNN中，对每个特征列保持相同的卷积核，最后再连一个全连接层，在NN里使用BN避免过拟合。最终平均7个CNN的结果作为最终结果。

### 其它magic

有看到可以通过将特征列 groupby 后求 方差var，并将方差加减原特征构成新的特征，400维统一放入到 lgb 中训练，当同时考虑加减并消去 fake 后可以取得 0.922的成绩。其实是变相的count，因为在groupby之后，unique的数据方差为0，只有count数较大的值才会有方差，anyway，也是一个比较有趣的思路。

### 收获

第一次尝试参赛，在末尾几天开始入场，自然没打算去取得好成绩，主要是疯狂去追 kernels 和看 Discussion，在比赛时段看和在结束之后看是两种不同的心态。在**比赛结束之前**，大部分人是比较焦虑的，Discussion 里有很多人发言，但是你也不能确定他们的idea是不是对的，只能用自己的知识和实践去推断，在这一期间，务必要保证自己的理智，多尝试与创新，而不要被误导到去钻牛角尖！在**比赛后**要做好比赛的回顾，因为比赛后有很多人会公布他们的思路和代码，这个时候就要去潜心学习下比你做的好的人的做法，反思不足，这样才能争取在之后的比赛中更进一步。



最后推荐几个kernels：

1. 第一名的 [kernel链接](https://www.kaggle.com/fl2ooo/nn-wo-pseudo-1-fold-seed)
2. 号称actual winner的最早fake test提出者：[kernel链接](https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split) ，可以说比赛时注意到这个kernel，银牌不是问题了
3. 参赛期间看得一个[kernel](https://www.kaggle.com/hjd810/keras-lgbm-aug-feature-eng-sampling-prediction) ，里面包行augmentation，代码风格不错，读起来挺舒服，可以在此基础上做自己的修改与尝试
4. 第三名的 [kernel链接](https://www.kaggle.com/nawidsayed/lightgbm-and-cnn-3rd-place-solution)