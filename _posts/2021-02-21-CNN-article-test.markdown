----------

> * 1.前言
> * 2.激活函数$Relu$
> * 3.全连接网络 $VS$ 卷积网络
> * 4.卷积神经网络是啥
> * 5.卷积神经网络输出值的计算
> * 6.卷积神经网络的训练

----------


## 1.前言

- 我们介绍了全连接神经网络，以及它的训练和使用。然而，这种结构的网络对于图像识别任务来说并不是很合适。

- 现介绍一种更适合图像、语音识别任务的神经网络结构——**卷积神经网络(Convolutional Neural Network, CNN)**。说卷积神经网络是最重要的一种神经网络也不为过，它在最近几年大放异彩，几乎所有图像、语音识别领域的重要突破都是卷积神经网络取得的，比如谷歌的GoogleNet、微软的ResNet等，打败李世石的AlphaGo也用到了这种网络。


----------


## 2.激活函数$Relu$

- 最近几年卷积神经网络中，激活函数往往不选择 $sigmoid$ 或 $tanh$ 函数，而是选择 **$Relu$ 函数**。$Relu$ 函数的定义是：
    $$
        f(x) = max(0, x)
    $$

- $Relu$ 函数图像如下图所示：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/cstgvnj8n2qpkn6mzj29ru8f/image_1dqh40pfa1b851p1516vc34q1c8h13.png" width="250" />
    </div>
    <br>

- $Relu$ 函数作为激活函数，有下面几大优势：
    - **速度快**
        -  和$Sigmoid$ 函数需要计算指数和倒数相比，$Relu$ 函数其实就是一个 $max(0,x)$ ，计算代价小很多。
    - **减轻梯度消失问题**
        - 回忆一下计算梯度的公式：
    $$
        \nabla=\sigma'\delta x
    $$
        - 其中，**$sigmoid$ 函数的导数**为 $\sigma'$
        - 在使用反向传播算法进行梯度计算时，每经过一层 $sigmoid$ 神经元，梯度就要乘上一个 **$sigmoid$ 函数的导数**
        - 从下图可以看出，**$sigmoid$ 函数的导数**函数最大值是 1/4：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/tzx06x71ynszbeowbgkt25m5/image_1dqh41nbtaqga418sm19mlo7l20.png" width="350" />
    </div>
    <br>
        - 因此，乘一个 **$sigmoid$ 函数的导数** 会**导致梯度越来越小**，这对于深层网络的训练是个很大的问题。
        - 而**$relu$ 函数的导数是1，不会导致梯度变小**。
        - 当然，激活函数仅仅是导致梯度减小的一个因素，但无论如何在这方面 $relu$ 的表现强于 $sigmoid$ 。使用 $relu$ 激活函数可以让你训练更深的网络。
    - **稀疏性**
        - 通过对大脑的研究发现，大脑在工作的时候只有大约5%的神经元是激活的，而采用 $sigmoid$ 激活函数的人工神经网络，其激活率大约是50%。有论文声称人工神经网络在15%-30%的激活率时是比较理想的。因为**$relu$ 函数在输入小于0时是完全不激活的**，因此**可以获得一个更低的激活率**。


----------


        
## 3.全连接网络 VS 卷积网络

- 全连接神经网络之所以不太适合图像识别任务，**主要有以下几个方面的问题**：
    - **参数数量太多** 考虑一个输入1000*1000像素的图片(一百万像素，现在已经不能算大图了)，输入层有1000*1000=100万节点。假设第一个隐藏层有100个节点(这个数量并不多)，那么仅这一层就有(1000*1000+1)*100=1亿参数，这实在是太多了！我们看到图像只扩大一点，参数数量就会多很多，因此它的扩展性很差。
    - **没有利用像素之间的位置信息** 对于图像识别任务来说，每个像素和其周围像素的联系是比较紧密的，和离得很远的像素的联系可能就很小了。如果一个神经元和上一层所有神经元相连，那么就相当于对于一个像素来说，把图像的所有像素都等同看待，这不符合前面的假设。当我们完成每个连接权重的学习之后，最终可能会发现，有大量的权重，它们的值都是很小的(也就是这些连接其实无关紧要)。努力学习大量并不重要的权重，这样的学习必将是非常低效的。
    - **网络层数限制** 我们知道网络层数越多其表达能力越强，但是通过梯度下降方法训练深度全连接神经网络很困难，因为全连接神经网络的梯度很难传递超过3层。因此，我们不可能得到一个很深的全连接神经网络，也就限制了它的能力。

- 那么，卷积神经网络又是怎样解决这个问题的呢？**主要有三个思路**：
    - **局部连接**  这个是最容易想到的，每个神经元不再和上一层的所有神经元相连，而只和一小部分神经元相连。这样就减少了很多参数。
    - **权值共享**  一组连接可以共享同一个权重，而不是每个连接有一个不同的权重，这样又减少了很多参数。
    - **下采样**  可以使用Pooling来减少每层的样本数，进一步减少参数数量，同时还可以提升模型的鲁棒性。

- 对于图像识别任务来说，**卷积神经网络通过尽可能保留重要的参数，去掉大量不重要的参数，来达到更好的学习效果**。


----------


## 4.卷积神经网络是啥

- 首先，我们先获取一个感性认识，下图是一个卷积神经网络的示意图：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/hoiuj0br7u1awt2ekvqnp09y/image_1dqh474s414f890lppa1mfo1mqu2d.png" width="450" />
    </div>
    <br>

- 网络架构
    - 如上图所示，一个**卷积神经网络**由若干**卷积层**、**Pooling层**、**全连接层**组成。你可以构建各种不同的卷积神经网络，它的常用架构模式为：
        - `INPUT -> [[CONV]*N -> POOL?]*M -> [FC]*K`
    - 也就是N个卷积层叠加，然后(可选)叠加一个Pooling层，重复这个结构M次，最后叠加K个全连接层。
    - 对于上图展示的卷积神经网络：
        - `INPUT -> CONV -> POOL -> CONV -> POOL -> FC -> FC`
    - 按照上述模式可以表示为：
        - `INPUT -> [[CONV]*1 -> POOL]*2 -> [FC]*2`
    - 也就是：
        - `N=1, M=2, K=2`

- 三维的层结构
    - 从上图我们可以发现**卷积神经网络的层结构和全连接神经网络的层结构有很大不同**。**全连接神经网络**每层的神经元是按照一维排列的，也就是排成一条线的样子；而**卷积神经网络**每层的神经元是按照**三维**排列的，也就是排成一个长方体的样子，有**宽度**、**高度**和**深度**。
    - 对于上图展示的神经网络，我们看到**输入层的宽度和高度对应于输入图像的宽度和高度**，而它的深度为1。接着，第一个卷积层对这幅图像进行了卷积操作(后面我们会讲如何计算卷积)，得到了三**个Feature Map**。这里的 "3" 可能是让很多初学者迷惑的地方，实际上，就是这个卷积层包含三个Filter，也就是三套参数，每个Filter都可以把原始输入图像卷积得到一个Feature Map，三个Filter就可以得到三个Feature Map。至于一个卷积层可以有多少个Filter，那是可以自由设定的。也就是说，**卷积层的Filter个数也是一个超参数**。我们可以把Feature Map可以看做是通过卷积变换提取到的图像特征，三个Filter就对原始图像提取出三组不同的特征，也就是得到了三个Feature Map，也称做三个**通道(channel)**。
    - 继续观察上图，在**第一个卷积层**之后，Pooling层对三个Feature Map做了**下采样**(后面我们会讲如何计算下采样)，得到了三个更小的Feature Map。接着，是**第二个卷积层**，它有5个Filter。每个Fitler都把前面**下采样**之后的**3个Feature Map**卷积**在一起，得到一个新的Feature Map**。这样，**5个Filter就得到了5个Feature Map**。接着，是**第二个Pooling，继续对5个Feature Map**进行**下采样**，得到了5个更小的Feature Map。
    - 上图所示网络的最后两层是全连接层。第一个全连接层的每个神经元，和上一层5个Feature Map中的每个神经元相连，第二个全连接层(也就是输出层)的每个神经元，则和第一个全连接层的每个神经元相连，这样得到了整个网络的输出。
- 至此，我们对**卷积神经网络**有了最基本的感性认识。接下来，我们将介绍**卷积神经网络**中各种层的计算和训练。


----------


## 5.卷积神经网络输出值的计算

### 5.1 卷积层输出值的计算

#### 5.1.1 深度为1，步幅为1

- 我们用一个简单的例子来讲述如何计算**卷积**，然后，我们抽象出**卷积层**的一些重要概念和计算方法。

- 假设有一个5*5的图像，使用一个3*3的filter进行卷积，想得到一个3*3的Feature Map，如下所示：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/nfrl7r3my4yu22scccgcdc5y/image_1dqgv8s18cjo1iph2au1sia186d6k.png" width="350" />
    </div>
    <br>

- 为了清楚的**描述卷积计算过程**，我们作如下表示
    - 对图像的每个像素进行编号，**图像的第 $i$ 行第 $j$ 列元素**表示为：$x_{i,j}$
    - 对filter的每个权重进行编号，**第 $m$ 行第 $n$ 列权重**表示为：$w_{m,n}$
    - **filter的偏置项**表示为：$w_b$
    - 对Feature Map的每个元素进行编号，**Feature Map的第 $i$ 行第 $j$ 列元素**表示为：$a_{i,j}$
    - 用 $f$ 表示**激活函数(这个例子选择relu函数作为激活函数)**。

- 然后，使用下列公式计算卷积：
    $$
        a_{i,j}=f(\sum_{m=0}^{2}\sum_{n=0}^{2}w_{m,n}x_{i+m,j+n}+w_b)  \qquad  (式1)
    $$
    - 例如，对于Feature Map左上角元素 $a_{0,0}$ 来说，其卷积计算方法为：
    $$
        \begin{array}{l}
            a_{0,0}
            &=
            f(\sum_{m=0}^{2}\sum_{n=0}^{2}w_{m,n}x_{m+0,n+0}+w_b)  \\
            &=
            relu(w_{0,0}x_{0,0}+w_{0,1}x_{0,1}+w_{0,2}x_{0,2}+w_{1,0}x_{1,0}+w_{1,1}x_{1,1}+w_{1,2}x_{1,2}+w_{2,0}x_{2,0}+w_{2,1}x_{2,1}+w_{2,2}x_{2,2}+w_b)  \\
            &=
            relu(1+0+1+0+1+0+0+0+1+0)  \\
            &=
            relu(4)\\
            &=
            4
        \end{array}
    $$
        - 计算结果如下图所示：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/mun6qpjtbmklo5rqjontfv51/image_1dqh89s2lrap111rlutkqn1ccd2q.png" width="350" />
    </div>
    <br>
    - 接下来，Feature Map的元素 $a_{0,1}$ 的卷积计算方法为：
    $$
        \begin{array}{l}
            a_{0,1}
            &=
            f(\sum_{m=0}^{2}\sum_{n=0}^{2}w_{m,n}x_{m+0,n+1}+w_b)  \\
            &=
            relu(w_{0,0}x_{0,1}+w_{0,1}x_{0,2}+w_{0,2}x_{0,3}+w_{1,0}x_{1,1}+w_{1,1}x_{1,2}+w_{1,2}x_{1,3}+w_{2,0}x_{2,1}+w_{2,1}x_{2,2}+w_{2,2}x_{2,3}+w_b)  \\
            &=
            relu(1+0+0+0+1+0+0+0+1+0)  \\
            &=
            relu(3)\\
            &=
            3
        \end{array}
    $$
        - 计算结果如下图所示：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/78fngxyme2zdrx0477de6m1k/image_1dqh8l7o5s7612rg1rss2ei1ph837.png" width="350" />
    </div>
    <br>
    - 可以依次计算出Feature Map中所有元素的值。下面显示了整个Feature Map的计算过程：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/qnn3b0j5rgjsceyl6b21olp4/image_1dqhjp2puhu610vfp6utgu45454.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/stfuo0abg2bha3eahziuf1xs/image_1dqhjpfn31f12rfr5ss1qq9n295h.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/3qt06id7ns3751mjeusxmcak/image_1dqhjpu3vj861fqtskdhugt05u.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/5bw5oikp7ya9m3cyz0z8eo8i/image_1dqhjq9t31l06bq7eq4ol7p3t6b.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/o6i3ojdzqrgcnqhwbjbod6qs/image_1dqhjqm892ulvsl14q71lcqkou6o.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/k4jok9hhx4ebkyilepotldii/image_1dqhk039oaf5185h1h011s8gs9p75.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/w7c9orlri7bfjxa0uyx3iold/image_1dqhk0ebm12ja1kgp12a01cqdfmn7i.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/4foirqnv5m5hgubgmlmd7b7r/image_1dqhk0r501f3aefj131v1krg1pp27v.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/lidirwlwqcuz80ojvwaocpyf/image_1dqhk142vkff4e91c781fo21js68c.png" width="350" />
    </div>
    <br>

#### 5.1.2 深度为1，步幅大于1

- 上面的计算过程中，**步幅(stride)**为1，**步幅可以设为大于1的数**。

- 例如，当步幅为2时，Feature Map计算如下：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/43lyybo6ibhqv4g7cqt0kyn8/image_1dqipfdfjrte5u41r7hk51d2l19.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/fydd0smxq5y2ud80vvl25t90/image_1dqipft661msn78kubf1lgv1op71m.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/m0prwqv5c0sombk5broqin0n/image_1dqiph2f3rgq1086uab2h7a113j.png" width="350" />
    <img src="http://static.zybuluo.com/georay53/mnlioqyqs9y0h2om0oskkn1q/image_1dqiphgmupvl1kr3ichv3m1knh40.png" width="350" />
    </div>
    <br>
    - 我们注意到，当步幅设置为2的时候，Feature Map就变成2*2了。这说明**图像大小、步幅和卷积后的Feature Map大小是有关系的**。事实上，它们满足下面的关系：
    $$
        \begin{array}{c}
            W_2 &= (W_1 - F + 2P)/S + 1\qquad(式2)  \\
            H_2 &= (H_1 - F + 2P)/S + 1\qquad(式3)
        \end{array}
    $$
    - 在上面两个公式中**，$W_2$ 是卷积后Feature Map的宽度**；**$W_1$ 是卷积前图像的宽度**；**$F$ 是filter的宽度**；**$P$ 是Zero Padding数量**；**$S$ 是步幅**；**$H_2$ 是卷积后Feature Map的高度**；**$H_1$ 是卷积前图像的高度**
    - Zero Padding是指在原始图像周围补几圈0。如果的 $P$ 值是1，那么就补1圈0；
    - **式2和式3本质上是一样的**。
    - 以本例来说
        - 图像宽度：$W_1=5$
        - filter宽度：$F=3$
        - Zero Padding：$P=0$
        - 步幅：$S=2$
        - 则
    $$
        \begin{array}{l}
            W_2 &= (W_1 - F + 2P)/S + 1  \\
            &= (5 - 3 + 0)/2 + 1  \\
            &=2
        \end{array}
    $$
        - 说明Feature Map宽度是2。同样，我们也可以计算出Feature Map高度也是2。

#### 5.1.3 深度大于1

- 前面我们已经讲了深度为1的卷积层的计算方法，如果**深度大于1**怎么计算呢？其实也是类似的。如果**卷积前的图像深度为$D$，那么相应的filter的深度也必须为$D$**。我们扩展一下式1，得到了深度大于1的卷积计算公式：
    $$
        a_{i,j}=f(\sum_{d=0}^{D-1}\sum_{m=0}^{F-1}\sum_{n=0}^{F-1}w_{d,m,n}x_{d,i+m,j+n}+w_b)  \qquad  (式4)
    $$
    - 在式4中，
        - $D$ 是深度；$F$ 是filter的大小(宽度或高度，两者相同)；
        - filter的第 $d$ 层第 $m$ 行第 $n$ 列权重表示为：$w_{d,m,n}$
        - 图像的第 $d$ 层第 $i$ 行第 $j$ 列像素表示为：$a_{d,i,j}$
        - 其它的符号含义和式1是相同的，不再赘述。

- 我们前面还曾提到，**每个卷积层可以有多个filter**。每个filter和原始图像进行卷积后，都可以得到一个Feature Map。因此，**卷积后Feature Map的深度(个数)和卷积层的filter个数是相同的**。

- 下面显示了包含两个filter的卷积层的计算。我们可以看到7*7*3输入，经过两个3*3*3filter的卷积(**步幅为2**)，得到了3*3*2的输出。另外我们也会看到下图的Zero padding是1，也就是在输入元素的周围补了一圈0。**Zero padding对于图像边缘部分的特征提取是很有帮助的**。
    <div align="middle">
    <img src="http://upload-images.jianshu.io/upload_images/2256672-958f31b01695b085.gif" width="450" />
    </div>
    <br>

#### 5.1.4 卷积层计算总结

- 以上就是卷积层的计算方法。这里面体现了**局部连接**和**权值共享**：每层神经元只和上一层部分神经元相连(卷积计算规则)，且filter的权值对于上一层所有神经元都是一样的。对于包含两个$3*3*3$的fitler的卷积层来说，其参数数量仅有$(3*3*3+1)*2=56$个，且参数数量与上一层神经元个数无关。**与全连接神经网络相比，其参数数量大大减少了**。

#### 5.1.5 用卷积公式来表达卷积层计算

- **式4**的表达很是繁冗，最好能简化一下。就像**利用矩阵可以简化表达全连接神经网络的计算一样**，我们**利用卷积公式可以简化卷积神经网络的表达**。

- 下面我们介绍**二维卷积公式**。
    - 设矩阵 $A$、$B$，其行、列数分别为 $m_a$ 、 $n_a$ 、$m_b$ 、$n_b$，则二维卷积公式如下：
    $$
        C_{s,t}=\sum_0^{m_a-1}\sum_0^{n_a-1}         A_{m,n}B_{s-m,t-n}
    $$
    - 且 $s$ 、$t$ 满足条件：$0\le{s}\lt{m_a+m_b-1}, 0\le{t}\lt{n_a+n_b-1}$
    - 我们可以把上式写成
    $$
        C = A * B\qquad(式5)
    $$

- 如果我们按照**式5**来计算卷积，我们可以发现**矩阵A实际上是filter**，而**矩阵B是待卷积的输入**，位置关系也有所不同：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/54yj9nf0pfcizrzy0s6zvimi/image_1dqissu6clbv1e7cmqrlfppke4d.png" width="400" />
    </div>
    <br>
    - 从上图可以看到，$A$左上角的值 $a_{0,0}$ 与$B$对应区块中右下角的值 $b_{1,1}$ 相乘，而不是与左上角的 $b_{0,0}$ 相乘。因此，**数学中的卷积和卷积神经网络中的『卷积』还是有区别的**，为了避免混淆，**我们把卷积神经网络中的『卷积』操作叫做互相关(cross-correlation)操作**。
    - **卷积和互相关操作是可以转化的**。首先，我们把矩阵$A$翻转180度，然后再交换$A$和$B$的位置（即把$B$放在左边而把$A$放在右边。卷积满足交换率，这个操作不会导致结果变化），那么**卷积就变成了互相关**。
    - 如果我们不去考虑两者这么一点点的区别，我们可以把**式5**代入到式4：
    $$
        A=f(\sum_{d=0}^{D-1}X_d*W_d+w_b)\qquad(式6)
    $$
        - 其中**，$A$ 是卷积层输出的feature map**。同**式4**相比，**式6**就简单多了。然而，这种简洁写法只适合步长为1的情况。

### 5.2 Pooling层输出值的计算

- Pooling层主要的作用是**下采样**，通过去掉Feature Map中不重要的样本，**进一步减少参数数量**。Pooling的方法很多，最常用的是**Max Pooling**。Max Pooling实际上就是在n*n的样本中取最大值，作为采样后的样本值。下图是2*2 max pooling：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/h5spxeq864oqyz7reuaz9hn6/image_1dqit87pd1j2c2cc16is1h4l19na4q.png" width="400" />
    </div>
    <br>

- 除了Max Pooing之外，常用的还有**Mean Pooling——取各样本的平均值**。

- 对于深度为D的Feature Map，**各层独立做Pooling，因此Pooling后的深度仍然为D**。

### 5.3 全连接层输出值的计算

- **全连接层输出值的计算和神经网络和反向传播算法采用的全连接神经网络是一样的**，这里就不再赘述了。


----------


## 6.卷积神经网络的训练（梯度下降、反向传播、链式求导法则）

- 和全连接神经网络相比，卷积神经网络的训练要复杂一些。但**训练的原理是一样的：利用链式求导计算损失函数对每个权重的偏导数（梯度），然后根据梯度下降公式更新权重。训练算法依然是反向传播算法**。

- 我们先回忆一下**神经网络和反向传播算法**介绍的反向传播算法，整个算法分为四个步骤：
    - 前向计算每个神经元的**输出值 $a_j$**（表示网络的第 $j$ 个神经元，以下同）；
    - 反向计算每个神经元的**误差项 $\delta_j$**，$\delta_j$ 在有的文献中也叫做**敏感度(sensitivity)**。它实际上是**网络的损失函数 $E_d$ 对神经元加权输入 $net_j$ 的偏导数**，即：
    $$
        \delta_j=\frac{\partial{E_d}}{\partial{net_j}}
    $$
    - 计算每个神经元**连接权重 $w_{ji}$ 的梯度**（$w_{ji}$ 表示从神经元 $i$ 连接到神经元 $j$ 的权重），公式为：
    $$
        \frac{\partial{E_d}}{\partial{w_{ji}}}=a_i\delta_j
    $$
        - 其中，**$a_i$ 表示神经元 $i$ 的输出**。
    - 最后，**根据梯度下降法则更新每个权重**即可。

- 对于卷积神经网络，由于涉及到**局部连接**、**下采样**的等操作，影响到了第二步**误差项 $\delta$** 的具体计算方法，而**权值共享**影响了第三步**权重 $w$ 的梯度**的计算方法。

- 接下来，我们分别介绍**卷积层**和**Pooling层**的训练算法。

### 6.1 卷积层的训练

- 对于卷积层，我们先来看看上面的第二步，即如何将**误差项 $\delta$** 传递到上一层；然后再来看看第三步，即如何计算filter **每个权值 $w$ 的梯度**。

#### 6.1.1 卷积层误差项的传递

##### 最简单情况下误差项的传递

- 我们先来考虑**步长为1**、**输入的深度为1**、**filter个数为1**的最简单的情况。

- 假设输入的大小为3*3，filter大小为2*2，按步长为1卷积，我们将得到2*2的 **feature map** 。如下图所示：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/k13ro5jiddtp02uva6shh3l9/image_1dqjcrqqmepq14mj7r31dnf1vaj9.png" width="400" />
    </div>
    <br>

- 在上图中，为了描述方便，我们为每个元素都进行了编号：
    - 第 $l-1$ 层第 $j$ 行第 $i$ 列的**误差项**表示为：$\delta^{l-1}_{i,j}$
    - filter第 $m$ 行第 $n$ 列**权重**表示为：$w_{m,n}$
    - filter的**偏置项**表示为：$w_b$
    - 第 $l-1$ 层第 $i$ 行第 $j$ 列**神经元的输出**表示为：$a^{l-1}_{i,j}$
    - 第 $l-1$ 行**神经元的加权输入**表示为：$net^{l-1}_{i,j}$
    - 第 $l$ 层第 $i$ 行第 $j$ 列的**误差项**表示为：$\delta^l_{i,j}$
    - 第 $l-1$ 层的**激活函数**表示为：$f^{l-1}$

- 它们之间的关系如下：
    $$
        \begin{array}{c}
            net^l=conv(W^l, a^{l-1})+w_b  \\
            a^{l-1}_{i,j}=f^{l-1}(net^{l-1}_{i,j})
    \end{array}
    $$
    - 上式中，$net^l$、$W^l$、$a^{l-1}$ 都是数组，
    - $W^l$ 是由 $w_{m,n}$ 组成的数组
    - $conv$ 表示卷积操作

- 在这里，我们假设第 $l$ 中的每个 $\theta^l$ 值都已经算好，我们要做的是计算第 $l-1$ 层每个神经元的误差项 $\theta^{l-1}$ 。

- 根据链式求导法则：
    $$
        \begin{array}{l}
            \delta^{l-1}_{i,j}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{i,j}}}  \\
            &=\frac{\partial{E_d}}{\partial{a^{l-1}_{i,j}}}\frac{\partial{a^{l-1}_{i,j}}}{\partial{net^{l-1}_{i,j}}}
        \end{array}
    $$
    - 我们先求第一项
    $$
        \frac{\partial{E_d}}{\partial{a^{l-1}_{i,j}}}
    $$
        - 我们先来看几个特例，然后从中总结出一般性的规律：
            - 例1，计算
    $$
        \frac{\partial{E_d}}{\partial{a^{l-1}_{1,1}}}
    $$
                - **第 $l-1$ 层的 $a^{l-1}_{1,1}$ 仅与第 $l$ 层的 $net^{l}_{1,1}$ 计算有关**，即：
    $$
        net^j_{1,1}=w_{1,1}a^{l-1}_{1,1}+w_{1,2}a^{l-1}_{1,2}+w_{2,1}a^{l-1}_{2,1}+w_{2,2}a^{l-1}_{2,2}+w_b
    $$
                - 因此：
    $$
        \begin{array}{l}
            \frac{\partial{E_d}}{\partial{a^{l-1}_{1,1}}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{a^{l-1}_{1,1}}}  \\
            &=\delta^l_{1,1}w_{1,1}
        \end{array}
    $$
            - 例2，计算
    $$
        \frac{\partial{E_d}}{\partial{a^{l-1}_{1,1}}}
    $$
                - **$a^{l-1}_{1,2}$ 与 $net^l_{1,1}$ 和 $net^l_{1,2}$的计算都有关**：
                - 即：
    $$
        \begin{array}{l}
        net^j_{1,1}=w_{1,1}a^{l-1}_{1,1}+w_{1,2}a^{l-1}_{1,2}+w_{2,1}a^{l-1}_{2,1}+w_{2,2}a^{l-1}_{2,2}+w_b\\
        net^j_{1,2}=w_{1,1}a^{l-1}_{1,2}+w_{1,2}a^{l-1}_{1,3}+w_{2,1}a^{l-1}_{2,2}+w_{2,2}a^{l-1}_{2,3}+w_b
        \end{array}
    $$
                - 因此：
    $$
        \begin{array}{l}
            \frac{\partial{E_d}}{\partial{a^{l-1}_{1,2}}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{a^{l-1}_{1,2}}}+\frac{\partial{E_d}}{\partial{net^{l}_{1,2}}}\frac{\partial{net^{l}_{1,2}}}{\partial{a^{l-1}_{1,2}}}\\ \\
            &=\delta^l_{1,1}w_{1,2}+\delta^l_{1,2}w_{1,1}
        \end{array}
    $$
            - 例3，计算
    $$
        \frac{\partial{E_d}}{\partial{a^{l-1}_{2,2}}}
    $$
                - **$a^{l-1}_{2,2}$ 与以下的计算都有关：$net^{l}_{1,1}$ 、$net^{l}_{1,2}$ 、$net^{l}_{2,1}$ 、$net^{l}_{2,2}$** 
                - 即：
    $$
        \begin{array}{l}
        net^j_{1,1}=w_{1,1}a^{l-1}_{1,1}+w_{1,2}a^{l-1}_{1,2}+w_{2,1}a^{l-1}_{2,1}+w_{2,2}a^{l-1}_{2,2}+w_b\\
        net^j_{1,2}=w_{1,1}a^{l-1}_{1,2}+w_{1,2}a^{l-1}_{1,3}+w_{2,1}a^{l-1}_{2,2}+w_{2,2}a^{l-1}_{2,3}+w_b\\
        net^j_{2,1}=w_{1,1}a^{l-1}_{2,1}+w_{1,2}a^{l-1}_{2,2}+w_{2,1}a^{l-1}_{3,1}+w_{2,2}a^{l-1}_{3,2}+w_b\\
        net^j_{2,2}=w_{1,1}a^{l-1}_{2,2}+w_{1,2}a^{l-1}_{2,3}+w_{2,1}a^{l-1}_{3,2}+w_{2,2}a^{l-1}_{3,3}+w_b
        \end{array}
    $$
                - 因此：
    $$
        \begin{array}{l}
        \frac{\partial{E_d}}{\partial{a^{l-1}_{2,2}}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{a^{l-1}_{2,2}}}+\frac{\partial{E_d}}{\partial{net^{l}_{1,2}}}\frac{\partial{net^{l}_{1,2}}}{\partial{a^{l-1}_{2,2}}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,1}}}\frac{\partial{net^{l}_{2,1}}}{\partial{a^{l-1}_{2,2}}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,2}}}\frac{\partial{net^{l}_{2,2}}}{\partial{a^{l-1}_{2,2}}}\\ \\
        &=\delta^l_{1,1}w_{2,2}+\delta^l_{1,2}w_{2,1}+\delta^l_{2,1}w_{1,2}+\delta^l_{2,2}w_{1,1}
        \end{array}
    $$
        - 从上面三个例子，我们发挥一下想象力，不难发现，计算 $\frac{\partial{E_d}}{\partial{a^{l-1}}}$，相当于把第 $l$ 层的sensitive map周围补一圈0，再与180度翻转后的filter进行**cross-correlation**，就能得到想要结果，如下图所示：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/r77k9k6qd8veji401zdfctd1/image_1dqjfskmo1dh91i3b1m4517i8d7fm.png" width="400" />
    </div>
    <br>
        - 因为**卷积**相当于**将filter旋转180度的cross-correlation**，因此上图的计算可以用卷积公式完美的表达：
    $$
        \frac{\partial{E_d}}{\partial{a_l}}=\delta^l*W^l
    $$
        - 上式中的 $W^l$ 表示第 $l$ 层的filter的**权重数组**。也可以把上式的卷积展开，写成求和的形式：
    $$
        \frac{\partial{E_d}}{\partial{a^l_{i,j}}}=\sum_m\sum_n{w^l_{m,n}\delta^l_{i+m,j+n}}
    $$
    - 我们再求第二项
    $$
        \frac{\partial{a^{l-1}_{i,j}}}{\partial{net^{l-1}_{i,j}}}
    $$
        - 因为
    $$
        a^{l-1}_{i,j}=f(net^{l-1}_{i,j})
    $$
        - 所以这一项极其简单，仅求激活函数 $f$ 的导数就行了：
    $$
        \frac{\partial{a^{l-1}_{i,j}}}{\partial{net^{l-1}_{i,j}}}=f'(net^{l-1}_{i,j})
    $$
    - 将第一项和第二项组合起来，我们得到最终的公式：
    $$
        \begin{array}{l}
\delta^{l-1}_{i,j}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{i,j}}}\\
&=\frac{\partial{E_d}}{\partial{a^{l-1}_{i,j}}}\frac{\partial{a^{l-1}_{i,j}}}{\partial{net^{l-1}_{i,j}}}\\
&=\sum_m\sum_n{w^l_{m,n}\delta^l_{i+m,j+n}}f'(net^{l-1}_{i,j})\qquad(式7)
\end{array}
    $$
        - 也可以将**式7**写成卷积的形式：
    $$
        \delta^{l-1}=\delta^l*W^l\circ f'(net^{l-1})\qquad(式8)
    $$
        - 其中，符号"$\circ$"表示**element-wise product**，即将矩阵中每个对应元素相乘
        - 注意**式8**中的 $\delta^{l-1}$，$\delta^{l}$，$net^{l-1}$ 项都是矩阵

- 以上就是**步长为1**、**输入的深度为1**、**filter个数为1**的最简单的情况，卷积层误差项传递的算法。

##### 卷积步长为S时的误差传递

- 我们先来看看步长为S与步长为1的差别：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/m1dniu3a8s7wwbq5zv6rq6kf/image_1dqjgi4711uuninh19a51ti315le13.png" width="400" />
    </div>
    <br>

- 如上图，上面是步长为1时的卷积结果，下面是步长为2时的卷积结果。我们可以看出，**因为步长为2，得到的feature map跳过了步长为1时相应的部分**。因此，当我们反向计算误差项时，我们可以对步长为S的sensitivity map相应的位置进行补0，将其『还原』成步长为1时的sensitivity map，再用**式8**进行求解。

##### 输入层深度为D时的误差传递

- 如下图所示，当输入深度为$D$时，filter的深度也必须为$D$，$l-1$ 层的通道只与filter的 $d_i$ 通道的权重进行计算。因此，反向计算**误差项**时，我们可以使用**式8**，用filter的第 $d_i$ 通道权重对第 $l$ 层sensitivity map进行卷积，得到第 $l-1$ 层通道的sensitivity map。如下图：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/kqpv9vleeqi7uyoucs0x6j1v/image_1dqjgll6kq0kom61u1314l37ls1g.png" width="400" />
    </div>
    <br>

##### filter数量为N时的误差传递

- filter数量为$N$时，输出层的深度也为$N$，第 $i$ 个filter卷积产生输出层的第 $i$ 个feature map

- 第 $l-1$ 层**每个加权输入**$net^{l-1}_{d, i,j}$都同时影响了第 $l$ 层所有feature map的输出值

- 因此，反向计算**误差项**时，需要**使用全导数公式**。也就是，我们先使用第 $d$ 个filter对第 $l$ 层相应的第 $d$ 个sensitivity map进行卷积，得到一组$N$个 $l-1$ 层的偏sensitivity map。依次用每个filter做这种卷积，就得到$D$组偏sensitivity map。最后在各组之间将$N$个偏sensitivity map **按元素相加**，得到最终的$N$个 $l-1$ 层的sensitivity map：
    $$
        \delta^{l-1}=\sum_{d=0}^D\delta_d^l*W_d^l\circ f'(net^{l-1})\qquad(式9)
    $$

- 以上就是卷积层误差项传递的算法，如果还有所困惑，可以参考后面的代码实现来理解。

#### 6.1.2 卷积层filter权重梯度的计算

- 我们要在**得到第 $l$ 层sensitivity map的情况下，计算filter的权重的梯度**，由于卷积层是**权重共享**的，因此梯度的计算稍有不同。如下图所示：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/muf43abio8oslhhwhrulrbwd/image_1dqjh0rfq83io3717pemlnc3d1t.png" width="400" />
    </div>
    <br>
    - 第 $l-1$ 层的输出是：$a^l_{i,j}$
    - 第 $l$ 层filter的权重是：$w_{i,j}$
    - 第 $l$ 层的sensitivity map是：$\delta^l_{i,j}$

- 我们的任务是**计算 $w_{i,j}$ 的梯度**，即
    $$
        \frac{\partial{E_d}}{\partial{w_{i,j}}}
    $$

- 为了计算偏导数，我们需要考察权重 $w_{i,j}$ 对 $E_d$ 的影响。权重项 $w_{i,j}$ 通过影响 $net^l_{i,j}$ 的值，进而影响 $E_d$。

- 我们仍然通过几个具体的例子来看**权重项 $w\_{i,j}$ 对 $net^l\_{i,j}$ 的影响**，然后再从中总结出规律:
    - 例1，计算：
    $$
        \frac{\partial{E_d}}{\partial{w_{1,1}}}
    $$
        - 有
    $$
        net^j_{1,1}=w_{1,1}a^{l-1}_{1,1}+w_{1,2}a^{l-1}_{1,2}+w_{2,1}a^{l-1}_{2,1}+w_{2,2}a^{l-1}_{2,2}+w_b\\
        net^j_{1,2}=w_{1,1}a^{l-1}_{1,2}+w_{1,2}a^{l-1}_{1,3}+w_{2,1}a^{l-1}_{2,2}+w_{2,2}a^{l-1}_{2,3}+w_b\\
        net^j_{2,1}=w_{1,1}a^{l-1}_{2,1}+w_{1,2}a^{l-1}_{2,2}+w_{2,1}a^{l-1}_{3,1}+w_{2,2}a^{l-1}_{3,2}+w_b\\
        net^j_{2,2}=w_{1,1}a^{l-1}_{2,2}+w_{1,2}a^{l-1}_{2,3}+w_{2,1}a^{l-1}_{3,2}+w_{2,2}a^{l-1}_{3,3}+w_b
    $$
        - 从上面的公式看出，由于**权值共享**，**权值 $w_{1,1}$ 对所有的 $net^l_{i,j}$ 都有影响**。$E_d$是$net^l_{1,1}$ 、$net^l_{1,2}$ 、$net^l_{2,1}$ ...的函数，而 $net^l_{1,1}$ 、$net^l_{1,2}$ 、$net^l_{2,1}$ ... 又是 $w_{1,1}$ 的函数，根据**全导数公式**，计算$\frac{\partial{E_d}}{\partial{w_{1,1}}}$ ，就是要把每个偏导数都加起来：
    $$
        \begin{array}{l}
            \frac{\partial{E_d}}{\partial{w_{1,1}}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{w_{1,1}}}+\frac{\partial{E_d}}{\partial{net^{l}_{1,2}}}\frac{\partial{net^{l}_{1,2}}}{\partial{w_{1,1}}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,1}}}\frac{\partial{net^{l}_{2,1}}}{\partial{w_{1,1}}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,2}}}\frac{\partial{net^{l}_{2,2}}}{\partial{w_{1,1}}}\\
            &=\delta^l_{1,1}a^{l-1}_{1,1}+\delta^l_{1,2}a^{l-1}_{1,2}+\delta^l_{2,1}a^{l-1}_{2,1}+\delta^l_{2,2}a^{l-1}_{2,2}
        \end{array}
    $$
    - 例2，计算：
    $$
        \frac{\partial{E_d}}{\partial{w_{1,2}}}
    $$
        - 通过查看 $w\_{1,2}$ 与 $net^l\_{i,j}$ 的关系，我们很容易得到：
    $$
        \frac{\partial{E_d}}{\partial{w_{1,2}}}=\delta^l_{1,1}a^{l-1}_{1,2}+\delta^l_{1,2}a^{l-1}_{1,3}+\delta^l_{2,1}a^{l-1}_{2,2}+\delta^l_{2,2}a^{l-1}_{2,3}
    $$

- 实际上，**每个权重项都是类似的**，我们不一一举例了。现在，是我们再次发挥想象力的时候，我们发现计算 $ \frac{\partial{E_d}}{\partial{w_{i,j}}}$ 的规律是：
    $$
        \frac{\partial{E_d}}{\partial{w_{i,j}}}=\sum_m\sum_n\delta_{m,n}a^{l-1}_{i+m,j+n}
    $$

- 也就是**用sensitivity map作为卷积核**，**在input上进行cross-correlation**，如下图所示：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/q4fbdy8d3tl4y69kxn582bzr/image_1dqjiaq31rrvbj21m68a9019r32a.png" width="400" />
    </div>
    <br>

- 最后，我们来看一看偏置项的梯度
    $$
        \frac{\partial{E_d}}{\partial{w_b}}
    $$
    - 通过查看前面的公式，我们很容易发现：
    $$
        \begin{array}{l}
            \frac{\partial{E_d}}{\partial{w_b}}&=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{w_b}}+\frac{\partial{E_d}}{\partial{net^{l}_{1,2}}}\frac{\partial{net^{l}_{1,2}}}{\partial{w_b}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,1}}}\frac{\partial{net^{l}_{2,1}}}{\partial{w_b}}+\frac{\partial{E_d}}{\partial{net^{l}_{2,2}}}\frac{\partial{net^{l}_{2,2}}}{\partial{w_b}}\\ \\
            &=\delta^l_{1,1}+\delta^l_{1,2}+\delta^l_{2,1}+\delta^l_{2,2}\\ \\
            &=\sum_i\sum_j\delta^l_{i,j}
        \end{array}
    $$
    - 可以看出**偏置项的梯度**就是**sensitivity map所有误差项之和**。

- 对于步长为$S$的卷积层，处理方法与传递**误差项**是一样的，**首先将sensitivity map『还原』成步长为1时的sensitivity map，再用上面的方法进行计算**。

- 获得了所有的**梯度**之后，就是**根据梯度下降算法来更新每个权重**。这在前面的文章中已经反复写过，这里就不再重复了。

- 至此，我们已经解决了**卷积层的训练问题**，接下来我们看一看**Pooling层的训练**。

### 6.2 Pooling层的训练

- 无论**Max Pooling**还是**Mean Pooling**，都没有需要学习的参数。因此，**在卷积神经网络的训练中，Pooling层需要做的仅仅是将误差项传递到上一层，而没有梯度的计算**。

#### 6.2.1 Max Pooling误差项的传递

- 如下图，假设第 $l-1$ 层大小为4*4，**pooling filter** 大小为2*2，步长为2，这样，max pooling之后，第 $l$ 层大小为2*2。
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/4k04snb689sydpjmpc4qmf7g/image_1dqjj88701u7obn2v091jm314in2n.png" width="400" />
    </div>
    <br>

- **假设第 $l$ 层的误差项 $\delta$ 值都已经计算完毕，我们现在的任务是计算第 $l-1$ 层的误差值$\delta$**。

- 我们用 $net^{l-1}\_{i,j}$ 表示第 $l-1$ 层的加权输入；用 $net^l\_{i,j}$ 表示第 $l$ 层的加权输入。

- 我们先来考察一个具体的例子，然后再总结一般性的规律。对于max pooling：
    $$
        net^l_{1,1}=max(net^{l-1}_{1,1},net^{l-1}_{1,2},net^{l-1}_{2,1},net^{l-1}_{2,2})
    $$
    - 也就是说，**只有区块中最大的 $net^{l-1}\_{i,j}$ 才会对 $net^l\_{i,j}$ 的值产生影响**。我们假设最大的值是 $net^{l-1}\_{1,1}$ ，则上式相当于：
    $$
        net^l_{1,1}=net^{l-1}_{1,1}
    $$
    - 那么，我们不难求得下面几个偏导数：
    $$
        \begin{array}{c}
            \frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{1,1}}}=1\\ \\
            \frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{1,2}}}=0\\ \\
            \frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{2,1}}}=0\\ \\
            \frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{2,2}}}=0
        \end{array}
    $$
    - 因此：
    $$
        \begin{array}{l}
            \delta^{l-1}_{1,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{1,1}}}\\ \\
            &=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{1,1}}}\\ \\
            &=\delta^{l}_{1,1}\\
        \end{array}
    $$
    - 而：
    $$
        \begin{array}{c}
            \delta^{l-1}_{1,2}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{1,2}}}=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{1,2}}}=0\\ \\
            \delta^{l-1}_{2,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{2,1}}}=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{2,1}}}=0\\ \\
            \delta^{l-1}_{1,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{2,2}}}=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{2,2}}}=0\\ \\
        \end{array}
    $$
- 现在，我们发现了规律：
    - 对于max pooling，**下一层的误差项的值会原封不动的传递到上一层对应区块中的最大值所对应的神经元**，而**其他神经元的误差项的值都是0**。如下图所示(假设 $a^{l-1}_{1,1}$ 、 $a^{l-1}_{1,4}$ 、 $a^{l-1}\_{4,1}$ 、 $a^{l-1}\_{4,4}$ 为所在区块中的最大输出值)：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/sd6v9jovdqhxh02q0owt3wma/image_1dqjjqohckj6173f1a2t1s7utf334.png" width="400" />
    </div>
    <br>

#### 6.2.2 Mean Pooling误差项的传递

- 我们还是用前面屡试不爽的套路，先研究一个特殊的情形，再扩展为一般规律。
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/6ymthqphd8tv6vh6lqzw3zze/image_1dqjju40a5scv1815gnc3o1j6q3h.png" width="400" />
    </div>
    <br>

- 如上图，我们先来考虑计算 $\delta^{l-1}\_{1,1}$ 。我们先来看看 $net^{l-1}\_{1,1}$ 如何影响 $net^l\_{1,1}$ ：
    $$
        net^j_{1,1}=\frac{1}{4}(net^{l-1}_{1,1}+net^{l-1}_{1,2}+net^{l-1}_{2,1}+net^{l-1}_{2,2})
    $$

- 根据上式，我们一眼就能看出来：
    $$
        \frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{1,1}}}=\frac{1}{4}\\ \\
        \frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{1,2}}}=\frac{1}{4}\\ \\
        \frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{2,1}}}=\frac{1}{4}\\ \\
        \frac{\partial{net^l_{1,1}}}{\partial{net^{l-1}_{2,2}}}=\frac{1}{4}\\ \\
    $$

- 所以，根据链式求导法则，我们不难算出：
    $$
        \begin{array}{l}
            \delta^{l-1}_{1,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{1,1}}}\\ \\
            &=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{1,1}}}\\ \\
            &=\frac{1}{4}\delta^{l}_{1,1}\\ \\
        \end{array}
    $$

- 同样，我们可以算出 $\delta^{l-1}\_{1,2}$ 、 $\delta^{l-1}\_{2,1}$ 、 $\delta^{l-1}\_{2,2}$ ：
    $$
        \begin{array}{c}
            \delta^{l-1}_{1,2}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{1,2}}}=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{1,2}}}=\frac{1}{4}\delta^{l}_{1,1}\\ \\
            \delta^{l-1}_{2,1}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{2,1}}}=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{2,1}}}=\frac{1}{4}\delta^{l}_{1,1}\\ \\
            \delta^{l-1}_{2,2}&=\frac{\partial{E_d}}{\partial{net^{l-1}_{2,2}}}=\frac{\partial{E_d}}{\partial{net^{l}_{1,1}}}\frac{\partial{net^{l}_{1,1}}}{\partial{net^{l-1}_{2,2}}}=\frac{1}{4}\delta^{l}_{1,1}\\
        \end{array}
    $$

- 现在，我们发现了规律：
    - 对**于mean pooling**，下一层的**误差项**的值会**平均分配**到上一层对应区块中的所有神经元。如下图所示：
    <div align="middle">
    <img src="http://static.zybuluo.com/georay53/gsiiwnpwxb3xuwrp6e8ujex3/image_1dqjkd5d91kmd4jv1pkg1fb31u353u.png" width="400" />
    </div>
    <br>

- 上面这个算法可以表达为高大上的**克罗内克积(Kronecker product)**的形式，有兴趣可以研究一下：
    $$
        \delta^{l-1} = \delta^l\otimes(\frac{1}{n^2})_{n\times n}
    $$

    - 其中，$n$ 是pooling层filter的大小，$\delta^{l-1}$、$\delta^l$ 都是矩阵。

- 至此，我们已经把**卷积层**、**Pooling层**的训练算法介绍完毕，加上上一篇文章讲的**全连接层**训练算法，已经具备了编写**卷积神经网络**代码所需要的知识。
