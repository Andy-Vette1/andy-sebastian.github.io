# Week 01 笔记：机器学习与数学基础回顾

本笔记基于 FIT3181/5215 课程 Week 01 内容整理。

## 一、线性代数基础 (Linear Algebra Revision)

### 1. 向量（Vector）
**概念**
一个 $n$ 维向量写作：
$$\mathbf{x} = (x\_1, x\_2, \dots, x\_n)^\top \in \mathbb{R}^n$$

**常见操作**：
* **转置 (Transpose)**：$\mathbf{x}^\top$ 把列向量变成行向量。
* **向量加法**：$(\mathbf{x} + \mathbf{y})\_i = x\_i + y\_i$。
* **内积 (Inner product)**：
    $$\mathbf{x}^\top \mathbf{y} = \sum\_{i=1}^n x\_i y\_i$$

### 2. 向量范数与长度 (Norms)
**概念**
$p$-范数定义为：
$$\|\mathbf{x}\|\_p = \left( \sum\_{i=1}^n |x\_i|^p \right)^{1/p}, \quad p>0$$

* $p=1$ (L1 范数)：$\|\mathbf{x}\|\_1 = \sum\_i |x\_i|$ (绝对值之和)
* $p=2$ (L2 范数)：$\|\mathbf{x}\|\_2 = \sqrt{\sum\_i x\_i^2}$，即向量的欧几里得长度。

### 3. 向量间距离与相似度
**欧氏距离 (Euclidean distance)**：
$$d\_{\text{euclid}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x}-\mathbf{y}\|\_2 = \sqrt{\sum\_{i=1}^n (x\_i - y\_i)^2}$$

**余弦相似度 (Cosine similarity)**：
衡量两个向量方向的相似程度，范围在 $[-1, 1]$ 之间：
$$\cos(\theta) = \cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|\_2 \|\mathbf{y}\|\_2}$$

**余弦距离**：
$$d\_{\text{cos}}(\mathbf{x}, \mathbf{y}) = 1 - \cos(\mathbf{x}, \mathbf{y})$$

### 4. 矩阵与张量 (Matrix & Tensor)
* **矩阵 (Matrix)**：二维数组，例如 $A \in \mathbb{R}^{m \times n}$。
    * 矩阵乘法：若 $A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}$，则 $C = AB \in \mathbb{R}^{m \times p}$。
    * 计算公式：$c\_{ij} = \sum\_{k=1}^n a\_{ik} b\_{kj}$。
* **张量 (Tensor)**：多维数组的统称。
    * 0D: 标量 (Scalar)
    * 1D: 向量 (Vector)
    * 2D: 矩阵 (Matrix)
    * 3D+: 张量 (如 RGB 图像 $H \times W \times 3$)

---

## 二、信息论基础 (Information Theory)

### 1. 熵 (Entropy)
对于离散分布 $p = (p\_1, \dots, p\_d)$，香农熵定义为：
$$H(p) = -\sum\_{i=1}^d p\_i \log p\_i$$
*直观理解：熵越大，分布越均匀，不确定性越高。*

### 2. KL 散度 (Kullback-Leibler Divergence)
衡量两个分布 $p$ (真实分布) 和 $q$ (预测分布) 之间的差异：
$$\mathrm{KL}(p \| q) = \sum\_{i=1}^d p\_i \log \frac{p\_i}{q\_i}$$
* 性质：$\mathrm{KL}(p \| q) \ge 0$，当且仅当 $p=q$ 时为 0。

### 3. 交叉熵 (Cross-Entropy)
深度学习中最常用的分类损失函数：
$$\mathrm{CE}(p, q) = -\sum\_{i=1}^d p\_i \log q\_i$$
关系公式：$\mathrm{CE}(p, q) = \mathrm{KL}(p \| q) + H(p)$。

---

## 三、机器学习回顾 (Machine Learning Revisited)

### 1. 机器学习三要素
1.  **数据 (Data)**：输入 $\mathbf{x} \in \mathbb{R}^d$，标签 $y$。
2.  **模型 (Model)**：函数 $f: \mathcal{X} \to \mathcal{Y}$。
3.  **评估 (Assessment)**：损失函数与准确率。

### 2. 判别式学习 (Discriminative ML)
* **Logits**: 模型输出的原始分数 $h(\mathbf{x}) = (h\_1, \dots, h\_M)$。
* **Softmax**: 将 logits 转化为概率分布：
    $$p\_m(\mathbf{x}) = \frac{\exp(h\_m(\mathbf{x}))}{\sum\_{i=1}^M \exp(h\_i(\mathbf{x}))}$$
* **预测**: $\hat{y} = \arg\max\_m p\_m(\mathbf{x})$。

### 3. 损失函数 (Loss Functions)
* **分类 (Classification)**: 使用交叉熵损失 (Cross-Entropy)。
    $$L = -\frac{1}{N} \sum\_{i=1}^N \log p\_{y\_i}(\mathbf{x}\_i)$$
* **回归 (Regression)**: 使用 L2 损失 (MSE) 或 L1 损失。
    $$\ell\_{L2} = \frac{1}{2}(y - \hat{y})^2$$

---

## 四、逻辑回归 (Logistic Regression)

逻辑回归可以看作是最简单的单层前馈神经网络。

* **模型结构**:
    $$h(\mathbf{x}) = \mathbf{x}W + \mathbf{b}$$
    $$p(\mathbf{x}) = \text{softmax}(h(\mathbf{x}))$$
* **参数**: 权重 $W \in \mathbb{R}^{d \times 2}$，偏置 $b \in \mathbb{R}^{1 \times 2}$ (以二分类为例)。
* **训练目标**: 最小化所有样本的平均交叉熵损失。
    $$(W^\*, b^\*) = \arg\min\_{W,b} L(D; W, b)$$

**总结**: 逻辑回归通过线性变换 + Softmax，将特征映射到类别概率，利用交叉熵衡量预测分布与 One-hot 真实标签的差异。