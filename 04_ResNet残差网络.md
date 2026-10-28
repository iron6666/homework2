# 04 ResNet 残差网络

> 深度学习课程 · 总结与课后作业
> 
> 日期：2026-10-27

---

## 本课核心要点

- **网络退化**：深网络训练误差反而更高——不是过拟合，而是让非线性层拟合恒等映射太困难。

- **残差学习**：不学 $H(x)$，学 $F(x) = H(x) - x$，输出 $H(x) = F(x) + x$。恒等映射成为"默认选项"。

- **梯度高速公路**：$\frac{\partial x_L}{\partial x_l} = 1 + \frac{\partial}{\partial x_l} \sum F$。"+1" 项保证梯度无衰减地回传。

- **两种构建块**：基础块用于 ResNet-18/34；瓶颈块用于 ResNet-50/101/152，参数量更少。

- **维度匹配**：当 $x$ 和 $F(x)$ 维度不匹配时，用 $1 \times 1$ 卷积做 projection shortcut。

---

## 1. 理论推导

### 题目

请写出 ResNet 中权重层 $W_l$ 的梯度 $\frac{\partial L}{\partial W_l}$ 的表达式。它与深层梯度 $\frac{\partial L}{\partial x_{l+1}}$ 是如何关联的？（提示：利用 $x_{l+1} = x_l + F(x_l, W_l)$ 进行链式求导）

### 解答

#### 1.1 前向传播公式

ResNet 的基本残差块可以表示为：

$$x_{l+1} = x_l + F(x_l, W_l)$$

其中：
- $x_l$ 是第 $l$ 层的输入
- $F(x_l, W_l)$ 是残差函数，通常由卷积、BN、ReLU 等操作组成
- $x_{l+1}$ 是第 $l$ 层的输出（即下一层的输入）

#### 1.2 求 $\frac{\partial L}{\partial W_l}$

根据链式法则：

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial x_{l+1}}{\partial W_l}$$

由 $x_{l+1} = x_l + F(x_l, W_l)$，可得：

$$\frac{\partial x_{l+1}}{\partial W_l} = \frac{\partial F(x_l, W_l)}{\partial W_l}$$

因此：

$$\boxed{\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial F(x_l, W_l)}{\partial W_l}}$$

#### 1.3 求深层梯度 $\frac{\partial L}{\partial x_l}$

同样利用链式法则，$L$ 对 $x_l$ 的梯度来自两条路径：

1. **直接路径（shortcut）**：$x_l$ 直接传递到 $x_{l+1}$
2. **间接路径（残差分支）**：$x_l$ 经过 $F(x_l, W_l)$ 再传递到 $x_{l+1}$$

因此：

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \left(1 + \frac{\partial F(x_l, W_l)}{\partial x_l}\right)$$

即：

$$\boxed{\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} + \frac{\partial L}{\partial x_{l+1}} \cdot \frac{\partial F(x_l, W_l)}{\partial x_l}}$$

#### 1.4 关联分析

从上面的推导可以看出：

1. **权重梯度与深层梯度的关系**：

   $\frac{\partial L}{\partial W_l}$ 直接依赖于 $\frac{\partial L}{\partial x_{l+1}}$，即第 $l$ 层的权重梯度由更深层的梯度回传得到。没有深层梯度，浅层权重就无法更新。

2. **梯度高速公路**：

   在 $\frac{\partial L}{\partial x_l}$ 的表达式中，"+1" 项来自恒等映射（shortcut connection）。这意味着即使 $\frac{\partial F(x_l, W_l)}{\partial x_l}$ 很小（梯度消失），梯度仍能通过 "+1" 项无损地传回浅层。这就是 ResNet 能解决深层网络梯度消失问题的关键。

3. **跨层梯度传播**：

   对于任意两层 $l < L$，有：

   $$\frac{\partial x_L}{\partial x_l} = \prod_{i=l}^{L-1} \left(1 + \frac{\partial F(x_i, W_i)}{\partial x_i}\right)$$

   展开后包含 $2^{L-l}$ 项，其中最重要的一项就是恒等映射项 "1"，它保证了梯度可以无衰减地从深层传回浅层。

---

## 2. 思考与对比

### 2.a Highway Network 与 ResNet 在快捷连接实现上的异同

#### Highway Network 简介

Highway Network（Srivastava et al., 2015）几乎与 ResNet（He et al., 2015）同时提出，也是一种通过"门控"快捷连接来训练深度网络的结构。

#### 相同点

| 方面 | 说明 |
|------|------|
| **核心思想** | 都通过引入"捷径"（shortcut/path）来解决深层网络梯度消失/退化问题 |
| **恒等映射** | 都允许信息直接跨层流动，保留恒等映射的能力 |
| **端到端训练** | 都可以使用标准反向传播进行端到端训练 |
| **深度网络** | 都成功训练了上百层的网络 |

#### 不同点

| 方面 | Highway Network | ResNet |
|------|-----------------|--------|
| **门控机制** | 有门控：$y = H(x, W_H) \cdot T(x, W_T) + x \cdot C(x, W_C)$ | 无门控：$y = F(x, W) + x$ |
| **Transform Gate** | $T \in [0, 1]$，可学习，控制信息流 | 不存在，恒等映射始终全开 |
| **Carry Gate** | $C = 1 - T$，控制原始信息通过量 | 不存在，shortcut 始终为 1 |
| **信息流动** | 信息流动是"软"的，由门控调节 | 信息流动是"硬"的，恒等映射固定 |
| **参数量** | 更多（需要额外的门控参数） | 更少（无需门控参数） |
| **计算复杂度** | 更高（需要计算门控值） | 更低 |
| **初始化** | 需要偏置初始化使 $T \approx 0$（即初始时偏向恒等映射） | 无需特殊初始化，恒等映射天然存在 |
| **训练稳定性** | 门控可能饱和，导致训练不稳定 | 更稳定，梯度高速公路更直接 |

#### 总结

Highway Network 可以看作是 ResNet 的"一般化"形式：
- 当 $T = 1, C = 0$ 时，Highway 退化为普通网络（无 shortcut）
- 当 $T = 0, C = 1$ 时，Highway 退化为 ResNet（纯恒等映射）

ResNet 的成功表明：**在图像识别等任务中，简单的恒等映射（无门控）已经足够好**，而且更简单、参数更少、训练更稳定。这也印证了"深度残差学习"论文中的观点：让网络学习残差比学习完整映射更容易。

---

### 2.b DenseNet 与 ResNet 的连接方式对比

#### DenseNet 简介

DenseNet（Dense Convolutional Network, Huang et al., 2017）在 ResNet 之后提出，进一步扩展了 shortcut connection 的思想。

#### 本质区别

| 方面 | ResNet | DenseNet |
|------|--------|----------|
| **连接方式** | 逐元素相加（additive）：$x_{l+1} = x_l + F(x_l)$ | 特征拼接（concatenative）：$x_{l+1} = [x_0, x_1, ..., x_l, F(x_l)]$ |
| **连接范围** | 只与前一层相连（跳跃一层） | 与之前所有层相连（密集连接） |
| **信息流** | 信息通过加法融合，可能丢失部分信息 | 信息通过拼接保留，所有层的特征都被保留 |
| **特征复用** | 有限（只复用前一层的输出） | 充分（每层都能访问之前所有层的特征） |
| **通道数** | 固定（与输入相同） | 逐层增加（$k$ 为 growth rate） |

#### DenseNet 的优点

1. **更强的特征复用**：每层都能直接访问之前所有层的特征图，鼓励特征重用，减少冗余计算。

2. **缓解梯度消失**：梯度可以通过多条路径（不止一条 shortcut）回传，进一步改善梯度流动。

3. **参数效率更高**：由于特征复用，DenseNet 可以用更少的参数达到与 ResNet 相当的性能。growth rate $k$ 通常较小（如 32）。

4. **正则化效果**：密集连接具有隐式的正则化效果，在小数据集上表现更好，过拟合风险更低。

5. **特征多样性**：拼接操作保留了不同层的特征，增加了特征的多样性。

#### DenseNet 的缺点

1. **内存消耗大**：由于要保存所有中间层的特征图用于拼接，训练时需要大量内存，尤其是深层网络。

2. **计算效率问题**：虽然参数少，但由于特征图拼接后通道数增加，实际计算量（FLOPs）可能并不比 ResNet 少。

3. **实现复杂**：DenseNet 的实现比 ResNet 更复杂，尤其是需要管理多个层的连接关系。

4. **推理速度**：由于密集连接，DenseNet 的推理速度通常比 ResNet 慢，不利于实时应用。

5. **过拟合风险（大数据集）**：在超大规模数据集上，DenseNet 的正则化优势可能变成劣势，导致欠拟合。

#### 总结

| 维度 | ResNet | DenseNet |
|------|--------|----------|
| **核心操作** | 加法（$+$） | 拼接（$[\cdot, \cdot]$） |
| **连接密度** | 稀疏（跳层） | 密集（全连接） |
| **参数效率** | 高 | 更高 |
| **内存效率** | 高 | 低 |
| **推理速度** | 快 | 慢 |
| **适用场景** | 通用，尤其是实时应用 | 小数据集、需要强正则化的场景 |

ResNet 和 DenseNet 都证明了：**让梯度/信息更容易地在层间流动是训练深层网络的关键**。ResNet 通过加法实现简洁高效的捷径，DenseNet 通过拼接实现最大化的特征复用。两者各有优劣，适用于不同的场景。

---

## 参考资料

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *CVPR 2016*.
2. Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Highway Networks. *ICML 2015 Deep Learning Workshop*.
3. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. *CVPR 2017*.
