# 手写数字识别 (MNIST CNN)

基于 PyTorch 的卷积神经网络 (CNN) 实现手写数字识别，适合深度学习新手入门学习。

## 项目效果

- **验证集准确率**: 99.55%
- **训练时间**: 约 1 小时 37 分钟 (CPU)
- **数据规模**: 训练集 33,600 张 / 验证集 8,400 张

---

## 新手可以学习到的内容

### 1. Python 库的基本使用

| 库 | 作用 | 学习重点 |
|---|---|---|
| `torch` | PyTorch 核心库，张量计算、自动求导 | 张量创建、GPU/CPU 切换 |
| `pandas` | 数据处理，读取 CSV 表格 | `read_csv()`, 数据切片 |
| `numpy` | 数值计算，数组操作 | `reshape()`, 数据类型转换 |
| `matplotlib` | 数据可视化 | `imshow()` 画图、`subplot()` 多图 |

```python
# 示例：读取 CSV 数据
import pandas as pd
train_df = pd.read_csv('train.csv')
print(train_df.shape)  # (42000, 785) - 42000张图，每张784个像素

# 示例：将一维像素数组变成 28x28 图片
pixels = sample.drop('label').values.reshape(28, 28)
plt.imshow(pixels, cmap='gray')  # 用灰度图显示
```

---

### 2. PyTorch 神经网络基础

#### 2.1 Dataset 与 DataLoader (数据加载)

**为什么需要？** 把数据想象成"打饭阿姨"，批量把数据喂给模型。

```python
from torch.utils.data import Dataset, DataLoader

class DigitDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test

    def __len__(self):  # 返回数据集大小
        return len(self.df)

    def __getitem__(self, idx):  # 根据索引获取数据
        image = self.df.iloc[idx, 1:].values.reshape(28, 28, 1).astype(np.float32)
        label = self.df.iloc[idx, 0]  # 第一列是标签
        return image, label

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
```

**新手知识点：**
- `Dataset`: 自定义数据集类，需要实现 `__len__` 和 `__getitem__`
- `DataLoader`: 批量加载数据，`batch_size` 是一次喂多少张图，`shuffle=True` 打乱顺序
- `__getitem__` 返回什么：训练时返回 (图片, 标签)，测试时只返回图片

---

#### 2.2 CNN 网络结构搭建

```python
import torch.nn as nn

class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类初始化

        # 卷积层：提取图片特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 输入1通道(灰度)，输出32通道
            nn.BatchNorm2d(32),               # 批归一化，加速训练
            nn.ReLU(),                         # 激活函数
            nn.MaxPool2d(2),                   # 池化，缩小图片
            nn.Dropout2d(0.25)                 # 随机丢弃，防止过拟合
        )

        # 全连接层：分类
        self.fc = nn.Sequential(
            nn.Flatten(),                      # 展平数据
            nn.Linear(128 * 3 * 3, 256),      # 全连接层
            nn.Linear(256, 10)                # 输出10类（0-9）
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc(x)
        return x
```

**新手知识点：**
- `nn.Module`: 所有神经网络模型的基类
- `super().__init__()`: 初始化父类，必须调用
- `nn.Conv2d(输入通道, 输出通道, 卷积核大小)`: 卷积操作，提取特征
- `nn.BatchNorm2d`: 批归一化，让训练更稳定
- `nn.ReLU()`: 激活函数，引入非线性
- `nn.MaxPool2d(2)`: 池化，把 28x28 变成 14x14，减少计算量
- `nn.Dropout`: 训练时随机关闭一些神经元，防止死记硬背（过拟合）
- `nn.Linear`: 全连接层，做最终分类

---

### 3. 数据预处理与增强

#### 3.1 图像变换 (transforms)

```python
from torchvision import transforms

# 训练集：做数据增强，提高泛化能力
train_transform = transforms.Compose([
    transforms.ToTensor(),                    # 转为 PyTorch 张量
    transforms.RandomRotation(20),            # 随机旋转 ±20 度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

# 测试集：只做归一化，不增强
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

**为什么需要数据增强？**
- 让模型见过更多"变化"，考试时更聪明
- 旋转、平移、缩放都是为了让模型不只是记住标准写法

---

### 4. 模型训练过程

#### 4.1 损失函数与优化器

```python
# 损失函数：交叉熵，用于分类任务
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 优化器：Adam，自动调节学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 学习率调度器：自动调整学习率
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=30, steps_per_epoch=len(train_loader)
)
```

**新手知识点：**
- `CrossEntropyLoss`: 多分类常用损失函数
- `Adam`: 最常用的优化器，结合了动量和自适应学习率
- `label_smoothing`: 标签平滑，让模型不要太自信，防止过拟合
- `OneCycleLR`: 学习率先增后降，训练更稳定

---

#### 4.2 训练循环

```python
def model_train():
    model.train()  # 设置为训练模式（开启 Dropout）

    for epoch in range(n_epochs):
        for images, labels in train_loader:
            # 1. 数据搬到 GPU/CPU
            images, labels = images.to(device), labels.to(device)

            # 2. 清空梯度
            optimizer.zero_grad()

            # 3. 前向传播
            outputs = model(images)

            # 4. 计算损失
            loss = criterion(outputs, labels)

            # 5. 反向传播
            loss.backward()

            # 6. 更新参数
            optimizer.step()
            scheduler.step()

        # 7. 验证集评估
        model.eval()  # 评估模式（关闭 Dropout）
        correct = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                outputs = model(val_images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == val_labels).sum().item()
```

**训练流程（背下来！）：**
1. `model.train()` - 训练模式
2. `optimizer.zero_grad()` - 清空梯度
3. `outputs = model(images)` - 前向传播
4. `loss = criterion(outputs, labels)` - 计算损失
5. `loss.backward()` - 反向传播
6. `optimizer.step()` - 更新参数
7. 验证时用 `model.eval()` 和 `torch.no_grad()`

---

### 5. GPU 加速（可选）

```python
# 检查是否有 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 把模型和数据都搬到 GPU 上
model = model.to(device)
images = images.to(device)
```

**新手建议：** MNIST 数据量小，CPU 就够用。GPU 主要用于大数据集。

---

### 6. 模型预测与提交

```python
def generate_submission():
    model.eval()  # 评估模式
    all_preds = []

    with torch.no_grad():  # 不计算梯度，节省显存
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 取概率最大的类别
            all_preds.extend(predicted.cpu().numpy())

    # 生成提交文件
    submission_df = pd.DataFrame({
        'ImageId': range(1, len(all_preds) + 1),
        'Label': all_preds
    })
    submission_df.to_csv('submission.csv', index=False)
```

---

## 项目文件结构

```
D:\Digit Recognizer\
├── README.md                    # 本说明文件
├── mnist_cnn.pth               # 训练好的模型权重
├── mnist-first-cnn.ipynb       # 完整训练代码 (Jupyter Notebook)
├── data\
│   ├── train.csv               # 训练数据 (42000行)
│   ├── test.csv                # 测试数据 (28000行)
│   └── sample_submission.csv   # 提交样例
└── out\
    └── submission.csv           # 预测结果提交文件
```

---

## 如何运行

### 方式一：Jupyter Notebook

1. 安装依赖：
```bash
pip install torch pandas numpy matplotlib tqdm torchvision
```

2. 打开 `mnist-first-cnn.ipynb`，依次运行每个单元格

### 方式二：Python 脚本

将 notebook 内容保存为 `.py` 文件后运行

---

## 核心学习要点总结

| 概念 | 作用 | 新手必知 |
|---|---|---|
| Dataset/DataLoader | 批量加载数据 | batch_size、shuffle |
| Conv2d | 卷积提取特征 | 输入输出通道数 |
| BatchNorm | 加速训练稳定 | 训练时开启 |
| Dropout | 防止过拟合 | 训练开启，评估关闭 |
| ReLU | 激活函数 | 引入非线性 |
| CrossEntropyLoss | 分类损失 | 多分类用这个 |
| Adam | 优化器 | 最常用 |
| forward/backward | 前向/反向传播 | 训练核心步骤 |
| model.train/eval | 切换模式 | 训练/评估不同行为 |

---

## 扩展学习建议

1. **尝试修改网络**：增加/减少卷积层，看看效果变化
2. **调整超参数**：修改 `batch_size`、`learning_rate`、`n_epochs`
3. **尝试其他优化器**：SGD、RMSprop
4. **增加更多数据增强**：颜色变化、噪声等
5. **保存和加载模型**：
```python
torch.save(model.state_dict(), 'model.pth')  # 保存
model.load_state_dict(torch.load('model.pth'))  # 加载
```

---

## 参考

- MNIST 数据集：LeCun 大神的手写数字数据集，深度学习入门必练
- PyTorch 官方文档：https://pytorch.org/docs/
