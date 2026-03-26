import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from neural_network import NeuralNetwork
from data_loader import get_training_dataloader,get_test_dataloader

from pathlib import Path

# Step1 获取训练&测试数据加载器
batch_size = 64
train_dataloader = get_training_dataloader(batch_size=batch_size)
test_dataloader = get_test_dataloader(batch_size=batch_size)

# Step2 实例化神经网络
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
model = NeuralNetwork().to(device)


def train_loop(dataloader, model, loss_fn, optimizer):
    """训练循环"""
    # 1. 获取数据集总大小（用于后续打印日志使用）
    size = len(dataloader.dataset)
    # 2. 将模型设置为训练模式: 对于Batch Normalization和Dropout Layers 非常重要。在此情境下并非必须，但为了遵循最佳实践而添加
    model.train()
    # 3. 遍历数据集
    for batch, (X, y) in enumerate(dataloader):
        # batch 表示当前属于第几批，从0开始，每次循环会加1
        # X.shape: torch.Size([64, 1, 28, 28])
        # y.shape: torch.Size([64])

        X = X.to(device)
        y = y.to(device)

        # 执行一次前向传播
        pred = model(X)

        # 计算损失
        loss = loss_fn(pred, y)

        # 反向传播(含梯度计算)
        loss.backward()
        # 模型参数更新(梯度应用)
        optimizer.step()
        # 梯度清零(以便下一次重新计算梯度)
        optimizer.zero_grad()

        # 日志打印
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}, 训练进度: [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # 1. 变量定义，用于打印日志
    # 数据集总大小
    size = len(dataloader.dataset)
    # 批次数量
    num_batches = len(dataloader)
    # 指标
    test_loss, correct = 0, 0

    # 2. 将模型设置为训练模式: 对于Batch Normalization和Dropout Layers 非常重要。在此情境下并非必须，但为了遵循最佳实践而添加
    model.eval()
    

    # 3. 使用 torch.no_grad() 评估模型可确保在测试模式下不计算梯度，同时也能减少对 requires_grad=True 的张量进行不必要的梯度计算和内存消耗。
    with torch.no_grad():
        # 遍历验证数据集
        for X, y in dataloader:
            # X.shape: torch.Size([64, 1, 28, 28])
            # y.shape: torch.Size([64])

            X = X.to(device)
            y = y.to(device)

            # 执行一次前向传播, pred.shape: torch.Size([64, 10])
            pred = model(X)
            # 计算损失并累加
            test_loss += loss_fn(pred, y).item()
            # 计算准确率
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # 4. 计算验证集准确率与平均损失
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练伦次(每一轮都将遍历数据集一次)
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

# 训练完保存模型参数
current_folder = current_file = Path(__file__).resolve().parent
torch.save(model.state_dict(), f'{current_folder}/model_weights.pth')

print("Done!")