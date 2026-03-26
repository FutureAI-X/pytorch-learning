import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork
from data_loader import get_test_data

# 数据集加载
test_data = get_test_data()
test_data_len = len(test_data)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 模型加载
current_folder = current_file = Path(__file__).resolve().parent
model = NeuralNetwork()
model.load_state_dict(torch.load(f'{current_folder}/model_weights.pth', weights_only=True))
model.eval()

# 执行测试
figure = plt.figure(figsize=(8, 8))
for i in range(1, 10):
    # 从数据集中随机获取一条数据
    sample_idx = torch.randint(test_data_len, size=(1,)).item()
    img, label = test_data[sample_idx]

    # 执行预测并输出
    logits = model(img)
    pred = logits.argmax(1).item()

    # 绘制图片
    figure.add_subplot(3, 3, i)
    plt.title(f"{labels_map[label]}, {labels_map[pred]}")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()