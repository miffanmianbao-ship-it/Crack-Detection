from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =====================
# 1. 路径设置
# =====================
DATA_DIR = Path(r"D:\desktop\concrete_crack_detection\data")
MODEL_DIR = Path(r"D:\desktop\concrete_crack_detection\models")
MODEL_DIR.mkdir(exist_ok=True)

# =====================
# 2. 参数设置
# =====================
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.001
IMAGE_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用设备：", device)

# =====================
# 3. 图像预处理
# =====================
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_val = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =====================
# 4. 读取数据
# =====================
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform_train)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("类别对应关系：", train_dataset.class_to_idx)
print("训练集图片数量：", len(train_dataset))
print("验证集图片数量：", len(val_dataset))


# =====================
# 5. 定义 CNN 模型
# =====================
class CrackCNN(nn.Module):
    def __init__(self):
        super(CrackCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CrackCNN().to(device)

# =====================
# 6. 损失函数和优化器
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_acc = 0.0

# =====================
# 7. 开始训练
# =====================
for epoch in range(EPOCHS):
    print(f"\n第 {epoch + 1}/{EPOCHS} 轮训练")

    # 训练阶段
    model.train()
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = train_correct / train_total

    # 验证阶段
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total

    print(f"训练准确率：{train_acc:.4f}")
    print(f"验证准确率：{val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_DIR / "crack_cnn.pth")
        print("模型已保存。")

print("\n训练完成！")
print("最佳验证准确率：", best_val_acc)
print("模型保存位置：", MODEL_DIR / "crack_cnn.pth")