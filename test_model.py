from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt


# =====================
# 1. 路径设置
# =====================
DATA_DIR = Path(r"D:\desktop\concrete_crack_detection\data")
MODEL_PATH = Path(r"D:\desktop\concrete_crack_detection\models\crack_cnn.pth")
RESULT_DIR = Path(r"D:\desktop\concrete_crack_detection\results")
RESULT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用设备：", device)


# =====================
# 2. 图像预处理
# =====================
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# =====================
# 3. 读取测试集
# =====================
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("类别对应关系：", test_dataset.class_to_idx)
print("测试集图片数量：", len(test_dataset))


# =====================
# 4. 定义和训练时一样的 CNN 模型
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


# =====================
# 5. 加载训练好的模型
# =====================
model = CrackCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("模型加载成功：", MODEL_PATH)


# =====================
# 6. 测试模型
# =====================
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())


# =====================
# 7. 输出结果
# =====================
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n测试集准确率：", acc)
print("\n混淆矩阵：")
print(cm)

print("\n分类报告：")
print(classification_report(
    all_labels,
    all_preds,
    target_names=test_dataset.classes
))


# =====================
# 8. 保存混淆矩阵图片
# =====================
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], test_dataset.classes)
plt.yticks([0, 1], test_dataset.classes)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_matrix.png")
plt.show()

print("\n混淆矩阵图片已保存到：", RESULT_DIR / "confusion_matrix.png")