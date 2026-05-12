from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# =====================
# 1. 路径设置
# =====================
MODEL_PATH = Path(r"D:\desktop\concrete_crack_detection\models\crack_cnn.pth")

# 这里换成你想测试的图片路径
IMAGE_PATH = Path(r"D:\desktop\concrete_crack_detection\data\test\crack\7048-2.jpg")

IMAGE_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用设备：", device)


# =====================
# 2. 定义 CNN 模型
# 必须和 train.py 里的模型结构一样
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
# 3. 图像预处理
# =====================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# =====================
# 4. 加载模型
# =====================
model = CrackCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("模型加载成功")


# =====================
# 5. 读取并预测图片
# =====================
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()

classes = ["crack", "non_crack"]

predicted_class = classes[predicted_index]
crack_prob = probabilities[0][0].item()
non_crack_prob = probabilities[0][1].item()

print("测试图片：", IMAGE_PATH)
print("预测结果：", predicted_class)
print(f"crack 概率：{crack_prob:.4f}")
print(f"non_crack 概率：{non_crack_prob:.4f}")