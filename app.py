from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st


# =====================
# 1. 路径设置
# =====================
MODEL_PATH = Path("models/crack_cnn.pth")
IMAGE_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# 2. 定义 CNN 模型
# 必须和 train.py 里的模型结构一致
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
# 3. 加载模型
# =====================
@st.cache_resource
def load_model():
    model = CrackCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


model = load_model()


# =====================
# 4. 图像预处理
# =====================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# =====================
# 5. 网页界面
# =====================
st.title("混凝土裂缝智能识别系统")

st.write("上传一张混凝土表面图片，系统会判断其是否存在裂缝。")

uploaded_file = st.file_uploader(
    "请选择一张图片",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="上传的图片", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()

    classes = ["crack", "non_crack"]

    predicted_class = classes[predicted_index]
    crack_prob = probabilities[0][0].item()
    non_crack_prob = probabilities[0][1].item()

    if predicted_class == "crack":
        st.error("识别结果：有裂缝 crack")
    else:
        st.success("识别结果：无裂缝 non_crack")

    st.write("预测概率：")
    st.write(f"crack 有裂缝：{crack_prob:.4f}")
    st.write(f"non_crack 无裂缝：{non_crack_prob:.4f}")

    st.bar_chart({
        "probability": {
            "crack": crack_prob,
            "non_crack": non_crack_prob
        }
    })

st.warning("说明：当前模型为课程项目基础版，只能进行有无裂缝分类，不能精确定位裂缝位置或测量裂缝宽度。")
