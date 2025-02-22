import torch
import torch.nn as nn
import streamlit as st
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# ✅ Load Model Class (Matching Original Trained Model)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, 1, 0)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

        # ✅ Ensure same fc1 input size as training model (294912 features)
        self.fc1 = nn.Linear(294912, 512)  # Use the correct number here
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ✅ Load the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

model_path = "model.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    st.sidebar.success("✅ Model Loaded Successfully!")
except FileNotFoundError:
    st.sidebar.error("🚨 Model file not found! Please check 'model.pth'.")
except RuntimeError as e:
    st.sidebar.error(f"🚨 Model loading error: {e}")

# ✅ Define Image Transformations (Ensure Match with Training Data)
transform = transforms.Compose([
    transforms.Resize((400, 400)),  # Ensure this is the same size used in training
    transforms.ToTensor()
])

# ✅ Streamlit UI
st.title("🖼️ AI vs. Real Image Detector")
st.write("Upload an image, and the model will predict whether it's **AI-generated** or **Real**.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # ✅ Load and display the image
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB mode
    st.image(image, caption="📌 Uploaded Image", use_container_width=True)

    # ✅ Preprocess Image
    image = transform(image).unsqueeze(0).to(device)

    # ✅ Make Prediction
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0] * 100
        _, predicted = torch.max(output, 1)

    # ✅ Ensure Correct Label Mapping
    label = "🟢 Real Image" if predicted.item() == 0 else "🔴 AI-Generated"

    # ✅ Display Result
    st.subheader(f"🎯 Prediction: {label}")

    # ✅ Confidence Scores
    st.write(f"**Confidence Levels:**")
    st.progress(float(probabilities[0].item() / 100))  # Show AI probability
    st.write(f"📷 Real Image: **{probabilities[0]:.2f}%**")
    st.progress(float(probabilities[1].item() / 100))  # Show AI probability
    st.write(f"🧠 AI-Generated: **{probabilities[1]:.2f}%**")

    # ✅ Final Message
    if predicted.item() == 0:
        st.success("✅ This image is likely **Real**!")
    else:
        st.error("⚠️ This image is likely **AI-Generated**.")
