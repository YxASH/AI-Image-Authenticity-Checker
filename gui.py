import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from torchvision import models

# ✅ Load the trained model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, 1, 0)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(96 * 96 * 32, 512) 
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ✅ Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor()
])

# ✅ Function to Open File and Predict
def classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            result = "Real Image" if predicted.item() == 0 else "AI-Generated Image"
        
        label_result.config(text=f"Prediction: {result}", fg="blue", font=("Arial", 14, "bold"))

# ✅ Create Tkinter GUI
root = tk.Tk()
root.title("AI vs. Real Image Classifier")
root.geometry("400x300")

label_title = Label(root, text="AI vs. Real Image Classifier", font=("Arial", 16, "bold"))
label_title.pack(pady=10)

btn_upload = Button(root, text="Upload Image", command=classify_image, font=("Arial", 12))
btn_upload.pack(pady=20)

label_result = Label(root, text="Prediction: ", font=("Arial", 14))
label_result.pack(pady=10)

root.mainloop()
