import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ✅ Define Dataset Paths
train_dir = "C:/Users/Lenovo/Desktop/Data_Science/train"
test_dir = "C:/Users/Lenovo/Desktop/Data_Science/test"

# ✅ Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor()
])

# ✅ Load Training and Testing Data
train_data = datasets.ImageFolder(train_dir, transform)
test_data = datasets.ImageFolder(test_dir, transform)

train_loader = DataLoader(train_data, batch_size=64, num_workers=0, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, num_workers=0, shuffle=True)

print(f"Total training images: {len(train_data)}")
print(f"Total testing images: {len(test_data)}")

# ✅ Define the CNN Model
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

# ✅ Set Device to GPU if Available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ✅ Initialize Model
model = CNN().to(device)

# ✅ Load Model if Exists (Continue Training)
model_path = "model.pth"
try:
    model.load_state_dict(torch.load(model_path))
    print("Loaded existing model. Continuing training...")
except FileNotFoundError:
    print("No saved model found. Training from scratch...")

# ✅ Define Loss Function and Optimizer
learning_rate = 0.00025
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Training Function
def train_model(model, train_loader, optimizer, loss_fn, num_epochs=7):  # Updated to 7 epochs
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        correct_predictions, total_samples, loss_per_epoch = 0, 0, 0
        
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_per_epoch += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if i % 50 == 0:
                print(f"Batch {i} --> Loss: {loss.item():.4f}")

        # Calculate average loss and accuracy
        loss_avg = loss_per_epoch / len(train_loader)
        accuracy = (correct_predictions / total_samples) * 100
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss_avg:.4f}, Accuracy: {accuracy:.2f}%\n")
        
        # Save model only if loss improves
        if loss_avg < best_loss:
            best_loss = loss_avg
            torch.save(model.state_dict(), model_path)
            print("✅ Model improved and saved!")

# ✅ Testing Function
def test_model(model, test_loader, loss_fn):
    correct_predictions, total_samples, loss_per_step = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            loss = loss_fn(output, labels)
            loss_per_step += loss.item()
            
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    test_loss = loss_per_step / len(test_loader)
    test_accuracy = (correct_predictions / total_samples) * 100
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

# ✅ Run Training & Testing
if __name__ == '__main__':
    train_model(model, train_loader, optimizer, loss_fn, num_epochs=10)  # Updated to 7 epochs
    test_model(model, test_loader, loss_fn)