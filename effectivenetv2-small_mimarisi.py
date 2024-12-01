import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import os

# NVIDIA GPU kullanımı zorunlu
device = torch.device("cuda:0")  # CUDA'ya özel GPU kullanımı
assert torch.cuda.is_available(), "CUDA is not available. Please ensure NVIDIA drivers and CUDA are properly installed."

# EfficientNetV2-Small modelini oluşturma
def create_model(num_classes):
    model = efficientnet_v2_s(pretrained=True)  # Önceden eğitilmiş model
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# Eğitim fonksiyonu
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10):
    model = model.to(device)  # Modeli GPU'ya taşı

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)  # Veriyi GPU'ya taşı

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model

# Rastgele bir resimle test
def predict_image(model, image_path, class_names, transform):
    model.eval()
    model = model.to(device)

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Veriyi GPU'ya taşı

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]

# Veri artırma ve normalleştirme
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(300),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(380),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(380),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Veri kümesini ayarlama
data_dir = r"C:\Users\mucah\OneDrive\Desktop\python\İshak Hoca Ödev\processed_dataset"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=(x == 'train'))
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

# Modeli oluştur ve eğit
model = create_model(len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer)
torch.save(trained_model.state_dict(), 'domates_model.pth')  # Modeli kaydet

# Test için tahmin
test_image_path = r"C:\Users\mucah\OneDrive\Desktop\python\İshak Hoca Ödev\Tomato___Tomato_Yellow_Leaf_Curl_Virus\image (3).JPG"  # Test resmi yolu
result = predict_image(trained_model, test_image_path, class_names, data_transforms['test'])
print(f"Tahmin edilen sınıf: {result}")
