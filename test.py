import torch
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import os
from matplotlib import pyplot as plt

# NVIDIA GPU kullanımı
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# EfficientNetV2-Small modelini oluşturma
def create_model(num_classes):
    model = efficientnet_v2_s(weights=None)  # Ağırlıkları manuel olarak yüklüyoruz
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
    return model

# Test veri setini yüklemek için veri dönüşümleri
test_transform = transforms.Compose([
    transforms.Resize(380),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Görüntüyü tahmin et ve sonucu döndür
def predict_image(model, image_path, class_names):
    model.eval()
    model = model.to(device)

    # Görüntüyü yükle ve işleme al
    image = Image.open(image_path).convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()], image

# Test işlemi
def test_model_on_dataset(model_path, test_dir, class_names):
    # Modeli yükle
    model = create_model(len(class_names))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # Test veri setini yükle
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    correct = 0
    total = len(test_dataset)

    for idx, (inputs, labels) in enumerate(test_loader):
        image_path, _ = test_dataset.samples[idx]
        predicted_class, original_image = predict_image(model, image_path, class_names)

        true_class = class_names[labels.item()]

        # Tahmin sonucu doğru mu?
        is_correct = predicted_class == true_class
        correct += is_correct

        # Görüntüyü göster ve sonuçları ekrana yaz
        plt.imshow(original_image)
        plt.axis('off')
        plt.title(f"Tahmin: {predicted_class}\nGerçek: {true_class}\n{'Doğru' if is_correct else 'Yanlış'}")
        plt.show()

        print(f"Görüntü: {os.path.basename(image_path)}")
        print(f"Tahmin Edilen Sınıf: {predicted_class}")
        print(f"Gerçek Sınıf: {true_class}")
        print(f"Tahmin {'Doğru' if is_correct else 'Yanlış'}")
        print('-' * 50)

    accuracy = (correct / total) * 100
    print(f"Toplam Test Verisi: {total}")
    print(f"Doğru Tahminler: {correct}")
    print(f"Başarı Oranı: {accuracy:.2f}%")

# Test işlemini başlat
if __name__ == "__main__":
    model_path = "domates_model.pth"  # Eğitilen model dosyasının yolu
    test_dir = r"C:\Users\mucah\OneDrive\Desktop\python\İshak Hoca Ödev\processed_dataset\test"  # Test veri seti dizini

    # Sınıf isimlerini belirle
    test_dataset = datasets.ImageFolder(test_dir)
    class_names = test_dataset.classes

    # Test modeli çalıştır
    test_model_on_dataset(model_path, test_dir, class_names)
