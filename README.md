**About Dataset**

This dataset focuses on classifying tomato leaf images into distinct categories based on health and various diseases. Each class corresponds to either a specific disease affecting tomato plants or a healthy state of the plant. Below is a detailed explanation of each class:
https://www.kaggle.com/datasets/ahmadzargar/tomato-leaf-disease-dataset-segmented
Classes and Explanations

**Bacterial Spot**

Cause: Xanthomonas bacteria infecting tomato leaves.
Symptoms: Small, dark, and water-soaked spots appear on the leaves, which later turn brown. These spots may merge, causing significant damage.
Impact: Reduces photosynthesis and leads to premature leaf drop, affecting yield.

![image (6)](https://github.com/user-attachments/assets/d170ba38-6edb-4d42-aea4-119d366451ea)

**Early Blight**

Cause: Fungal infection by Alternaria solani.
Symptoms: Circular brown lesions with concentric rings, commonly called a "bullseye" appearance. These spots start at the lower leaves and progress upwards.
Impact: Weakens the plant, reducing fruit size and production.

![image (6)](https://github.com/user-attachments/assets/099d8eb1-596a-48db-8e16-bfbc82c253f1)


**Late Blight**

Cause: Phytophthora infestans, a water mold.
Symptoms: Irregular, water-soaked lesions on leaves that rapidly turn dark brown or black. White mold may appear under humid conditions.
Impact: Can destroy entire crops quickly if not controlled.

![image (4)](https://github.com/user-attachments/assets/6c63b676-835d-45f3-a4ef-843f71be0fd7)


**Leaf Mold**

Cause: Fungus Cladosporium fulvum.
Symptoms: Yellow spots on the upper leaf surface, with corresponding olive-green mold on the underside.
Impact: Reduces photosynthetic activity and overall plant health.

![image (6)](https://github.com/user-attachments/assets/8bb57d7d-08ea-489e-8c09-ebd19aae6068)


**Septoria Leaf Spot**

Cause: Septoria lycopersici fungus.
Symptoms: Numerous small, circular spots with grayish centers and dark brown margins, usually on lower leaves.
Impact: Leads to severe leaf drop, exposing fruits to sunscald.

![image (7)](https://github.com/user-attachments/assets/6509d762-2655-44df-a7dd-b858aeace814)


**Spider Mites (Two-Spotted Spider Mites)**

Cause: Tiny arachnids, Tetranychus urticae.
Symptoms: Leaves appear speckled, yellowish, and may have a fine webbing. Infested leaves eventually dry out.
Impact: Reduces plant vigor and yield.

![image (7)](https://github.com/user-attachments/assets/c0a89240-9549-4822-b9f2-c9edfbf51657)


**Target Spot**

Cause: Fungus Corynespora cassiicola.
Symptoms: Large brown lesions with concentric rings, often appearing on both leaves and fruits.
Impact: Decreases fruit quality and overall plant health.

![image (6)](https://github.com/user-attachments/assets/72485f9d-24ce-4614-88d2-5b3d8635407a)


**Tomato Mosaic Virus**

Cause: Viral infection by the Tomato mosaic virus (ToMV).
Symptoms: Leaves show mottled yellow and green patterns, become distorted, and fruits may develop uneven ripening.
Impact: Affects fruit appearance, making it less marketable.

![image (12)](https://github.com/user-attachments/assets/c4760bcf-d473-493b-9266-0dd5f13c1718)


**Tomato Yellow Leaf Curl Virus**

Cause: A viral infection spread by whiteflies (Bemisia tabaci).
Symptoms: Leaves curl upward, turn yellow, and plants become stunted. Flowers may drop before fruit formation.
Impact: Severely reduces yield.

![image (6)](https://github.com/user-attachments/assets/b45e7134-5310-4bd6-8fbd-5c79b1a97f83)


**Healthy**

Represents leaves without any signs of diseases or stress.
A benchmark class used to ensure the model can distinguish between diseased and healthy plants.

![image (11)](https://github.com/user-attachments/assets/0ac13214-a5c7-427c-9315-61ebe114607c)



**Dataset Splitting Methodology**
In this project, the dataset is divided into three parts: training (70%), validation (15%), and test (15%). The goal of this splitting strategy is to ensure the model can learn effectively, validate its performance during training, and evaluate its generalization ability on unseen data.



**ABOUT ALGORİTHM**

EfficientNetV2-Small Algorithm
EfficientNetV2-Small is a state-of-the-art convolutional neural network (CNN) architecture designed for image classification tasks. It is part of the EfficientNet family, which focuses on achieving high accuracy with fewer computational resources. The "Small" variant is optimized for speed and efficiency while maintaining competitive accuracy on large datasets like ImageNet.

Key Features of EfficientNetV2-Small
Scalable Architecture:

EfficientNetV2 introduces compound scaling, which balances three key dimensions of the model:
Depth: Number of layers in the model.
Width: Number of channels in each layer.
Resolution: Input image size.
The "Small" version uses a compact configuration suitable for tasks requiring lower latency.
MBConv and Fused-MBConv Blocks:

Combines traditional Mobile Inverted Bottleneck Convolution (MBConv) with Fused-MBConv for faster and more efficient operations.
Fused-MBConv simplifies computations by combining depthwise separable convolution and squeeze-and-excitation layers.
Pretraining on Large Datasets:

EfficientNetV2-Small is pretrained on the ImageNet-1K dataset, enabling it to generalize well to diverse datasets with fewer labeled examples.
Optimization for Speed:

Designed with smaller convolutional kernels and optimized batch normalization, making it faster on modern GPUs and TPUs.
It’s well-suited for real-time applications requiring fast inference times.
Dynamic Training Techniques:

Employs techniques like progressive learning, where smaller resolutions are trained first, gradually increasing the resolution to improve efficiency during training.

**Why EfficientNetV2-Small Was Chosen**
Accuracy vs. Efficiency:

It achieves high accuracy while using fewer parameters compared to other architectures like ResNet or Inception.
Suitable for datasets like tomato leaf disease classification, which requires robust feature extraction from high-resolution images.
Pretrained Weights:

Pretrained on ImageNet, it provides a strong starting point for transfer learning, reducing training time and improving performance on smaller datasets.
Scalability:

The architecture can be scaled up (e.g., EfficientNetV2-Large) or down (EfficientNetV2-Tiny) based on the dataset size and computational resources available.
Integration with PyTorch:

PyTorch’s library supports EfficientNetV2, making it easy to implement, fine-tune, and integrate into custom workflows.


**RESULT**

Epoch
Train Loss
Train Acc
Val Loss
Val Acc


1
0.4000
0.9300
0.4600
0.9400


2
0.3200
0.9500
0.3800
0.9500


3
0.2600
0.9600
0.3000
0.9600


4
0.2100
0.9700
0.2500
0.9700


5
0.1700
0.9750
0.1900
0.9750


6
0.1400
0.9800
0.1400
0.9800


7
0.1100
0.9850
0.1000
0.9850


8
0.0900
0.9880
0.0800
0.9870


9
0.0800
0.9900
0.0720
0.9880


10
0.0700
0.9860
0.0700
0.9860

 

