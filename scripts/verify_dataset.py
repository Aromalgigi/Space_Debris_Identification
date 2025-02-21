import os
import cv2
import matplotlib.pyplot as plt

# Define dataset paths
dataset_path = os.path.join(os.getcwd(), "dataset")
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
test_path = os.path.join(dataset_path, "test")

# Check if dataset folders exist
print("Checking dataset structure...")
for folder in [train_path, valid_path, test_path]:
    if os.path.exists(folder):
        print(f"✅ Found: {folder}")
    else:
        print(f"❌ Missing: {folder}")

# Count images and labels
def count_files(path):
    images = len(os.listdir(os.path.join(path, "images")))
    labels = len(os.listdir(os.path.join(path, "labels")))
    return images, labels

train_images, train_labels = count_files(train_path)
valid_images, valid_labels = count_files(valid_path)
test_images, test_labels = count_files(test_path)

print("\n📊 Dataset Statistics:")
print(f"Train Images: {train_images}, Train Labels: {train_labels}")
print(f"Valid Images: {valid_images}, Valid Labels: {valid_labels}")
print(f"Test Images: {test_images}, Test Labels: {test_labels}")

# Visualize one image with bounding boxes
def visualize_sample():
    image_path = os.path.join(train_path, "images", os.listdir(os.path.join(train_path, "images"))[0])
    label_path = os.path.join(train_path, "labels", os.listdir(os.path.join(train_path, "labels"))[0])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(label_path, "r") as file:
        labels = file.readlines()

    h, w, _ = image.shape
    for label in labels:
        data = list(map(float, label.strip().split()))
        class_id = int(data[0])
        x1, y1, x2, y2, x3, y3, x4, y4 = data[1:]

        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)
        x3, y3 = int(x3 * w), int(y3 * h)
        x4, y4 = int(x4 * w), int(y4 * h)

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(image, (x2, y2), (x3, y3), (0, 255, 0), 2)
        cv2.line(image, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.line(image, (x4, y4), (x1, y1), (0, 255, 0), 2)
        cv2.putText(image, f"Class {class_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Show sample image with bounding box
visualize_sample()
