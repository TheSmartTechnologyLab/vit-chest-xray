# Chest X-ray Image Classifier

This repository contains a fine-tuned **Vision Transformer (ViT)** model for classifying chest X-ray images, utilizing the **CheXpert** dataset. The model is fine-tuned on the task of classifying various lung diseases from chest radiographs, achieving impressive accuracy in distinguishing between different conditions.

## Model Overview

The fine-tuned [model on Huggingface](https://huggingface.co/codewithdark/vit-chest-xray) is based on the **Vision Transformer (ViT)** architecture, which excels in handling image-based tasks by leveraging attention mechanisms for efficient feature extraction. The model was trained on the **CheXpert dataset**, which consists of labeled chest X-ray images for detecting diseases such as pneumonia, cardiomegaly, and others.

## Performance

- **Final Validation Accuracy**: 98.46%
- **Final Training Loss**: 0.1069
- **Final Validation Loss**: 0.0980

The model achieved a significant accuracy improvement during training, demonstrating its ability to generalize well to unseen chest X-ray images.

## Dataset

The dataset used for fine-tuning the model is the **CheXpert** dataset, which includes chest X-ray images from various patients with multi-label annotations. The data includes frontal and lateral views of the chest for each patient, annotated with labels for various lung diseases.

For more details on the dataset, visit the [CheXpert official website](https://stanfordmlgroup.github.io/chexpert/).

## Training Details

The model was fine-tuned using the following settings:

- **Optimizer**: AdamW
- **Learning Rate**: 3e-5
- **Batch Size**: 32
- **Epochs**: 10
- **Loss Function**: Binary Cross-Entropy with Logits
- **Precision**: Mixed precision (via `torch.amp`)

## Usage

### Inference

To use the fine-tuned model for inference, simply load the model from Hugging Face's Model Hub and input a chest X-ray image:

```python
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model and processor
processor = AutoImageProcessor.from_pretrained("codewithdark/vit-chest-xray")
model = AutoModelForImageClassification.from_pretrained("codewithdark/vit-chest-xray")

# Define label columns (class names)
label_columns = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

# Step 1: Load and preprocess the image
image_path = "/content/images.jpeg"  # Replace with your image path

# Open the image
image = Image.open(image_path)

# Ensure the image is in RGB mode (required by most image classification models)
if image.mode != 'RGB':
    image = image.convert('RGB')
    print("Image converted to RGB.")

# Step 2: Preprocess the image using the processor
inputs = processor(images=image, return_tensors="pt")

# Step 3: Make a prediction (using the model)
with torch.no_grad():  # Disable gradient computation during inference
    outputs = model(**inputs)

# Step 4: Extract logits and get the predicted class index
logits = outputs.logits  # Raw logits from the model
predicted_class_idx = torch.argmax(logits, dim=-1).item()  # Get the class index

# Step 5: Map the predicted index to a class label
# You can also use `model.config.id2label`, but we'll use `label_columns` for this task
predicted_class_label = label_columns[predicted_class_idx]

# Output the results
print(f"Predicted Class Index: {predicted_class_idx}")
print(f"Predicted Class Label: {predicted_class_label}")

'''
Output :
Predicted Class Index: 4
Predicted Class Label: No Finding
'''
```

### Fine-Tuning

To fine-tune the model on your own dataset, you can follow the instructions in this repo to adapt the code to your dataset and training configuration.

## Contributing

We welcome contributions! If you have suggestions, improvements, or bug fixes, feel free to fork the repository and open a pull request.

## License

This model is available under the MIT License. See [LICENSE](LICENSE) for more details.

## Acknowledgements

- [CheXpert Dataset](https://stanfordmlgroup.github.io/chexpert/)
- Hugging Face for providing the `transformers` library and Model Hub.

---
Happy coding! 🚀

