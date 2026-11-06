
import torch
from torch import nn
from torchvision import models, transforms

from .paths import OUTPUT_DIR
from .class_names import CLASS_NAMES


MODEL_PATH = OUTPUT_DIR / "bodypart_resnet18.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(path=MODEL_PATH):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["num_classes"] != len(CLASS_NAMES):
        raise ValueError("Body-part checkpoint has an unexpected number of classes.")
    if "class_names" in checkpoint and checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("Body-part checkpoint class order does not match this application.")

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def predict_image(model, image, threshold=0.5):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0]

    results = [
        {
            "class_id": i,
            "score": float(probabilities[i])
        }
        for i in range(len(probabilities))
    ]

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    selected = [
        result for result in results
        if result["score"] >= threshold
    ]

    return selected, results
