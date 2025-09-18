import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print(
    f"CLIP clip.available_models() = {clip.available_models()}\n"
)

model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize([
    "human",
    "robot",
    "machine learning",
    "framework",
    "diagram",
    "picture",
    "painting",
    "photo",
    "math",
    "unknown",
    "dog",
    "cat",
]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print(
    f"Label probs (image -> text): {probs}\n"
    f"Total probability: {probs.sum()}",
)  # prints: [[0.9927937  0.00421068 0.00299572]]
