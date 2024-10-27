import os
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

# Load processor and model
processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# Create a 'results' directory if it does not exist
os.makedirs("results", exist_ok=True)

# Function to preprocess and unnormalize the image
def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

# Loop through all images in the directory
image_directory = "images"
output_directory = "results"

for image_file in os.listdir(image_directory):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_directory, image_file)
        image = Image.open(image_path)

        image = Image.open(image_path).convert("RGB")
        # Text labels for detection
        texts = [["a photo of a cat", "a photo of a dog", "a photo of a human", "a photo of a bird" ]]
        inputs = processor(text=texts, images=image, return_tensors="pt")

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Unnormalize image
        unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        # Get target sizes
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs, threshold=0.2, target_sizes=target_sizes
        )

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(unnormalized_image)
        font = ImageFont.load_default()

        i = 0  # First image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        for box, score, label in zip(boxes, scores, labels):
            box = [round(coord, 2) for coord in box.tolist()]
            draw.rectangle(box, outline="red", width=3)  # Bounding box in red
            draw.text((box[0], box[1] - 10), f"{text[label]}: {round(score.item(), 2)}", fill="black", font=font)  # Label with white text

        # Save the image with bounding boxes in the 'results' directory
        output_path = os.path.join(output_directory, f"processed_{image_file}")
        unnormalized_image.save(output_path)
        print(f"Processed and saved: {output_path}")
