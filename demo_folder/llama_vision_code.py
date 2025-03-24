# !pip install unsloth pillow transformers torch sentence-transformers

# from unsloth import FastVisionModel
# import torch
# from PIL import Image
# from transformers import TextStreamer
# from sentence_transformers import SentenceTransformer, util

from sentence_transformers import SentenceTransformer, util
import json

def compute_similarity(similarity_model, desc1, desc2):
    print(desc1)
    # Convert JSON strings to dictionaries if needed
    if isinstance(desc1, str):
        desc1 = json.loads(desc1)  # Convert from string to dict

    if isinstance(desc2, str):
        desc2 = json.loads(desc2)  # Convert from string to dict
    print(desc1)
    scores = {}

    # Function to compute cosine similarity between text values
    def text_similarity(text1, text2):
        embedding1 = similarity_model.encode(text1.lower(), convert_to_tensor=True)
        embedding2 = similarity_model.encode(text2.lower(), convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding1, embedding2).item() * 100  # Convert to percentage

    # Compare individual features
    scores["eyes_color"] = text_similarity(desc1["eyes"]["color"], desc2["eyes"]["color"])
    scores["eyes_shape"] = text_similarity(desc1["eyes"]["shape"], desc2["eyes"]["shape"])

    scores["face_shape"] = text_similarity(desc1["face"]["shape"], desc2["face"]["shape"])
    scores["face_features"] = text_similarity(", ".join(desc1["face"]["features"]), ", ".join(desc2["face"]["features"]))

    scores["hair_color"] = text_similarity(desc1["hair"]["color"], desc2["hair"]["color"])
    scores["hair_style"] = text_similarity(desc1["hair"]["style"], desc2["hair"]["style"])

    # Compute overall similarity as the average of key-wise scores
    final_similarity = sum(scores.values()) / len(scores)

    return scores, final_similarity

import torch
import re
import json
from PIL import Image
from transformers import TextStreamer
from sentence_transformers import SentenceTransformer, util
from unsloth import FastVisionModel

# Constants
MODEL_NAME = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
SIMILARITY_THRESHOLD = 80  # Percentage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Vision Model
def load_vision_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

# Load Sentence Transformer Model for Similarity Comparison
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Extract and Clean JSON Data from Raw Text
def extract_all_json(raw_description):
    matches = re.findall(r'\{[\s\S]*?\}', raw_description)

    if not matches:
        print("No valid JSON found.")
        return []

    json_list = []
    for json_str in matches:
        try:
            json_obj = json.loads(json_str)
            json_list.append(json_obj)
        except json.JSONDecodeError:
            print("Error: Skipping invalid JSON block.")

    return json_list

# Describe an Anime Character from Image
def describe_anime_image(model, tokenizer, image_path):
    image = Image.open(image_path).convert("RGB")
    instruction = '''You are an anime character expert. Analyze the given image and describe the anime character strictly in JSON format.
    Ensure the output follows this exact JSON structure:

    <json>
    {
        "eyes": {
            "color": "eye color",
            "shape": "eye shape",
            "size": "eye size",
            "shine": "eye shine level",
            "pupil_shape": "pupil shape",
            "eyelashes": "eyelash style",
            "expression": "eye expression"
        },
        "face": {
            "shape": "face shape",
            "nose": "nose type",
            "mouth": "mouth type",
            "eyebrows": "eyebrow shape",
            "features": ["list of distinct facial features"]
        },
        "hair": {
            "color": "hair color",
            "style": "hair style",
            "length": "hair length",
            "texture": "hair texture",
            "parting": "hair parting",
            "accessories": ["list of hair accessories"]
        },
        "clothing": {
            "style": "clothing style",
            "color": "clothing color",
            "accessories": ["list of clothing accessories"]
        }
    }
    </json>

    Only return valid JSON inside <json> tags. Do not add extra text before or after the JSON.
    '''

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": instruction}
    ]}]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        images=image,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(DEVICE)

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=0.01,
        min_p=0.1,
    )

    raw_description = tokenizer.decode(output[0], skip_special_tokens=True)
    matches = re.findall(r"<json>\s*(\{.*?\})\s*</json>", raw_description, re.DOTALL)

    if matches:
        last_json = matches[-1]
        return last_json

# Compute Similarity Between Two Descriptions
# def compute_similarity(similarity_model, desc1, desc2):
#     embedding1 = similarity_model.encode(json.dumps(desc1), convert_to_tensor=True)
#     embedding2 = similarity_model.encode(json.dumps(desc2), convert_to_tensor=True)
#     return util.pytorch_cos_sim(embedding1, embedding2).item() * 100

# Main Execution
model, tokenizer = load_vision_model()
similarity_model = load_similarity_model()

image1_path = "/content/raiden_1.png"  # Replace with actual file path
image2_path = "/content/raiden_@.png"  # Replace with actual file path

print("Processing images...")
description1 = describe_anime_image(model, tokenizer, image1_path)
description2 = describe_anime_image(model, tokenizer, image2_path)

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute similarity
key_similarities, overall_similarity = compute_similarity(similarity_model, str(description1), str(description2))

print("Key-wise Similarities:", key_similarities)
print(f"Overall Similarity: {overall_similarity:.2f}%")