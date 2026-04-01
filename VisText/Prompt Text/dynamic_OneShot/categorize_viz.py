# Prompt gemini to categorize the vis, and output it in a strict scheme as input for the dynamic prompt
import json
import os
from pathlib import Path

from google import genai
from google.genai import types


def load_gemini_api_key() -> str:
    for parent in Path(__file__).resolve().parents:
        config_path = parent / "config"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)

            api_key = config.get("gemini_api_key")
            if not isinstance(api_key, str) or not api_key.strip():
                raise KeyError(
                    "gemini_api_key is missing or empty in the root config file"
                )
            return api_key.strip()

    raise FileNotFoundError("Could not find root config file while walking parent directories")


os.environ["GEMINI_API_KEY"] = load_gemini_api_key()

contents = """
You are a chart identification assistant. You will be given a visualization of a scientific chart,
and your task is to identify its category. The possible categories are: bar, area, line.
Respond with only the category, and no other text. The value must be one of the allowed categories.
"""

image_path_area = r'VisText\Prompt Text\ground-truth_val\area\3.png'
image_path_bar = r'VisText\Prompt Text\ground-truth_val\bar\18.png'
image_path_line = r'VisText\Prompt Text\ground-truth_val\line\26.png'

with open(image_path_area, 'rb') as f:
    image_bytes = f.read()
image = types.Part.from_bytes(
  data=image_bytes, mime_type="image/jpeg"
)

allowed_categories = ["bar", "area", "line"]

client = genai.Client()
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[contents, image],
    config=types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=types.Schema(
        type="STRING",
        enum=allowed_categories,
    ),
),
)
print(response.text)
