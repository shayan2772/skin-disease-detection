import os
import torch
from flask import Flask

app = Flask(__name__)
app.config['UPLOAD_PATH'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'skin-model-pokemon.pt')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Please place 'skin-model-pokemon.pt' in the project root directory."
    )
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
model.eval()

CLASSES = [
    'Acanthosis Nigricans',
    'Acne',
    'Acne Scars',
    'Alopecia Areata',
    'Dry Skin',
    'Melasma',
    'Oily Skin',
    'Vitiligo',
    'Warts'
]

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)

from app import routes
