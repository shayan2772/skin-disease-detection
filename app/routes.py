import os
import base64
import io
import requests
from flask import request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as T
import torch

from app import app, model, CLASSES

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models'
GEMINI_MODEL = 'gemini-2.5-flash'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_transforms():
    return T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])


def image_to_base64(img):
    """Convert PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    img_copy = img.copy()
    img_copy.thumbnail((512, 512))
    img_copy.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def predict_image_local(img):
    """Local model prediction as fallback."""
    tr = get_transforms()
    img_tensor = tr(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img_tensor)
    probs = torch.nn.functional.softmax(out, dim=1)
    confidence, idx = torch.max(probs, 1)
    return CLASSES[idx.item()], round(confidence.item() * 100, 1)


def gemini_request(parts, max_tokens=600, temperature=0.7):
    """Make a request to Gemini API. Returns response text or None."""
    api_key = os.environ.get('GOOGLE_API_KEY', '')
    if not api_key:
        return None

    try:
        response = requests.post(
            f'{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent?key={api_key}',
            headers={'Content-Type': 'application/json'},
            json={
                'contents': [{'parts': parts}],
                'generationConfig': {
                    'maxOutputTokens': max_tokens,
                    'temperature': temperature,
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            candidates = result.get('candidates', [])
            if candidates:
                resp_parts = candidates[0].get('content', {}).get('parts', [])
                for p in resp_parts:
                    if 'text' in p:
                        return p['text']
    except requests.RequestException:
        pass

    return None


def predict_image_gemini(img):
    """Use Gemini Vision to classify the skin condition."""
    b64 = image_to_base64(img)
    conditions_list = ', '.join(CLASSES)

    prompt = (
        f"You are an expert dermatologist. Analyze this skin image carefully and classify it as "
        f"EXACTLY ONE of these conditions: {conditions_list}.\n\n"
        f"Respond with ONLY the exact condition name from the list above. "
        f"No explanation, no extra words, just the condition name."
    )

    parts = [
        {'text': prompt},
        {'inline_data': {'mime_type': 'image/jpeg', 'data': b64}},
    ]

    content = gemini_request(parts, max_tokens=50, temperature=0.1)
    if content:
        # Match against known classes
        for cls in CLASSES:
            if cls.lower() in content.lower():
                return cls
    return None


# --- Page Routes ---

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/app')
def app_page():
    return render_template('app.html')


# --- API Routes ---

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(f.filename):
        return jsonify({'error': 'File type not allowed. Use PNG, JPG, JPEG, or WEBP'}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_PATH'], filename)
    f.save(path)

    try:
        img = Image.open(path).convert('RGB')

        # Try Gemini Vision first for accurate classification
        gemini_prediction = predict_image_gemini(img)

        # Always get local model confidence for the bar
        local_prediction, local_confidence = predict_image_local(img)

        if gemini_prediction:
            prediction = gemini_prediction
            source = 'gemini'
        else:
            prediction = local_prediction
            source = 'local-model'

        return jsonify({
            'prediction': prediction,
            'confidence': local_confidence,
            'source': source,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.route('/advice', methods=['POST'])
def advice():
    data = request.get_json()
    if not data or 'condition' not in data:
        return jsonify({'error': 'No condition provided'}), 400

    condition = data['condition']
    confidence = data.get('confidence', None)

    api_key = os.environ.get('GOOGLE_API_KEY', '')
    if not api_key:
        return jsonify({'error': 'Google API key not configured'}), 500

    confidence_note = ''
    if confidence is not None:
        confidence_note = f' The AI model confidence for this detection is {confidence}%.'

    prompt = (
        f"You are a board-certified dermatologist. Patient diagnosed with **{condition}**.{confidence_note}\n\n"
        f"Write a concise clinical report using ONLY bullet points. No paragraphs. No filler. "
        f"Every line must be a bullet point with specific, actionable information.\n\n"
        f"## Clinical Assessment\n"
        f"- What it is (one line)\n"
        f"- How it looks clinically\n"
        f"- Severity: mild/moderate/severe\n"
        f"- Areas typically affected\n\n"
        f"## Causes & Triggers\n"
        f"- 4-5 bullet points, each one specific cause\n\n"
        f"## Medications\n"
        f"- List each med as: **Drug Name concentration** — how to use, how long\n"
        f"- Include 3-4 topical + 1-2 oral meds with exact dosages\n\n"
        f"## Clinical Procedures\n"
        f"- 2-3 procedures with sessions needed\n\n"
        f"## Daily Routine\n"
        f"- **AM:** numbered steps (cleanser → serum → sunscreen etc.) with key ingredients\n"
        f"- **PM:** numbered steps with key ingredients\n\n"
        f"## Diet\n"
        f"- 3-4 foods to eat\n"
        f"- 3-4 foods to avoid\n\n"
        f"## Prevention\n"
        f"- 4-5 bullet points\n\n"
        f"## Red Flags — See Doctor If\n"
        f"- 3-4 warning signs\n\n"
        f"RULES: Be 100% specific to {condition}. Use real drug names and exact dosages. "
        f"Keep total under 500 words. Bullet points only — no long paragraphs.\n\n"
        f"End with: *AI-generated report — not a substitute for in-person dermatology consultation.*"
    )

    parts = [{'text': prompt}]
    advice_text = gemini_request(parts, max_tokens=1500, temperature=0.7)

    if advice_text:
        return jsonify({'advice': advice_text})

    return jsonify({'error': 'AI advice temporarily unavailable. Please try again later.'}), 503
