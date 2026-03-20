from flask import Flask, request, jsonify, render_template
import sys, os, uuid
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from caption import load_model, generate_caption

app    = Flask(__name__)
CKPT   = os.path.join(os.path.dirname(__file__), '../checkpoints/best_caption_model.pt')
UPLOAD = os.path.join(os.path.dirname(__file__), 'static/uploads')
os.makedirs(UPLOAD, exist_ok=True)

model, vocab = load_model(CKPT)
print("Model loaded!")

@app.route("/")
def index(): return render_template("index.html")

@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})
    file     = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD, filename)
    img      = Image.open(file.stream).convert("RGB")
    img.save(filepath)
    cap = generate_caption(filepath, model, vocab)
    os.remove(filepath)
    return jsonify({"caption": cap})

if __name__ == "__main__":
    print("Starting at http://localhost:5000")
    app.run(debug=True, port=5000)