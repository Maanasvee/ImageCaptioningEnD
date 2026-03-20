import torch
import pickle
import os
import sys
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from model import ImageCaptioningModel
from preprocess import val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    vocab = ckpt["vocab"]
    model = ImageCaptioningModel(
        ckpt["embed_dim"], ckpt["hidden_dim"],
        len(vocab), ckpt["num_layers"], ckpt["dropout"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab

def generate_caption(image_path, model, vocab):
    img     = Image.open(image_path).convert("RGB")
    img     = val_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encoder(img)
        caption  = model.decoder.generate(features, vocab, device=DEVICE)
    return caption

if __name__ == "__main__":
    ckpt  = os.path.join(os.path.dirname(__file__), "../checkpoints/best_caption_model.pt")
    model, vocab = load_model(ckpt)
    # Test with a sample image
    test_img = "../data/Images/667626_18933d713e.jpg"
    if os.path.exists(test_img):
        cap = generate_caption(test_img, model, vocab)
        print(f"Caption: {cap}")
    else:
        print("Place a test image path above to test!")