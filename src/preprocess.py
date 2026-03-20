import os
import json
import pickle
from collections import Counter
from PIL import Image
import torchvision.transforms as transforms

# ── Vocabulary ─────────────────────────────────────────────────────
PAD   = "<pad>"
SOS   = "<sos>"
EOS   = "<eos>"
UNK   = "<unk>"
MIN_FREQ = 3

def build_vocab(captions, min_freq=MIN_FREQ):
    counter = Counter()
    for cap in captions:
        counter.update(cap.lower().split())
    vocab = {PAD:0, SOS:1, EOS:2, UNK:3}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode_caption(caption, vocab, max_len=50):
    tokens = caption.lower().split()[:max_len]
    ids    = [vocab[SOS]]
    ids   += [vocab.get(t, vocab[UNK]) for t in tokens]
    ids   += [vocab[EOS]]
    return ids

def load_captions(caption_file):
    """
    Load captions from Flickr8k captions.txt
    Returns dict: {image_name: [caption1, caption2, ...]}
    """
    image_captions = {}
    with open(caption_file, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",", 1)
            if len(parts) != 2: continue
            img_name, caption = parts
            img_name = img_name.strip()
            caption  = caption.strip()
            if img_name not in image_captions:
                image_captions[img_name] = []
            image_captions[img_name].append(caption)
    return image_captions

# Image transform for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Image transform for inference
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    # Find caption file
    caption_file = None
    for root, dirs, files in os.walk("../data"):
        for f in files:
            if "caption" in f.lower() and f.endswith(".txt"):
                caption_file = os.path.join(root, f)
                break

    assert caption_file, "Caption file not found in data/"
    print(f"Found captions: {caption_file}")

    image_captions = load_captions(caption_file)
    print(f"Total images : {len(image_captions)}")

    all_captions = [cap for caps in image_captions.values() for cap in caps]
    print(f"Total captions: {len(all_captions)}")

    vocab = build_vocab(all_captions)
    print(f"Vocab size: {len(vocab)}")

    # Save
    with open("../data/vocab.pkl",           "wb") as f: pickle.dump(vocab,           f)
    with open("../data/image_captions.pkl",  "wb") as f: pickle.dump(image_captions,  f)
    print("Saved vocab.pkl and image_captions.pkl")