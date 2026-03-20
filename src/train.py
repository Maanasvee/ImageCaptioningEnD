import torch
import torch.nn as nn
import pickle
import os
import random
import time
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from model import ImageCaptioningModel
from preprocess import encode_caption, train_transform, val_transform

# ══════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════
EMBED_DIM  = 256   # CNN output + word embedding dimension
HIDDEN_DIM = 512   # LSTM hidden state size
NUM_LAYERS = 1     # LSTM layers
DROPOUT    = 0.3   # Dropout rate
BATCH_SIZE = 32    # Images per batch
N_EPOCHS   = 10    # Training epochs
LR         = 3e-4  # Learning rate (Adam)
CLIP       = 5.0   # Gradient clip
MAX_LEN    = 30    # Max caption length
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flickr8kDataset(Dataset):
    def __init__(self, image_captions, vocab, img_dir, transform, max_samples=None):
        self.items     = []
        self.img_dir   = img_dir
        self.transform = transform
        for img_name, captions in image_captions.items():
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path): continue
            for cap in captions:
                self.items.append((img_name, encode_caption(cap, vocab, MAX_LEN)))
        random.shuffle(self.items)
        if max_samples:
            self.items = self.items[:max_samples]
        print(f"Dataset size: {len(self.items)}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_name, caption = self.items[i]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(caption)

def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs, 0)
    caps = pad_sequence(caps, batch_first=True, padding_value=0)
    return imgs, caps

if __name__ == "__main__":
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Using device: {DEVICE}")

    # Load data
    with open("../data/vocab.pkl",          "rb") as f: vocab          = pickle.load(f)
    with open("../data/image_captions.pkl", "rb") as f: image_captions = pickle.load(f)

    # Find images folder
    img_dir = None
    for root, dirs, files in os.walk("../data"):
        for d in dirs:
            if "image" in d.lower() or "img" in d.lower():
                img_dir = os.path.join(root, d)
                break
        if img_dir: break
    assert img_dir, "Images folder not found!"
    print(f"Images folder: {img_dir}")
    print(f"Vocab size: {len(vocab)}")

    # Split data
    all_imgs = list(image_captions.keys())
    random.shuffle(all_imgs)
    split     = int(0.9 * len(all_imgs))
    train_imgs = {k: image_captions[k] for k in all_imgs[:split]}
    val_imgs   = {k: image_captions[k] for k in all_imgs[split:]}

    train_ds = Flickr8kDataset(train_imgs, vocab, img_dir, train_transform)
    val_ds   = Flickr8kDataset(val_imgs,   vocab, img_dir, val_transform)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Build model
    model     = ImageCaptioningModel(EMBED_DIM, HIDDEN_DIM, len(vocab), NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    best_val = float("inf")
    os.makedirs("../checkpoints", exist_ok=True)

    for epoch in range(1, N_EPOCHS+1):
        # Train
        model.train()
        tr_loss = 0
        t0 = time.time()
        for imgs, caps in train_dl:
            imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, caps)
            loss    = criterion(
                outputs.reshape(-1, len(vocab)),
                caps[:, 1:].reshape(-1) if outputs.size(1)==caps.size(1)-1
                else caps.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, caps in val_dl:
                imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
                outputs  = model(imgs, caps)
                val_loss += criterion(
                    outputs.reshape(-1, len(vocab)),
                    caps[:, 1:].reshape(-1) if outputs.size(1)==caps.size(1)-1
                    else caps.reshape(-1)
                ).item()
        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{N_EPOCHS} | Train: {tr_loss:.4f} | Val: {val_loss:.4f} | Time: {time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state" : model.state_dict(),
                "vocab"       : vocab,
                "embed_dim"   : EMBED_DIM,
                "hidden_dim"  : HIDDEN_DIM,
                "num_layers"  : NUM_LAYERS,
                "dropout"     : DROPOUT,
            }, "../checkpoints/best_caption_model.pt")
            print(f"  ✓ Saved (val: {best_val:.4f})")

    print("Training complete!")