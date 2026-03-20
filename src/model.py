import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN Encoder using pretrained ResNet50.
    Extracts visual features from images.
    Output: feature vector of size embed_dim
    """
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet   = nn.Sequential(*modules)
        self.linear   = nn.Linear(resnet.fc.in_features, embed_dim)
        self.bn       = nn.BatchNorm1d(embed_dim)
        # Freeze CNN — only train linear layer
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)        # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]
        features = self.bn(self.linear(features))        # [B, embed_dim]
        return features


class DecoderRNN(nn.Module):
    """
    RNN Decoder using LSTM.
    Takes image features + previous words → generates next word.
    Uses attention-like mechanism via feature injection.
    """
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                 batch_first=True,
                                 dropout=dropout if num_layers>1 else 0)
        self.linear    = nn.Linear(hidden_dim, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, features, captions):
        # features: [B, embed_dim]
        # captions: [B, seq_len] — exclude last token for input
        embeddings = self.dropout(self.embedding(captions[:, :-1]))  # [B, seq-1, E]
        # Inject image features as first input
        features   = features.unsqueeze(1)                           # [B, 1, E]
        inputs     = torch.cat((features, embeddings), dim=1)        # [B, seq, E]
        outputs, _ = self.lstm(inputs)                               # [B, seq, H]
        predictions = self.linear(outputs)                           # [B, seq, V]
        return predictions

    def generate(self, features, vocab, max_len=30, device="cpu"):
        """Generate caption for a single image."""
        result    = []
        sos_id    = vocab["<sos>"]
        eos_id    = vocab["<eos>"]
        input_tok = torch.tensor([[sos_id]]).to(device)  # [1,1]
        hidden    = None
        # First input is image feature
        inp = features.unsqueeze(1)  # [1,1,E]
        out, hidden = self.lstm(inp, hidden)

        for _ in range(max_len):
            emb        = self.embedding(input_tok)       # [1,1,E]
            out, hidden = self.lstm(emb, hidden)          # [1,1,H]
            pred       = self.linear(out.squeeze(1))     # [1,V]
            top        = pred.argmax(1).item()
            if top == eos_id: break
            result.append(top)
            input_tok = torch.tensor([[top]]).to(device)

        inv_vocab = {v:k for k,v in vocab.items()}
        return " ".join([inv_vocab.get(i,"") for i in result])


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderRNN(embed_dim, hidden_dim, vocab_size, num_layers, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs  = self.decoder(features, captions)
        return outputs