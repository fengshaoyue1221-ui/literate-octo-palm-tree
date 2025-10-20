# literate-octo-palm-tree
Multimodal Speech-Language AI for Children is an advanced AI system designed to analyze children’s speech and corresponding text inputs to provide emotion recognition, pronunciation accuracy scoring, and personalized feedback.
multimodal_speech_ai/
│
├── data/
│   ├── raw/                # 原始音频和文本
│   └── processed/          # 处理后音频特征 + 文本embedding
│
├── notebooks/
│   └── demo.ipynb          # 演示特征提取 + 推理
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── utils/
│   ├── visualization.py
│   └── metrics.py
│
├── requirements.txt
└── README.md
import librosa
import torch
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def extract_audio_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    return torch.tensor(features, dtype=torch.float).unsqueeze(0)  # (1, feat_dim, time)

def tokenize_text(text, max_len=128):
    return tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
import torch
import torch.nn as nn
from transformers import BertModel

class AudioBranch(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        audio_feat = (lstm_out * attn_weights).sum(dim=1)
        return audio_feat

class MultimodalModel(nn.Module):
    def __init__(self, audio_dim=40, hidden_dim=128, bert_name='bert-base-uncased', output_dim=3):
        super().__init__()
        self.audio_branch = AudioBranch(audio_dim, hidden_dim)
        self.bert = BertModel.from_pretrained(bert_name)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2 + 768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, audio_feats, input_ids, attention_mask):
        audio_out = self.audio_branch(audio_feats)
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        fused = torch.cat([audio_out, text_out], dim=1)
        output = self.fusion(fused)
        return output
import torch
import torch.nn as nn
from src.models import MultimodalModel
from src.data_processing import extract_audio_features, tokenize_text

# Initialize model, loss, optimizer
model = MultimodalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Dummy training loop (replace with dataset loader)
audio_feats = extract_audio_features('data/raw/sample.wav')
text_inputs = tokenize_text("Hello, I am learning English.")

for epoch in range(5):
    output = model(audio_feats, text_inputs['input_ids'], text_inputs['attention_mask'])
    loss = criterion(output, torch.tensor([1]))  # Example label
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")
import torch
import torch.nn as nn
from src.models import MultimodalModel
from src.data_processing import extract_audio_features, tokenize_text

# Initialize model, loss, optimizer
model = MultimodalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Dummy training loop (replace with dataset loader)
audio_feats = extract_audio_features('data/raw/sample.wav')
text_inputs = tokenize_text("Hello, I am learning English.")

for epoch in range(5):
    output = model(audio_feats, text_inputs['input_ids'], text_inputs['attention_mask'])
    loss = criterion(output, torch.tensor([1]))  # Example label
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")
import torch
import torch.nn as nn
from src.models import MultimodalModel
from src.data_processing import extract_audio_features, tokenize_text

# Initialize model, loss, optimizer
model = MultimodalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Dummy training loop (replace with dataset loader)
audio_feats = extract_audio_features('data/raw/sample.wav')
text_inputs = tokenize_text("Hello, I am learning English.")

for epoch in range(5):
    output = model(audio_feats, text_inputs['input_ids'], text_inputs['attention_mask'])
    loss = criterion(output, torch.tensor([1]))  # Example label
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")

import torch
import torch.nn as nn
from src.models import MultimodalModel
from src.data_processing import extract_audio_features, tokenize_text

# Initialize model, loss, optimizer
model = MultimodalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Dummy training loop (replace with dataset loader)
audio_feats = extract_audio_features('data/raw/sample.wav')
text_inputs = tokenize_text("Hello, I am learning English.")

for epoch in range(5):
    output = model(audio_feats, text_inputs['input_ids'], text_inputs['attention_mask'])
    loss = criterion(output, torch.tensor([1]))  # Example label
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")

from src.models import MultimodalModel
from src.data_processing import extract_audio_features, tokenize_text
import torch

def run_inference(audio_path, text):
    model = MultimodalModel()
    model.eval()
    audio_feats = extract_audio_features(audio_path)
    text_inputs = tokenize_text(text)
    with torch.no_grad():
        output = model(audio_feats, text_inputs['input_ids'], text_inputs['attention_mask'])
    predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

# Example usage
if __name__ == "__main__":
    pred = run_inference('data/raw/sample.wav', "Hello world")
    print("Predicted Emotion Class:", pred)

import matplotlib.pyplot as plt
import librosa.display

def plot_mfcc(mfcc):
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title("MFCC Heatmap")
    plt.show()

torch>=2.0
transformers>=4.50
librosa>=0.10
numpy
matplotlib






