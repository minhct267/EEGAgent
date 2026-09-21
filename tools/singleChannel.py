import os
import torch
import torch.nn as nn
from typing import List
from .localModels.net import RoPETransformer
from .registerData import getRegisteredData
from .polar import index2name, name2index
from .register import function_register


def _unique_sorted_channel_ids(name: List[str]):
    # Unknown names still KeyError into the existing execution-error path.
    ids = sorted({name2index[item] for item in name})
    return ids, [index2name[i] for i in ids]


class SingleChannelEEG(nn.Module):
    def __init__(self, cls, embed_dim=128, num_heads=4, max_len=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.MaxPool1d(2, 2),  # -> 128
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.MaxPool1d(2, 2),  # -> 64
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.MaxPool1d(2, 2),  # -> 32
            nn.Conv1d(64, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.LayerNorm(32)
        )

        self.attn_layers = nn.ModuleList([
            RoPETransformer(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(2)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, embed_dim))
        self.fc = nn.Linear(embed_dim, cls)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)  # [B, C=embed_dim, T']
        x = x.transpose(1, 2)  # [B, T', C]

        for layer in self.attn_layers:
            x, _ = layer(x)  # v = x

        x = self.pool(x).squeeze(1)  # [B, C]
        x = self.fc(x)
        return x

device = 'cpu'
model_musle_eyem = SingleChannelEEG(cls=2)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "localModels", "muscle&eyem.pth")
model_musle_eyem.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))  # Remap saved tensors onto `device`.
model_musle_eyem.eval()

@function_register.register(
    description=(
        "Binary artifact-TYPE classifier: Eye movement vs Muscle only. "
        "Call it only after another tool has already flagged artifact, when you need to know "
        "whether that artifact looks like eye movement or muscle. "
        "It is NOT a clean-vs-artifact or signal-quality detector. "
        "The two class probabilities always sum to about 1.0; there is no 'clean' class. "
        "A high Muscle probability does NOT mean the EEG is unusable and must NOT be used "
        "to discard seizure or epileptiform findings from other tools. "
        "To decide seizure vs artifact vs background, call seizureArtiBckgModel_OneSecond instead. "
        "Channels must be bipolar (e.g., 'FP1-F7'). Duration must be 1-10 seconds."
    ),
    parameters=[
        {
            "name": "name",
            "type": "List[str]",
            "description": (
                "List of EEG channel pairs to analyze (e.g., ['FP1-F7', 'F7-T3']). "
                "If the user mentions a single channel like 'FP1–F7', wrap it in a list."
            )
        },
        {
            "name": "start",
            "type": "int",
            "description": (
                "Start time in seconds. Extract this from user query, e.g., 'from 10s'."
            )
        },
        {
            "name": "end",
            "type": "int",
            "description": (
                "End time in seconds. Extract this from user query, e.g., 'to 11s'. "
                "The interval should typically be 1 second."
            )
        }
    ],
    returns={
        "type": "List[Dict]",
        "description": (
            "One dict per second: duration plus per-channel "
            "{'Eye movement': p, 'Muscle artifact': 1-p}. "
            "These two numbers always compete with each other; they are not contamination rates."
        )
    }
)
def eyemMuscleModel_OneSecond(name: List[str], start:int, end:int, config):
    if end - start > 10:
        raise ValueError("The time interval between start and end should not exceed 10 seconds.")
    
    fs = config['fs']
    N = end - start
    data = getRegisteredData(start, end, config)
    ids, chs = _unique_sorted_channel_ids(name)
    infos = []
    for i in range(N):
        start_idx = i * fs
        end_idx = (i + 1) * fs
        x = data[ids, start_idx:end_idx]
        info = {}
        info['duration'] = f"{start + i}s-{start + (i + 1)}s"

        data_tensor = torch.tensor(x, dtype=torch.float32).to(device)  # (1, T)
        logits = model_musle_eyem(data_tensor)
        probs = torch.softmax(logits, dim=-1)

        for j, v in enumerate(chs):
            info[v] = {
                "Eye movement": round(probs[j, 0].item(), 2),
                "Muscle artifact": round(probs[j, 1].item(), 2),
            }
        infos.append(info)
    return infos

model_seiz_arti_bckg = SingleChannelEEG(cls=3)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "localModels", "seiz&arti&bckg.pth")
model_seiz_arti_bckg.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))  # Remap saved tensors onto `device`.
model_seiz_arti_bckg.eval()

@function_register.register(
    description=(
        "Fine-grained 1-second 3-class classifier per channel: background (bckg), "
        "artifact, or seizure (seiz). This is the tool that can separate seizure from artifact. "
        "Use it when a coarse seizure score is high and you need to test whether the second "
        "is artifact-driven. Do not substitute eyemMuscleModel_OneSecond for this decision. "
        "Duration must be 1-10 seconds."
    ),
    parameters=[
        {
            "name": "name",
            "type": "List[str]",
            "description": (
                "List of EEG channel pairs to analyze (e.g., ['FP1-F7', 'C3-P3']). "
                "Supports reversed order like 'F7-FP1'."
            )
        },
        {
            "name": "start",
            "type": "int",
            "description": "Start time in seconds (e.g., 20)."
        },
        {
            "name": "end",
            "type": "int",
            "description": (
                "End time in seconds (e.g., 21). "
                "The duration must not exceed 10 seconds; 1 second is recommended."
            )
        }
    ],
    returns={
        "type": "List[Dict]",
        "description": (
            "One dict per second: duration plus per-channel "
            "{'bckg': p, 'artifact': p, 'seiz': p} (softmax, sum ~ 1). "
            "High seiz supports reporting a discharge on that channel and second."
        )
    }
)
def seizureArtiBckgModel_OneSecond(name: List[str], start:int, end:int, config):
    if end - start > 10:
        raise ValueError("The time interval between start and end should not exceed 10 seconds.")
    
    fs = config['fs']
    N = end - start
    data = getRegisteredData(start, end, config)
    ids, chs = _unique_sorted_channel_ids(name)
    infos = []
    for i in range(N):
        start_idx = i * fs
        end_idx = (i + 1) * fs
        x = data[ids, start_idx:end_idx]
        info = {}
        info['duration'] = f"{start + i}s-{start + (i + 1)}s"

        data_tensor = torch.tensor(x, dtype=torch.float32).to(device)  # (1, T)
        logits = model_seiz_arti_bckg(data_tensor)
        probs = torch.softmax(logits, dim=-1)

        for j, v in enumerate(chs):
            info[v] = {
                "bckg": round(probs[j, 0].item(),2),
                "artifact": round(probs[j, 1].item(),2),
                "seiz": round(probs[j, 2].item(),2)
            }
        infos.append(info)
    return infos


model_seiz_normal = SingleChannelEEG(cls=2)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "localModels", "seiz&normal.pth")
model_seiz_normal.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))  # Remap saved tensors onto `device`.
model_seiz_normal.eval()

@function_register.register(
    description=(
        "Fine-grained 1-second 2-class classifier per channel: Non-seiz vs seiz. "
        "Use it to localize epileptic activity in time and channel after a coarse window looks abnormal. "
        "It does not label artifact type. If you must test artifact vs seizure, "
        "also call seizureArtiBckgModel_OneSecond. Duration must be 1-10 seconds."
    ),
    parameters=[
        {
            "name": "name",
            "type": "List[str]",
            "description": "Channels to analyze, e.g., ['FP1-F7', 'F7-T3']"
        },
        {
            "name": "start",
            "type": "int",
            "description": "Start time in seconds."
        },
        {
            "name": "end",
            "type": "int",
            "description": "End time in seconds. Maximum allowed interval: 10 seconds."
        }
    ],
    returns={
        "type": "List[Dict]",
        "description": (
            "One dict per second: duration plus per-channel "
            "{'Non-seiz': p, 'seiz': p}. High seiz is evidence of a discharge on that channel and second."
        )
    }
)
def seizureNormalModel_OneSecond(name: List[str], start:int, end:int, config):
    if end - start > 10:
        raise ValueError("The time interval between start and end should not exceed 10 seconds.")
    
    fs = config['fs']
    N = end - start
    data = getRegisteredData(start, end, config)
    ids, chs = _unique_sorted_channel_ids(name)
    infos = []
    for i in range(N):
        start_idx = i * fs
        end_idx = (i + 1) * fs
        x = data[ids, start_idx:end_idx]
        info = {}
        info['duration'] = f"{start + i}s-{start + (i + 1)}s"

        data_tensor = torch.tensor(x, dtype=torch.float32).to(device)  # (1, T)
        logits = model_seiz_normal(data_tensor)
        probs = torch.softmax(logits, dim=-1)

        for j, v in enumerate(chs):
            info[v] = {
                "Non-seiz": round(probs[j, 0].item(),2),
                "seiz": round(probs[j, 1].item(),2)
            }
        infos.append(info)
    return infos
