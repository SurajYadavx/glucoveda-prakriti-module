# save as test_install.py in your project root
import mediapipe as mp
import cv2
import torch
import librosa
import fastapi
import numpy as np

print("mediapipe:", mp.__version__)
print("opencv:", cv2.__version__)
print("torch:", torch.__version__)
print("librosa:", librosa.__version__)
print("fastapi:", fastapi.__version__)
print("CUDA available:", torch.cuda.is_available())
print("All good ✅")