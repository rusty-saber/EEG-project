# ============================================================
# REVE Channel Expansion - Google Colab Training Setup
# ============================================================
# Repository: https://github.com/rusty-saber/EEG-project
# ============================================================

# =============================================================================
# CELL 1: Mount Google Drive (for data and checkpoints)
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

# Create directories in your Google Drive
!mkdir -p /content/drive/MyDrive/EEG_Channel_Expansion/data
!mkdir -p /content/drive/MyDrive/EEG_Channel_Expansion/checkpoints

# =============================================================================
# CELL 2: Clone Repository
# =============================================================================
!git clone https://github.com/rusty-saber/EEG-project.git
%cd EEG-project

# =============================================================================
# CELL 3: Install Dependencies
# =============================================================================
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q mne transformers==4.44.0 omegaconf tqdm scipy numpy

# =============================================================================
# CELL 4: Verify GPU
# =============================================================================
import torch
print("="*60)
print("GPU Status:")
print("="*60)
print(f"✅ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🚀 GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU detected! Go to Runtime > Change runtime type > Select GPU")
print("="*60)

# =============================================================================
# CELL 5A: Upload Preprocessed Data to Google Drive (ONE-TIME SETUP)
# =============================================================================
# INSTRUCTIONS:
# 1. On your local machine, locate: channel expansion/colab_data.zip (~287MB)
# 2. Upload it to: Google Drive > EEG_Channel_Expansion/data/
# 3. Then run this cell to extract it:

!unzip -q /content/drive/MyDrive/EEG_Channel_Expansion/data/colab_data.zip -d /content/EEG-project/
print("✅ Data extracted successfully!")

# =============================================================================
# CELL 5B: OR Run Preprocessing in Colab (Alternative - ~2 minutes)
# =============================================================================
# If you haven't uploaded data, you can preprocess in Colab instead:
# (Note: This requires downloading raw data first, which takes time)

# Uncomment if you want to preprocess in Colab:
# !python scripts/download_data.py --dataset physionet --num_subjects 109
# !python scripts/preprocess.py \
#     --raw_dir "data/raw/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0" \
#     --output_dir data/processed \
#     --config configs/data/physionet.yaml \
#     --num_subjects 109

# =============================================================================
# CELL 6: Verify Data is Ready
# =============================================================================
import os
processed_dir = "data/processed"
if os.path.exists(processed_dir):
    num_files = len([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    print(f"✅ Found {num_files} preprocessed subject files")
else:
    print("❌ Data not found! Please run Cell 5A or 5B first")

# =============================================================================
# CELL 7: Start Training (Saves checkpoints to Google Drive)
# =============================================================================
!python scripts/run_mvp.py \
    --epochs 100 \
    --skip_download \
    --skip_preprocess \
    --output_dir /content/drive/MyDrive/EEG_Channel_Expansion/checkpoints/run_$(date +%Y%m%d_%H%M%S)

# =============================================================================
# CELL 8: Monitor Training (Optional - run in separate cell)
# =============================================================================
# You can monitor the latest checkpoint:
# !ls -lht /content/drive/MyDrive/EEG_Channel_Expansion/checkpoints/

# =============================================================================
# CELL 9: Resume Training from Checkpoint (if interrupted)
# =============================================================================
# If training gets interrupted, you can resume:
# !python scripts/train.py \
#     --config configs/data/physionet.yaml \
#     --resume /content/drive/MyDrive/EEG_Channel_Expansion/checkpoints/LATEST_RUN/best_model.pt \
#     --output_dir /content/drive/MyDrive/EEG_Channel_Expansion/checkpoints/resumed_run

print("✅ Colab setup complete! Ready to train.")
