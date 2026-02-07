"""
═══════════════════════════════════════════════════════════════════════════
              EEG Channel Expansion - Google Colab Training
═══════════════════════════════════════════════════════════════════════════
Repository: https://github.com/rusty-saber/EEG-project

ARCHITECTURE:
├─ GitHub Repository (code):
│  ├─ src/ (source code)
│  ├─ scripts/ (training scripts)
│  └─ configs/ (configuration files)
│
└─ colab_data.zip (data only):
   └─ data/processed/ (109 preprocessed .npz files)

BEFORE STARTING:
1. Runtime → Change runtime type → GPU (T4 or A100)
2. Have colab_data.zip ready on your computer (~287MB)
   Location: channel expansion/colab_data.zip

TRAINING TIME: 5-10 hours on T4 GPU
═══════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Verify GPU
# ═══════════════════════════════════════════════════════════════════════════
import torch
print("="*70)
print("Checking GPU...")
print("="*70)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"   Memory: {gpu_mem:.1f} GB")
else:
    print("❌ NO GPU DETECTED!")
    print("\nFix: Runtime → Change runtime type → GPU (T4) → Save")
    raise RuntimeError("GPU required for training")
print("="*70)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Install Dependencies (~1 minute)
# ═══════════════════════════════════════════════════════════════════════════
print("\nInstalling dependencies...")
!pip install -q torch torchvision torchaudio
!pip install -q mne transformers==4.44.0 omegaconf tqdm scipy numpy
print("✅ Dependencies installed\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Clone Repository (gets code ONLY, no data)
# ═══════════════════════════════════════════════════════════════════════════
print("="*70)
print("Cloning repository from GitHub...")
print("="*70)
!rm -rf /content/EEG-project  # Clean up if exists
!git clone https://github.com/rusty-saber/EEG-project.git /content/EEG-project
print("✅ Repository cloned")

# Verify code files exist
import os
code_check = {
    'src': os.path.isdir('/content/EEG-project/src'),
    'scripts': os.path.isdir('/content/EEG-project/scripts'),
    'configs': os.path.isdir('/content/EEG-project/configs'),
}
print("\nCode verification:")
for name, exists in code_check.items():
    status = "✅" if exists else "❌"
    print(f"  {status} {name}/")

if not all(code_check.values()):
    print("\n❌ ERROR: Repository is incomplete!")
    raise FileNotFoundError("Source code directories missing from repository")
print("="*70 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Upload Data File (colab_data.zip contains ONLY data/processed/)
# ═══════════════════════════════════════════════════════════════════════════
from google.colab import files

print("="*70)
print("UPLOAD DATA FILE")
print("="*70)
print("In the file picker:")
print("1. Navigate to: channel expansion/")
print("2. Select: colab_data.zip (~287MB)")
print("3. Click 'Open' and wait for upload (2-5 minutes)")
print("="*70)
print("\n[File picker will appear below]")
print("="*70 + "\n")

# Upload to /content (Colab's working directory)
os.chdir('/content')
uploaded = files.upload()

# Verify upload
if 'colab_data.zip' not in uploaded:
    print("\n❌ ERROR: Wrong file uploaded!")
    print("Expected: colab_data.zip")
    print(f"Got: {list(uploaded.keys())}")
    raise FileNotFoundError("Please upload colab_data.zip")

file_size_mb = len(uploaded['colab_data.zip']) / (1024*1024)
print(f"\n✅ Upload successful!")
print(f"   File: colab_data.zip")
print(f"   Size: {file_size_mb:.1f} MB")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Extract Data into Project Directory
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Extracting data...")
print("="*70)

import zipfile

try:
    # Extract zip contents to project directory
    with zipfile.ZipFile('/content/colab_data.zip', 'r') as zip_ref:
        # List contents
        file_list = zip_ref.namelist()
        print(f"Zip contains {len(file_list)} files")
        
        # Extract to project
        zip_ref.extractall('/content/EEG-project/')
        print("✅ Extraction complete")
        
except zipfile.BadZipFile:
    print("❌ ERROR: Invalid zip file!")
    raise
except Exception as e:
    print(f"❌ ERROR during extraction: {e}")
    raise

# Clean up zip file (free up space)
os.remove('/content/colab_data.zip')
print("✅ Cleanup complete")
print("="*70 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Verify Data
# ═══════════════════════════════════════════════════════════════════════════
os.chdir('/content/EEG-project')

print("="*70)
print("Verifying data integrity...")
print("="*70)

data_dir = 'data/processed'
if not os.path.exists(data_dir):
    print(f"❌ ERROR: {data_dir} not found!\n")
    print("Directory structure in data/:")
    !ls -R data/
    raise FileNotFoundError(f"{data_dir} missing")

# Count .npz files
npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
print(f"✅ Found {len(npz_files)} preprocessed subject files")

# Load statistics if available
import json
stats_file = f'{data_dir}/preprocessing_stats.json'
if os.path.exists(stats_file):
    with open(stats_file) as f:
        stats = json.load(f)
    print(f"   Total segments: {stats.get('total_segments', 'N/A')}")
    print(f"   Valid segments: {stats.get('valid_segments', 'N/A')}")
    print(f"   Rejection rate: {stats.get('rejection_rate', 'N/A'):.1f}%")

print("="*70 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Mount Google Drive (for saving checkpoints)
# ═══════════════════════════════════════════════════════════════════════════
print("="*70)
print("Mounting Google Drive...")
print("="*70)
print("You'll need to:")
print("1. Click the link that appears")
print("2. Sign in to your Google account")
print("3. Copy the authorization code")
print("4. Paste it below")
print("="*70 + "\n")

from google.colab import drive
drive.mount('/content/drive')

# Create checkpoint directory
checkpoint_base = '/content/drive/MyDrive/EEG_Checkpoints'
os.makedirs(checkpoint_base, exist_ok=True)

print(f"\n✅ Google Drive mounted")
print(f"   Checkpoints will save to: {checkpoint_base}")
print("="*70 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Start Training
# ═══════════════════════════════════════════════════════════════════════════
from datetime import datetime

# Create unique run directory
run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
output_dir = f"{checkpoint_base}/{run_name}"

print("="*70)
print("🚀 STARTING TRAINING")
print("="*70)
print(f"Run ID: {run_name}")
print(f"Epochs: 100")
print(f"Output: {output_dir}")
print(f"Estimated time: 5-10 hours (T4 GPU)")
print("="*70)
print("\n💡 TIP: You can close this tab - training continues in background!")
print("   Checkpoints save to Google Drive automatically every epoch.")
print("\n⏳ Training will start in a few seconds...")
print("="*70 + "\n")

# Run training
!python scripts/run_mvp.py \
    --epochs 100 \
    --skip_download \
    --skip_preprocess \
    --output_dir "{output_dir}"

# Training complete
print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"Results saved to: {output_dir}")
print("\nFiles in checkpoint directory:")
!ls -lh "{output_dir}"
print("="*70)

"""
═══════════════════════════════════════════════════════════════════════════
                          OPTIONAL CELLS
═══════════════════════════════════════════════════════════════════════════

CELL A: Monitor Training Progress
Run this in a NEW cell WHILE training is running to see live progress:
─────────────────────────────────────────────────────────────────────────── 
"""
# import glob
# import os
#
# # Find latest run
# runs = sorted(glob.glob('/content/drive/MyDrive/EEG_Checkpoints/run_*'))
# if runs:
#     latest = runs[-1]
#     print(f"Latest run: {latest}")
#     print("\nCheckpoint files:")
#     !ls -lh {latest}
#
#     # Show training log
#     log_path = f"{latest}/training.log"
#     if os.path.exists(log_path):
#         print("\n📊 Recent training log:")
#         !tail -30 {log_path}
# else:
#     print("No runs found yet")

"""
─────────────────────────────────────────────────────────────────────────── 
CELL B: Download Best Model
Run after training completes to download the best checkpoint:
─────────────────────────────────────────────────────────────────────────── 
"""
# from google.colab import files
# import glob
#
# runs = sorted(glob.glob('/content/drive/MyDrive/EEG_Checkpoints/run_*'))
# if runs:
#     latest = runs[-1]
#     model_path = f"{latest}/best_model.pt"
#
#     if os.path.exists(model_path):
#         print(f"Downloading: {model_path}")
#         files.download(model_path)
#         print("✅ Download complete!")
#     else:
#         print("❌ best_model.pt not found")
#         print(f"\nAvailable files in {latest}:")
#         !ls -lh {latest}
# else:
#     print("❌ No checkpoint directories found")

"""
═══════════════════════════════════════════════════════════════════════════
                          TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════

ISSUE: No GPU detected
└─ FIX: Runtime → Change runtime type → GPU → Save
       Then: Runtime → Disconnect and delete runtime → Reconnect

ISSUE: colab_data.zip upload fails
└─ FIX: Check file is exactly named: colab_data.zip (~287MB)
       Check internet connection
       Try uploading again

ISSUE: "Data directory not found" after extraction
└─ FIX: Verify colab_data.zip contains: data/processed/ folder
       Check zip has 109 .npz files
       Re-download colab_data.zip from local machine

ISSUE: Out of memory during training
└─ FIX: Edit configs/data/physionet.yaml
       Change: batch_size: 4 (reduce from 16)
       Restart from Step 8

ISSUE: Training interrupted
└─ FIX: Checkpoints are saved in Google Drive
       Check: /content/drive/MyDrive/EEG_Checkpoints/
       Find latest run_* directory
       Resume with: scripts/train.py --resume <checkpoint>

ISSUE: Repository clone fails
└─ FIX: Check internet connection
       Verify repository exists: https://github.com/rusty-saber/EEG-project
       Try again: !git clone https://github.com/rusty-saber/EEG-project.git

═══════════════════════════════════════════════════════════════════════════
                        EXPECTED RESULTS
═══════════════════════════════════════════════════════════════════════════

After 100 epochs (~5-10 hours):
  • Mean Pearson r: 0.70-0.75
  • Mean SNR: 2.5-3.0 dB
  • Best model: best_model.pt (~300MB)
  • Checkpoints: Every epoch saved to Drive

Free Colab Tier:
  • ~12 GPU hours/month
  • T4 GPU (16GB)
  • Session timeout: 12 hours (reconnects auto)

Colab Pro ($10/month):
  • 100 GPU hours/month
  • Longer sessions
  • Priority access to better GPUs (V100/A100)

═══════════════════════════════════════════════════════════════════════════
"""

print("\n✅ Notebook ready! Run Steps 1-8 in order to train.")
