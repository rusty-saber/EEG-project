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
# STEP 4: Get Data File - Choose ONE method (A or B)
# ═══════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────
# METHOD A: Upload from Computer (if you don't have it on Drive yet)
# ──────────────────────────────────────────────────────────────────────────
# from google.colab import files
# import os
#
# print("="*70)
# print("UPLOAD DATA FILE FROM COMPUTER")
# print("="*70)
# print("Select: colab_data.zip (~287MB)")
# print("Upload time: 2-5 minutes")
# print("="*70 + "\n")
#
# os.chdir('/content')
# uploaded = files.upload()
#
# if 'colab_data.zip' not in uploaded:
#     raise FileNotFoundError("Please upload colab_data.zip")
#
# file_size_mb = len(uploaded['colab_data.zip']) / (1024*1024)
# print(f"\n✅ Upload successful! Size: {file_size_mb:.1f} MB")

# ──────────────────────────────────────────────────────────────────────────
# METHOD B: Use File Already on Google Drive (RECOMMENDED - No re-upload!)
# ──────────────────────────────────────────────────────────────────────────
from google.colab import drive
import os
import shutil

# Mount Drive first
print("="*70)
print("METHOD B: Using file from Google Drive")
print("="*70)
print("Mounting Google Drive...")
drive.mount('/content/drive')
print("✅ Drive mounted\n")

# Interactive file picker for Drive
print("="*70)
print("SELECT FILE FROM GOOGLE DRIVE")
print("="*70)
print("A file browser will appear below.")
print("Navigate to your file and click to select it.")
print("="*70 + "\n")

# Use Colab's file picker for Drive
from google.colab import files
import ipywidgets as widgets
from IPython.display import display

# Create file browser
def find_zip_files(drive_path='/content/drive/MyDrive'):
    """Find all .zip files in Drive"""
    import os
    zip_files = []
    for root, dirs, filenames in os.walk(drive_path):
        for filename in filenames:
            if filename.endswith('.zip'):
                full_path = os.path.join(root, filename)
                size_mb = os.path.getsize(full_path) / (1024*1024)
                # Only show files that look like our data file (200-300 MB)
                if 200 < size_mb < 400:
                    relative = full_path.replace('/content/drive/MyDrive/', '')
                    zip_files.append((f"{relative} ({size_mb:.1f} MB)", full_path))
    return zip_files

print("Scanning Google Drive for zip files (200-400 MB)...\n")
zip_files = find_zip_files()

if not zip_files:
    print("❌ No suitable zip files found in Google Drive!")
    print("\nOptions:")
    print("1. Upload colab_data.zip to Google Drive first")
    print("2. OR use METHOD A above to upload from computer")
    raise FileNotFoundError("No zip files found in Drive")

# Create dropdown with found files
print(f"Found {len(zip_files)} zip file(s):\n")
dropdown = widgets.Dropdown(
    options=zip_files,
    description='Select file:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='80%')
)
display(dropdown)

print("\n👆 Select your file from the dropdown above, then run the next cell")
print("="*70)

# Store selection for next cell
selected_file_path = dropdown.value

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Copy & Extract Data (works with both METHOD A and METHOD B)
# ═══════════════════════════════════════════════════════════════════════════
import os
import zipfile
import shutil

print("\n" + "="*70)
print("Processing data file...")
print("="*70)

# Determine source file location
if 'selected_file_path' in globals():
    # METHOD B: File from Drive
    source_file = selected_file_path
    print(f"Source: {source_file}")
    
    # Copy to /content for faster extraction
    print("Copying from Drive to local workspace...")
    shutil.copy(source_file, '/content/colab_data.zip')
    print("✅ Copy complete")
    
elif os.path.exists('/content/colab_data.zip'):
    # METHOD A: Already uploaded
    source_file = '/content/colab_data.zip'
    print(f"Source: Uploaded file")
    
else:
    print("❌ ERROR: No data file found!")
    print("Please run METHOD A or METHOD B in Step 4 first")
    raise FileNotFoundError("colab_data.zip not found")

# Verify file integrity
print("\nVerifying file...")
file_size = os.path.getsize('/content/colab_data.zip')
file_size_mb = file_size / (1024*1024)
print(f"File size: {file_size_mb:.1f} MB")

# Check if it's actually a zip file (magic number check)
with open('/content/colab_data.zip', 'rb') as f:
    header = f.read(4)
    is_zip = header[:2] == b'PK'  # ZIP files start with 'PK'
    print(f"ZIP magic number: {'✅ Valid' if is_zip else '❌ Invalid'}")

if not is_zip:
    print("\n❌ ERROR: File is not a valid ZIP file!")
    print("\nPossible causes:")
    print("1. File got corrupted during upload to Google Drive")
    print("2. Google Drive is still processing the file (try waiting a minute)")
    print("3. Wrong file was selected")
    print("\n🔧 SOLUTIONS:")
    print("Option 1: Re-upload colab_data.zip to Google Drive")
    print("Option 2: Use METHOD A to upload directly from your computer")
    print("Option 3: Check the file on your local machine is valid (try opening it)")
    raise ValueError("Invalid ZIP file")

# Extract zip contents
print("\nExtracting data...")
try:
    # Try Python's zipfile first
    with zipfile.ZipFile('/content/colab_data.zip', 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"Zip contains {len(file_list)} files")
        
        # Extract to project directory
        zip_ref.extractall('/content/EEG-project/')
        print("✅ Extraction complete")
        
except zipfile.BadZipFile as e:
    print("❌ Python zipfile failed, trying system unzip command...")
    
    # Fallback: Try system unzip command (more robust)
    result = !unzip -q /content/colab_data.zip -d /content/EEG-project/ 2>&1
    
    if any('error' in line.lower() or 'invalid' in line.lower() for line in result):
        print("❌ System unzip also failed!")
        print("\nUnzip output:")
        for line in result:
            print(f"  {line}")
        
        print("\n🔧 The file appears to be corrupted.")
        print("\nRECOMMENDED FIX:")
        print("1. On your LOCAL computer, verify colab_data.zip opens correctly")
        print("2. Delete the file from Google Drive")
        print("3. Re-upload colab_data.zip to Google Drive")
        print("4. OR use METHOD A to upload directly from your computer")
        raise
    else:
        print("✅ Extraction successful using system unzip")
        
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise

# Clean up (free space)
if os.path.exists('/content/colab_data.zip'):
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
