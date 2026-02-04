"""
REVE Channel Expansion - Google Colab Training
================================================

1. First, upload this notebook and the colab_package.zip to Colab
2. Run all cells in order
3. Training will use GPU automatically

Mount Google Drive (optional - for saving checkpoints):
>>> from google.colab import drive
>>> drive.mount('/content/drive')
"""

# Cell 1: Setup - Install Dependencies
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install mne transformers==4.44.0 omegaconf tqdm scipy numpy

# Cell 2: Extract the packages
# !unzip -q colab_package_code.zip -d /content/channel_expansion
# !unzip -q colab_data.zip -d /content/channel_expansion

# Cell 3: Change to project directory  
# %cd /content/channel_expansion

# Cell 4: Check GPU
# import torch
# print(f"GPU available: {torch.cuda.is_available()}")
# print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Cell 5: Run Training (100 epochs, GPU)
# !python scripts/run_mvp.py --epochs 100 --skip_download --skip_preprocess --output_dir outputs/colab_run

# Alternative: Run with specific batch size for memory management
# !python scripts/run_mvp.py --epochs 100 --batch_size 8 --skip_download --skip_preprocess

print("Colab setup instructions created!")
