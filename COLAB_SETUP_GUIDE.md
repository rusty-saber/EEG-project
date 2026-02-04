# Google Colab Setup Guide for EEG Channel Expansion

## Quick Start (3 Steps)

### 1. Upload Data to Google Drive
1. On your computer, find: `channel expansion/colab_data.zip` (~287MB)
2. Go to [Google Drive](https://drive.google.com)
3. Create folder: `EEG_Channel_Expansion/data/`
4. Upload `colab_data.zip` to that folder

### 2. Open Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy-paste cells from `colab_notebook.py` into Colab
4. **Important**: Set GPU runtime
   - Menu: `Runtime > Change runtime type`
   - Hardware accelerator: **GPU (T4 or A100)**
   - Click Save

### 3. Run Training
- Run all cells in order (Shift+Enter)
- Training will save checkpoints to your Google Drive automatically
- You can close the browser - training continues in background!

## Expected Timeline
- **Setup**: 5-10 minutes (one-time)
- **Training**: 5-10 hours on T4 GPU (vs 100+ hours on CPU)
- **Checkpoints**: Saved every epoch to Google Drive

## Troubleshooting

### "No GPU detected"
- Go to `Runtime > Change runtime type > GPU`
- Disconnect and reconnect: `Runtime > Disconnect and delete runtime`

### "Data not found"
- Make sure `colab_data.zip` is uploaded to Google Drive
- Check the path in Cell 5A matches your Drive location

### Training interrupted?
- Your checkpoints are safe in Google Drive
- Use Cell 9 to resume from latest checkpoint

## What Gets Saved to Google Drive?
```
EEG_Channel_Expansion/
├── data/
│   └── colab_data.zip (you upload this)
└── checkpoints/
    └── run_YYYYMMDD_HHMMSS/
        ├── best_model.pt
        ├── checkpoint_epoch_50.pt
        ├── training_log.txt
        └── metrics.json
```

## Cost Estimate
- Google Colab Free: 12-15 hours GPU/month (enough for 1-2 full runs)
- Colab Pro ($10/month): 100 GPU hours (enough for 10+ runs)
- Colab Pro+: Priority access to A100 GPUs (2-3x faster)

## Next Steps After Training
1. Download best checkpoint from Google Drive
2. Run evaluation: `python scripts/evaluate.py --checkpoint best_model.pt`
3. Visualize results in notebooks
