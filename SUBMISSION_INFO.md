# Enhanced Michael Agent - Submission Information

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Team Hash
Edit `submit_run.py` and replace `TEAM_HASH` with your actual team hash:
```python
TEAM_HASH = "YOUR_TEAM_HASH_HERE"
```

### 3. Run Submission
```bash
python submit_run.py
```

## Agent Details

- **Model**: HuggingFace Qwen2.5-0.5B
- **Track**: Social Detection (SecretMafia-v0)
- **Features**: Strategy Experience Pool with 158 pre-trained experiences
- **Learning**: Enabled - learns from successful strategies during gameplay

## System Requirements

- Python 3.8+
- 4GB+ RAM recommended
- 2GB+ VRAM for GPU acceleration
- Internet connection for initial model download

## Competition Compliance

✅ Fixed requirements.txt with package versions
✅ Single entry point (submit_run.py)
✅ HuggingFace model integration
✅ Comprehensive documentation

## Support

See README.md for detailed documentation and troubleshooting.