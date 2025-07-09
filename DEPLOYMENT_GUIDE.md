# Deployment Guide - Helmet Detection System

## Environment Setup

### Local Development

For local development:

```bash
# Install requirements
pip install -r requirements.txt

# Run the application locally
streamlit run app.py
```

### Streamlit Cloud Deployment

For deployment to Streamlit Cloud:

1. **Push your code to GitHub**
2. **Connect your repository to Streamlit Cloud**
3. **Streamlit Cloud will automatically use `requirements.txt`**

## Package Versions

The project uses flexible version requirements to ensure compatibility:

| Package    | Version | Reason              |
| ---------- | ------- | ------------------- |
| Streamlit  | Latest  | UI framework        |
| TensorFlow | Latest  | ML framework        |
| OpenCV     | Latest  | Image processing    |
| Pillow     | Latest  | Image handling      |
| NumPy      | Latest  | Numerical computing |
| Matplotlib | Latest  | Plotting            |

## Troubleshooting

### Python 3.13 Compatibility Issues

If you encounter TensorFlow compatibility errors with Python 3.13:

1. **Current Status**: The project uses flexible version requirements that should work with Python 3.13
2. **If issues persist**: Consider using the minimal app version without TensorFlow

### Local Installation Issues

If you encounter issues with local installation:

```bash
# Try with conda
conda install tensorflow
pip install -r requirements.txt

# Or create a fresh environment
conda create -n helmet-detection python=3.11
conda activate helmet-detection
pip install -r requirements.txt
```

### Cloud Deployment Issues

If Streamlit Cloud deployment fails:

1. Check that `requirements.txt` is present
2. Ensure `packages.txt` is present for system dependencies
3. Verify `.streamlit/config.toml` exists

## Quick Commands

### For Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

### For Cloud Deployment

```bash
# Simply push to GitHub and deploy on Streamlit Cloud
git add .
git commit -m "Update deployment"
git push
```

## Notes

- The project uses flexible version requirements for better compatibility
- Streamlit Cloud automatically handles Python version selection
- Keep `requirements.txt` updated with any new dependencies
- Monitor Streamlit Cloud for new supported versions
