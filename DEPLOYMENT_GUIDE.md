# Deployment Guide - Helmet Detection System

## Environment Setup

### Local Development

For local development with the latest package versions:

```bash
# Install local requirements
pip install -r requirements_local.txt

# Run the application locally
streamlit run app.py
```

### Streamlit Cloud Deployment

For deployment to Streamlit Cloud:

1. **Rename the cloud requirements file:**

   ```bash
   # Rename cloud requirements to the standard name
   mv requirements_cloud.txt requirements.txt
   ```

2. **Deploy to Streamlit Cloud:**

   - Push your code to GitHub
   - Connect your repository to Streamlit Cloud
   - Streamlit Cloud will automatically use `requirements.txt`

3. **After deployment, restore local requirements:**
   ```bash
   # Restore local requirements for continued development
   mv requirements.txt requirements_cloud.txt
   mv requirements_local.txt requirements.txt
   ```

## Package Version Differences

| Package    | Local Version | Cloud Version | Reason                    |
| ---------- | ------------- | ------------- | ------------------------- |
| TensorFlow | ==2.16.1      | ==2.14.0      | Python 3.13 compatibility |
| OpenCV     | ==4.8.1.78    | ==4.8.1.78    | System dependencies       |
| Pillow     | ==10.0.1      | ==10.0.1      | Stability                 |
| NumPy      | ==1.24.3      | ==1.24.3      | Compatibility             |
| Matplotlib | ==3.7.2       | ==3.7.2       | Consistency               |

## Troubleshooting

### Local Installation Issues

If you encounter issues with local installation:

```bash
# Try with conda
conda install tensorflow
pip install -r requirements_local.txt

# Or use the flexible requirements
pip install -r requirements_flexible.txt
```

### Cloud Deployment Issues

If Streamlit Cloud deployment fails:

1. Check that `requirements.txt` contains cloud-compatible versions
2. Ensure `packages.txt` is present for system dependencies
3. Verify `.streamlit/config.toml` exists

### Python 3.13 Compatibility Issues

If you encounter TensorFlow compatibility errors with Python 3.13:

1. **Use TensorFlow 2.14.0** (recommended for Python 3.13):

   ```bash
   # Use the Python 3.13 specific requirements
   cp requirements_python313.txt requirements.txt
   ```

2. **Alternative: Use TensorFlow 2.15.0**:

   ```bash
   # Use the flexible requirements
   cp requirements_flexible.txt requirements.txt
   ```

3. **Error Message**: `No matching distribution found for tensorflow==2.16.1`
   - This occurs because TensorFlow 2.16.1 doesn't have wheels for Python 3.13
   - Solution: Use TensorFlow 2.14.0 or 2.15.0 instead

### Version Conflicts

If you get version conflicts:

```bash
# Create a fresh environment
conda create -n helmet-detection python=3.9
conda activate helmet-detection
pip install -r requirements_local.txt
```

## Quick Commands

### For Local Development

```bash
pip install -r requirements_local.txt
streamlit run app.py
```

### For Cloud Deployment

```bash
# Temporarily use cloud requirements
cp requirements_cloud.txt requirements.txt
# Deploy to Streamlit Cloud
# Then restore local requirements
cp requirements_local.txt requirements.txt
```

## Notes

- Always test locally before deploying
- Keep both requirements files in version control
- Update versions periodically for security patches
- Monitor Streamlit Cloud for new supported versions
