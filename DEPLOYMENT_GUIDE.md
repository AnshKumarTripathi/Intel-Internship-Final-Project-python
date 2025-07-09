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

| Package    | Local Version | Cloud Version | Reason              |
| ---------- | ------------- | ------------- | ------------------- |
| TensorFlow | >=2.16.0      | ==2.14.0      | Cloud compatibility |
| OpenCV     | >=4.8.0       | ==4.8.0.76    | System dependencies |
| Pillow     | >=10.0.0      | ==9.5.0       | Stability           |
| NumPy      | >=1.24.0      | ==1.23.5      | Compatibility       |
| Matplotlib | >=3.7.0       | ==3.7.1       | Consistency         |

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
