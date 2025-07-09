#!/usr/bin/env python3
"""
Deployment script for Helmet Detection System
Helps switch between different configurations for different deployment scenarios
"""

import os
import shutil
import sys

def deploy_minimal():
    """Deploy minimal version without TensorFlow"""
    try:
        # Copy minimal requirements
        shutil.copy('requirements_minimal.txt', 'requirements.txt')
        
        # Copy minimal app
        shutil.copy('app_minimal.py', 'app.py')
        
        print("✅ Deployed MINIMAL version (no TensorFlow)")
        print("📦 Uses: requirements_minimal.txt")
        print("🚀 App: app_minimal.py")
        print("⚠️  Note: ML functionality temporarily disabled")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def deploy_flexible():
    """Deploy with flexible version requirements"""
    try:
        # Copy flexible requirements
        shutil.copy('requirements_flexible_cloud.txt', 'requirements.txt')
        
        print("✅ Deployed FLEXIBLE version")
        print("📦 Uses: requirements_flexible_cloud.txt")
        print("🚀 App: app.py (full version)")
        print("🎯 Note: Package manager will resolve compatible versions")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def deploy_latest():
    """Deploy with latest version requirements"""
    try:
        # Copy latest requirements
        shutil.copy('requirements_latest.txt', 'requirements.txt')
        
        print("✅ Deployed LATEST version")
        print("📦 Uses: requirements_latest.txt")
        print("🚀 App: app.py (full version)")
        print("🎯 Note: Uses latest compatible versions")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def deploy_tensorflow():
    """Deploy with TensorFlow (for compatible environments)"""
    try:
        # Copy TensorFlow requirements
        shutil.copy('requirements_local.txt', 'requirements.txt')
        
        # Copy full app
        shutil.copy('app.py', 'app_full.py')
        shutil.copy('app_minimal.py', 'app.py')
        
        print("✅ Deployed TENSORFLOW version")
        print("📦 Uses: requirements_local.txt")
        print("🚀 App: app.py (full version)")
        print("🎯 Note: Full ML functionality available")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def deploy_onnx():
    """Deploy with ONNX Runtime"""
    try:
        # Copy ONNX requirements
        shutil.copy('requirements_onnx.txt', 'requirements.txt')
        
        print("✅ Deployed ONNX version")
        print("📦 Uses: requirements_onnx.txt")
        print("🚀 App: app.py (needs ONNX model)")
        print("🎯 Note: Requires model conversion to ONNX format")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def show_status():
    """Show current deployment status"""
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        if 'tensorflow' in content:
            print("🖥️  Current: TENSORFLOW deployment")
        elif 'onnxruntime' in content:
            print("☁️  Current: ONNX deployment")
        else:
            print("📱 Current: MINIMAL deployment")
            
        print("\n📋 Current requirements.txt:")
        print(content)
        
        # Check if app.py exists
        if os.path.exists('app.py'):
            print("\n✅ app.py found")
        else:
            print("\n❌ app.py not found")
            
    except Exception as e:
        print(f"❌ Error reading status: {e}")

def main():
    """Main function"""
    print("🪖 Helmet Detection System - Deployment Manager")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python deploy.py minimal    - Deploy minimal version (no TensorFlow)")
        print("  python deploy.py flexible   - Deploy with flexible versions")
        print("  python deploy.py latest     - Deploy with latest versions")
        print("  python deploy.py tensorflow - Deploy with TensorFlow")
        print("  python deploy.py onnx       - Deploy with ONNX Runtime")
        print("  python deploy.py status     - Show current status")
        print("\nRecommended for Python 3.13:")
        print("  python deploy.py flexible")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'minimal':
        deploy_minimal()
    elif command == 'flexible':
        deploy_flexible()
    elif command == 'latest':
        deploy_latest()
    elif command == 'tensorflow':
        deploy_tensorflow()
    elif command == 'onnx':
        deploy_onnx()
    elif command == 'status':
        show_status()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: minimal, flexible, latest, tensorflow, onnx, or status")

if __name__ == "__main__":
    main() 