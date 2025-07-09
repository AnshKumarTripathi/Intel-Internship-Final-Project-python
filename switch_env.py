#!/usr/bin/env python3
"""
Environment switcher for Helmet Detection System
Helps switch between local and cloud requirements
"""

import os
import shutil
import sys

def switch_to_local():
    """Switch to local development requirements"""
    try:
        if os.path.exists('requirements_local.txt'):
            shutil.copy('requirements_local.txt', 'requirements.txt')
            print("✅ Switched to LOCAL requirements")
            print("📦 Use: pip install -r requirements.txt")
            print("🚀 Use: streamlit run app.py")
        else:
            print("❌ requirements_local.txt not found")
    except Exception as e:
        print(f"❌ Error: {e}")

def switch_to_cloud():
    """Switch to cloud deployment requirements"""
    try:
        if os.path.exists('requirements_cloud.txt'):
            shutil.copy('requirements_cloud.txt', 'requirements.txt')
            print("✅ Switched to CLOUD requirements")
            print("☁️  Ready for Streamlit Cloud deployment")
            print("📝 Don't forget to commit and push!")
        else:
            print("❌ requirements_cloud.txt not found")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_status():
    """Show current environment status"""
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        if 'tensorflow>=2.16.0' in content:
            print("🖥️  Current: LOCAL environment")
        elif 'tensorflow==2.14.0' in content:
            print("☁️  Current: CLOUD environment")
        else:
            print("❓ Current: UNKNOWN environment")
            
        print("\n📋 Current requirements.txt:")
        print(content)
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")

def main():
    """Main function"""
    print("🪖 Helmet Detection System - Environment Switcher")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python switch_env.py local   - Switch to local development")
        print("  python switch_env.py cloud   - Switch to cloud deployment")
        print("  python switch_env.py status  - Show current status")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'local':
        switch_to_local()
    elif command == 'cloud':
        switch_to_cloud()
    elif command == 'status':
        show_status()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: local, cloud, or status")

if __name__ == "__main__":
    main() 