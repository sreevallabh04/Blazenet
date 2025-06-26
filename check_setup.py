"""
Environment Setup and Validation Script
"""

import os
import sys
import subprocess
import pkg_resources

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(" Python 3.8+ required. Current version:", sys.version)
        return False
    print(f" Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_required_packages():
    """Check if all required packages are installed."""
    required_packages = [
        "torch", "fastapi", "streamlit", "rasterio", 
        "psycopg2", "requests", "pandas", "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.get_distribution(package)
            print(f" {package}")
        except pkg_resources.DistributionNotFound:
            print(f" {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n Install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f" Docker: {result.stdout.strip()}")
            return True
        else:
            print(" Docker not available")
            return False
    except:
        print(" Docker not available")
        return False

def check_directories():
    """Check if all required directories exist."""
    required_dirs = [
        "app/backend", "app/frontend", "app/ml", 
        "data/sample", "data/raw", "data/processed",
        "logs", "app/ml/weights"
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f" {directory}")
        else:
            print(f" {directory} - Missing")
            all_exist = False
    
    return all_exist

def check_config():
    """Check configuration file."""
    if os.path.exists("config.env"):
        print(" config.env")
        
        # Check for NASA API key
        with open("config.env", "r") as f:
            content = f.read()
            if "NASA_FIRMS_API_KEY" in content and "904187de8a6aa5475740a5799d207041" in content:
                print(" NASA FIRMS API Key configured")
            else:
                print(" NASA FIRMS API Key not configured")
        
        return True
    else:
        print(" config.env - Missing")
        return False

def main():
    """Main validation function."""
    print(" BlazeNet Environment Validation")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Docker", check_docker),
        ("Directories", check_directories),
        ("Configuration", check_config)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print(" All checks passed! BlazeNet is ready to run.")
        print(" Start with: python start_blazenet.py")
    else:
        print(" Some checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()

