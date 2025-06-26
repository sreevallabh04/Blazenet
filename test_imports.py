#!/usr/bin/env python3
"""
Test script to check if all production system modules can be imported successfully.
"""

import sys
import traceback
from pathlib import Path

def test_import(module_name, description):
    """Test importing a module and report success/failure."""
    try:
        exec(f"import {module_name}")
        print(f"✅ {description}: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ {description}: FAILED - {str(e)}")
        traceback.print_exc()
        return False

def test_backend_imports():
    """Test backend module imports."""
    print("🧠 Testing Backend Module Imports...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Test ML models
    total_tests += 1
    if test_import("backend.core.ml_models", "ML Models Module"):
        success_count += 1
        
        # Test specific classes
        try:
            from backend.core.ml_models import ProductionMLPipeline, get_ml_pipeline
            print("  ✅ ProductionMLPipeline class imported")
            print("  ✅ get_ml_pipeline function imported")
        except Exception as e:
            print(f"  ❌ ML classes failed: {str(e)}")
    
    # Test data processor
    total_tests += 1
    if test_import("backend.core.data_processor", "Data Processor Module"):
        success_count += 1
        
        try:
            from backend.core.data_processor import ISROSatelliteDataProcessor, get_data_processor
            print("  ✅ ISROSatelliteDataProcessor class imported")
            print("  ✅ get_data_processor function imported")
        except Exception as e:
            print(f"  ❌ Data processor classes failed: {str(e)}")
    
    # Test API
    total_tests += 1
    if test_import("backend.api.production_api", "Production API Module"):
        success_count += 1
        
        try:
            from backend.api.production_api import app
            print("  ✅ FastAPI app imported")
        except Exception as e:
            print(f"  ❌ API app failed: {str(e)}")
    
    # Test database
    total_tests += 1
    if test_import("backend.database.production_db", "Production Database Module"):
        success_count += 1
        
        try:
            from backend.database.production_db import ProductionDatabase, get_production_db
            print("  ✅ ProductionDatabase class imported")
            print("  ✅ get_production_db function imported")
        except Exception as e:
            print(f"  ❌ Database classes failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Backend Import Results: {success_count}/{total_tests} modules imported successfully")
    
    return success_count == total_tests

def test_main_system():
    """Test main system components."""
    print("\n🖥️ Testing Main System Components...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Test production system (without Streamlit context)
    total_tests += 1
    try:
        # We can't fully import production_system due to Streamlit, but we can check syntax
        with open("production_system.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic syntax check
        compile(content, "production_system.py", "exec")
        print("✅ Production System: Syntax valid")
        success_count += 1
    except UnicodeDecodeError:
        try:
            # Try with different encoding
            with open("production_system.py", 'r', encoding='latin-1') as f:
                content = f.read()
            compile(content, "production_system.py", "exec")
            print("✅ Production System: Syntax valid (latin-1 encoding)")
            success_count += 1
        except Exception as e:
            print(f"❌ Production System: Encoding error - {str(e)}")
    except Exception as e:
        print(f"❌ Production System: {str(e)}")
    
    # Test system launcher
    total_tests += 1
    if test_import("start_production_system", "System Launcher"):
        success_count += 1
    
    # Test validator
    total_tests += 1
    if test_import("validate_production_system", "System Validator"):
        success_count += 1
    
    # Test demo launcher
    total_tests += 1
    if test_import("launch_demo", "Demo Launcher"):
        success_count += 1
    
    print(f"Main System Results: {success_count}/{total_tests} components valid")
    
    return success_count == total_tests

def test_dependencies():
    """Test required dependencies."""
    print("\n📦 Testing Required Dependencies...")
    print("=" * 50)
    
    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
        ("folium", "Folium"),
        ("requests", "Requests"),
        ("asyncio", "AsyncIO"),
        ("pathlib", "PathLib")
    ]
    
    success_count = 0
    
    for package, name in required_packages:
        if test_import(package, name):
            success_count += 1
    
    print(f"Dependencies Results: {success_count}/{len(required_packages)} packages available")
    
    return success_count == len(required_packages)

def main():
    """Run all import tests."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║               🔍 ISRO AGNIRISHI IMPORT TESTS 🔍                   ║
    ║                  Production System Validation                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all tests
    backend_ok = test_backend_imports()
    system_ok = test_main_system()
    deps_ok = test_dependencies()
    
    # Overall results
    print("\n" + "=" * 70)
    print("🎯 OVERALL TEST RESULTS")
    print("=" * 70)
    
    print(f"🧠 Backend Modules: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"🖥️ Main System: {'✅ PASS' if system_ok else '❌ FAIL'}")
    print(f"📦 Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    
    overall_status = backend_ok and system_ok and deps_ok
    print(f"\n🏆 OVERALL STATUS: {'✅ ALL TESTS PASSED' if overall_status else '❌ SOME TESTS FAILED'}")
    
    if overall_status:
        print("\n🚀 SYSTEM READY FOR LAUNCH!")
        print("   Run: python launch_demo.py")
        print("   Or:  streamlit run production_system.py")
    else:
        print("\n🔧 SYSTEM NEEDS FIXES BEFORE LAUNCH")
        print("   Check error messages above for details")
    
    return 0 if overall_status else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 