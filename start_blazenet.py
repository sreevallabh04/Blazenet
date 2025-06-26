"""
BlazeNet Startup Script
Launches all services in the correct order for development.
"""

import subprocess
import time
import os
import sys
import psutil
import requests
from pathlib import Path

def is_port_in_use(port):
    """Check if a port is already in use."""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False

def wait_for_service(url, timeout=30, service_name="Service"):
    """Wait for a service to become available."""
    print(f"⏳ Waiting for {service_name} to start...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready!")
                return True
        except:
            pass
        time.sleep(2)
    
    print(f"❌ {service_name} failed to start within {timeout} seconds")
    return False

def check_docker():
    """Check if Docker is running."""
    try:
        result = subprocess.run(['docker', 'version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def start_docker_services():
    """Start Docker services using docker-compose."""
    
    print("🐳 Starting Docker services...")
    
    if not check_docker():
        print("❌ Docker is not running. Please start Docker Desktop first.")
        return False
    
    # Check if services are already running
    try:
        result = subprocess.run(['docker-compose', 'ps'], 
                              capture_output=True, text=True)
        if 'Up' in result.stdout:
            print("✅ Docker services already running")
            return True
    except:
        pass
    
    # Start services
    try:
        print("📦 Starting PostgreSQL and Redis...")
        subprocess.run(['docker-compose', 'up', '-d', 'db', 'redis'], 
                      check=True, timeout=60)
        
        # Wait for database
        print("⏳ Waiting for PostgreSQL...")
        time.sleep(10)  # Give PostgreSQL time to fully start
        
        print("✅ Docker services started successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Docker services took too long to start")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Docker services: {e}")
        return False

def start_api_server():
    """Start the FastAPI server."""
    
    print("🚀 Starting FastAPI server...")
    
    if is_port_in_use(8000):
        print("✅ API server already running on port 8000")
        return True
    
    try:
        # Start API server in background
        api_process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 
            'app.backend.main:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for API to be ready
        if wait_for_service("http://localhost:8000/health", 30, "FastAPI"):
            print("✅ FastAPI server started successfully")
            return api_process
        else:
            api_process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False

def start_frontend():
    """Start the Streamlit frontend."""
    
    print("🎨 Starting Streamlit frontend...")
    
    if is_port_in_use(8501):
        print("✅ Frontend already running on port 8501")
        return True
    
    try:
        # Start Streamlit in background
        frontend_process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run',
            'app/frontend/app.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for frontend to be ready
        if wait_for_service("http://localhost:8501", 30, "Streamlit"):
            print("✅ Streamlit frontend started successfully")
            return frontend_process
        else:
            frontend_process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return False

def generate_sample_data():
    """Generate sample data if it doesn't exist."""
    
    sample_dir = Path("data/sample")
    
    if sample_dir.exists() and list(sample_dir.glob("*.tif")):
        print("✅ Sample data already exists")
        return True
    
    print("📊 Generating sample data...")
    
    try:
        result = subprocess.run([
            sys.executable, 'data/scripts/generate_sample_data.py'
        ], timeout=120, check=True)
        
        print("✅ Sample data generated successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Sample data generation timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate sample data: {e}")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'torch', 
        'rasterio', 'folium', 'plotly', 'psycopg2'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    """Main startup sequence."""
    
    print("🔥 BlazeNet Startup Script")
    print("=" * 40)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Generate sample data
    if not generate_sample_data():
        print("\n⚠️ Could not generate sample data, continuing anyway...")
    
    # Start Docker services
    if not start_docker_services():
        print("\n❌ Failed to start Docker services")
        print("💡 Make sure Docker Desktop is running")
        sys.exit(1)
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("\n❌ Failed to start API server")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n❌ Failed to start frontend")
        if api_process and hasattr(api_process, 'terminate'):
            api_process.terminate()
        sys.exit(1)
    
    # Success message
    print("\n🎉 BlazeNet started successfully!")
    print("=" * 40)
    print("🔗 Access Points:")
    print("   📊 Dashboard:  http://localhost:8501")
    print("   🚀 API:        http://localhost:8000")
    print("   📚 API Docs:   http://localhost:8000/docs")
    print("=" * 40)
    print("\n💡 Press Ctrl+C to stop all services")
    
    try:
        # Keep script running and monitor processes
        while True:
            time.sleep(5)
            
            # Check if processes are still running
            if hasattr(api_process, 'poll') and api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                break
                
            if hasattr(frontend_process, 'poll') and frontend_process.poll() is not None:
                print("❌ Frontend stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down BlazeNet...")
        
        # Stop processes
        if hasattr(api_process, 'terminate'):
            api_process.terminate()
            print("✅ API server stopped")
            
        if hasattr(frontend_process, 'terminate'):
            frontend_process.terminate()
            print("✅ Frontend stopped")
        
        print("👋 BlazeNet shutdown complete")

if __name__ == "__main__":
    main() 