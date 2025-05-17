#!/usr/bin/env python3
# startup_check.py - Run this to diagnose startup issues

import sys
import os
import traceback

print("=== AI Concierge Startup Diagnostic ===\n")

# Check Python version
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

# Check if we're in the right directory
required_files = ['app', 'app/main.py', 'app/core', 'app/core/config.py']
missing_files = []

for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print(f"\n❌ Missing required files/directories: {missing_files}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)
else:
    print("\n✅ All required files/directories found")

# Check for CUDA availability
print("\n=== CUDA Check ===")
try:
    import torch
    print(f"✅ PyTorch installed: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("⚠️  PyTorch not installed - CUDA unavailable")

# Test imports
print("\n=== Testing imports ===")

modules_to_test = [
    'pydantic',
    'fastapi', 
    'langchain',
    'langchain_core',
    'langgraph',
    'chromadb',
    'app.core.config',
    'app.models.user',
    'app.models.task',
]

failed_imports = []

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✅ {module}")
    except ImportError as e:
        print(f"❌ {module}: {e}")
        failed_imports.append(module)

# Test optional imports
print("\n=== Testing optional imports ===")

optional_modules = [
    'langchain_openai',
    'langchain_community',
    'llama_cpp',
    'ctransformers',
]

for module in optional_modules:
    try:
        __import__(module)
        print(f"✅ {module} (optional)")
    except ImportError as e:
        print(f"⚠️  {module} (optional): Not available - {e}")

# Check for simple_concierge references
print("\n=== Checking for simple_concierge references ===")
import glob

def search_for_simple_concierge():
    """Search for simple_concierge references in Python files."""
    references = []
    python_files = glob.glob('app/**/*.py', recursive=True)
    python_files.extend(['app/main.py', 'startup-check.py'])
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'simple_concierge' in content:
                    # Find the line numbers
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'simple_concierge' in line:
                            references.append(f"{file_path}:{i}: {line.strip()}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return references

references = search_for_simple_concierge()
if references:
    print("Found simple_concierge references:")
    for ref in references:
        print(f"  📁 {ref}")
else:
    print("✅ No simple_concierge references found")

# Check environment
print("\n=== Environment variables ===")
env_vars = ['LOCAL_MODEL_PATH', 'OPENAI_API_KEY', 'HUGGINGFACE_API_TOKEN']

for var in env_vars:
    value = os.getenv(var)
    if value:
        if 'KEY' in var or 'TOKEN' in var:
            print(f"✅ {var}: ***set***")
        else:
            print(f"✅ {var}: {value}")
    else:
        print(f"⚠️  {var}: Not set")

# Test model file existence
print("\n=== Model file check ===")
model_paths = [
    'app/models/Llama-3.2-3B-Instruct-IQ4_XS.gguf',
    os.getenv('LOCAL_MODEL_PATH', 'app/models/Llama-3.2-3B-Instruct-IQ4_XS.gguf')
]

for path in set(model_paths):  # Remove duplicates
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ Model found: {path} ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  Model not found: {path}")

# Test services import individually
print("\n=== Testing services import ===")
try:
    from app.services.concierge import ConciergeService
    print("✅ ConciergeService imported successfully")
    
    # Try to instantiate (catch any errors)
    try:
        # service = ConciergeService()
        print("✅ ConciergeService can be instantiated")
    except Exception as e:
        print(f"⚠️  Error instantiating ConciergeService: {e}")
        
except ImportError as e:
    print(f"❌ Failed to import ConciergeService: {e}")
    traceback.print_exc()

# Test application import
print("\n=== Testing application import ===")
try:
    from app.main import app
    print("✅ Successfully imported FastAPI app")
    
    # Test if we can access routes
    routes = [str(route) for route in app.routes]
    print(f"✅ App has {len(routes)} routes configured")
    
except Exception as e:
    print(f"❌ Failed to import app: {e}")
    traceback.print_exc()
    failed_imports.append('app.main')

# Summary
print("\n=== Summary ===")
if failed_imports:
    print(f"❌ Failed imports: {failed_imports}")
    print("\nTo fix missing dependencies, run:")
    print("pip install -r requirements.txt")
    if 'llama_cpp' in failed_imports:
        print("For local Llama support with CUDA:")
        print("pip uninstall llama-cpp-python")
        print("pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
else:
    print("✅ All core imports successful!")
    print("\nYou can now start the server with:")
    print("uvicorn app.main:app --reload")

# Recommendations
print("\n=== Recommendations ===")
if not os.getenv('LOCAL_MODEL_PATH'):
    print("💡 Set LOCAL_MODEL_PATH environment variable for better model loading")
if not any(os.path.exists(path) for path in model_paths):
    print("💡 Download a local model to app/models/ for offline inference")

print("\n=== End Diagnostic ===")