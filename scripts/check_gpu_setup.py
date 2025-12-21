# /home/kirill/projects_2/folium/Xplore/scripts/check_gpu_setup.py

"""
GPU and Ollama Setup Verification Script

Run this script to verify your GPU and Ollama configuration is correct.
"""

import subprocess
import sys
import os
import requests
import time

def print_header(text):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_nvidia_smi():
    """Check if nvidia-smi is available and GPU is detected"""
    print_header("1. Checking NVIDIA GPU")
    
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected!")
            print("\nGPU Information:")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi failed")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("✗ nvidia-smi not found. NVIDIA drivers may not be installed.")
        print("  Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
        return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def check_ollama_installed():
    """Check if Ollama is installed"""
    print_header("2. Checking Ollama Installation")
    
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"✓ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("✗ Ollama command failed")
            return False
            
    except FileNotFoundError:
        print("✗ Ollama not found. Please install it:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        print("  Or visit: https://ollama.com/download")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False


def check_ollama_server():
    """Check if Ollama server is running"""
    print_header("3. Checking Ollama Server")
    
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            print(f"✓ Ollama server is running at {base_url}")
            
            models = response.json().get("models", [])
            print(f"\n  Available models: {len(models)}")
            for model in models:
                print(f"    - {model['name']}")
            
            return True
        else:
            print(f"✗ Ollama server responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Ollama server at {base_url}")
        print("  Start it with: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama server: {e}")
        return False


def check_model_available():
    """Check if the configured model is available"""
    print_header("4. Checking Model Availability")
    
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    try:
        response = requests.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✓ Model '{model_name}' is available")
            
            info = response.json()
            if "details" in info:
                print(f"\n  Model details:")
                details = info["details"]
                if "parameter_size" in details:
                    print(f"    Parameters: {details['parameter_size']}")
                if "quantization_level" in details:
                    print(f"    Quantization: {details['quantization_level']}")
            
            return True
        else:
            print(f"✗ Model '{model_name}' not found")
            print(f"  Pull it with: ollama pull {model_name}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking model: {e}")
        return False


def test_gpu_generation():
    """Test actual GPU usage with a small generation"""
    print_header("5. Testing GPU Generation")
    
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"Running test generation with {model_name}...")
    print("Watch GPU usage with: watch -n 0.5 nvidia-smi")
    print()
    
    try:
        payload = {
            "model": model_name,
            "prompt": "Write a haiku about AI.",
            "stream": False,
            "options": {
                "num_gpu": -1,  # Use all GPU layers
            }
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✓ Generation completed in {elapsed:.2f}s")
            print(f"\n  Response: {data.get('response', '')[:200]}")
            
            if "eval_duration" in data:
                eval_time = data["eval_duration"] / 1e9  # Convert ns to s
                eval_count = data.get("eval_count", 0)
                tokens_per_sec = eval_count / eval_time if eval_time > 0 else 0
                
                print(f"\n  Performance:")
                print(f"    Tokens evaluated: {eval_count}")
                print(f"    Evaluation time: {eval_time:.2f}s")
                print(f"    Speed: {tokens_per_sec:.1f} tokens/sec")
                
                if tokens_per_sec > 50:
                    print("\n  ✓ GPU appears to be working well (>50 tokens/sec)")
                elif tokens_per_sec > 10:
                    print("\n  ⚠ GPU might be working but performance is low")
                else:
                    print("\n  ✗ Performance is very low - GPU may not be used")
            
            return True
        else:
            print(f"✗ Generation failed with status {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"✗ Error during generation test: {e}")
        return False


def check_env_config():
    """Check environment configuration"""
    print_header("6. Checking Environment Configuration")
    
    configs = {
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "OLLAMA_TIMEOUT": os.getenv("OLLAMA_TIMEOUT", "600"),
        "OLLAMA_GPU_LAYERS": os.getenv("OLLAMA_GPU_LAYERS", "-1"),
        "OPENAI_API_KEY": "***" if os.getenv("OPENAI_API_KEY") else "Not set",
    }
    
    print("Environment variables:")
    for key, value in configs.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    # Check critical settings
    gpu_layers = os.getenv("OLLAMA_GPU_LAYERS", "-1")
    if gpu_layers == "-1":
        print("\n✓ GPU layers set to -1 (all layers on GPU)")
    elif gpu_layers == "0":
        print("\n⚠ WARNING: GPU layers set to 0 (CPU only!)")
    else:
        print(f"\n⚠ GPU layers set to {gpu_layers} (partial GPU usage)")
    
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
    if timeout < 300:
        print(f"⚠ WARNING: Timeout is {timeout}s - may be too low for long generations")
    else:
        print(f"✓ Timeout is {timeout}s - sufficient for ensemble runs")


def main():
    """Run all checks"""
    print("\n" + "=" * 80)
    print("  GPU AND OLLAMA SETUP VERIFICATION")
    print("=" * 80)
    
    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("\n✓ Loaded configuration from .env file")
    except ImportError:
        print("\n⚠ python-dotenv not installed, using system environment only")
    except Exception:
        print("\n⚠ Could not load .env file")
    
    # Run all checks
    results = {
        "GPU": check_nvidia_smi(),
        "Ollama Installed": check_ollama_installed(),
        "Ollama Server": check_ollama_server(),
        "Model Available": check_model_available(),
        "Generation Test": test_gpu_generation(),
    }
    
    check_env_config()
    
    # Summary
    print_header("SUMMARY")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    if all_passed:
        print("\n" + "=" * 80)
        print("  ✓ ALL CHECKS PASSED - SYSTEM IS READY!")
        print("=" * 80)
        print("\nYou can now run:")
        print("  python -m ui_api.main")
        return 0
    else:
        print("\n" + "=" * 80)
        print("  ✗ SOME CHECKS FAILED - PLEASE FIX ISSUES ABOVE")
        print("=" * 80)
        print("\nRefer to SETUP_AND_USAGE.md for troubleshooting")
        return 1


if __name__ == "__main__":
    sys.exit(main())