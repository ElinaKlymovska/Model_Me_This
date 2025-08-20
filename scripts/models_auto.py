#!/usr/bin/env python3
"""
Automatic Model Downloader for Portrait Enhancement Pipeline
Downloads models from CivitAI based on configuration
"""

import os
import sys
import yaml
import requests
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_config():
    """Load configuration from config/models.yaml"""
    config_file = project_root / "config" / "models.yaml"
    
    if not config_file.exists():
        print(f"[ERROR] Configuration file not found: {config_file}")
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        print(f"[ERROR] Failed to parse configuration: {e}")
        return None

def get_model_paths():
    """Get model paths from configuration"""
    config = load_config()
    if not config:
        return None
    
    paths = config.get('paths', {})
    return {
        'stable_diffusion': paths.get('stable_diffusion', 'models/Stable-diffusion'),
        'controlnet': paths.get('controlnet', 'models/ControlNet'),
        'adetailer': paths.get('adetailer', 'models/adetailer'),
        'lora': paths.get('lora', 'models/Lora'),
        'embeddings': paths.get('embeddings', 'models/embeddings')
    }

def get_download_settings():
    """Get download settings from configuration"""
    config = load_config()
    if not config:
        return None
    
    download = config.get('download', {})
    return {
        'api_key': download.get('civitai_api_key', ''),
        'timeout': download.get('timeout', 1800),
        'retry_attempts': download.get('retry_attempts', 3),
        'chunk_size': download.get('chunk_size', 1048576)
    }

def get_models_to_download():
    """Get list of models to download from configuration"""
    config = load_config()
    if not config:
        return []
    
    models = []
    
    # Stable Diffusion models
    sd_models = config.get('stable_diffusion', {}).get('models', [])
    for model in sd_models:
        models.append({
            'name': model['name'],
            'type': 'checkpoint',
            'url': model['url'],
            'description': model['description']
        })
    
    # ControlNet models
    cn_models = config.get('controlnet', {}).get('models', [])
    for model in cn_models:
        models.append({
            'name': model['name'],
            'type': 'controlnet',
            'url': model['url'],
            'description': model['description']
        })
    
    return models

def create_directories():
    """Create necessary model directories"""
    paths = get_model_paths()
    if not paths:
        return False
    
    for path_name, path_value in paths.items():
        full_path = project_root / path_value
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"[*] Created directory: {full_path}")
    
    return True

def civ_get(path, params=None, stream=False):
    """Make request to CivitAI API"""
    settings = get_download_settings()
    if not settings:
        return None
    
    headers = {"Authorization": f"Bearer {settings['api_key']}"} if settings['api_key'] else {}
    base_url = "https://civitai.com/api/v1"
    
    try:
        r = requests.get(f"{base_url}{path}", params=params, headers=headers, 
                        stream=stream, timeout=settings['timeout'])
        r.raise_for_status()
        return r
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return None

def pick_best_file(ver):
    """Pick the best file from model version"""
    files = ver.get("files") or []
    
    def score(f):
        name = (f.get("name") or "").lower()
        fmt = (f.get("format") or "").lower()
        s = 0
        if name.endswith(".safetensors"):
            s += 5
        if "sdxl" in name or "xl" in name:
            s += 3
        if "softedge" in name or "dexined" in name:
            s += 2
        if "canny" in name or "lineart" in name:
            s += 2
        if fmt == "safetensors":
            s += 1
        return s
    
    files.sort(key=score, reverse=True)
    return files[0] if files else None

def search_and_download(query, model_type, out_dir):
    """Search and download model from CivitAI"""
    print(f"[*] Searching {model_type}: {query}")
    
    # Search for models
    response = civ_get("/models", params={"query": query, "types": model_type, "limit": 10})
    if not response:
        raise RuntimeError("API request failed")
    
    items = response.json().get("items") or []
    if not items:
        raise RuntimeError("No results found")
    
    # Sort by download count and version ID
    items.sort(key=lambda it: (
        it.get("stats", {}).get("downloadCount", 0), 
        it.get("modelVersions", [{}])[0].get("id", 0)
    ), reverse=True)
    
    ver = items[0].get("modelVersions", [{}])[0]
    f = pick_best_file(ver)
    
    if not f:
        raise RuntimeError("No suitable file found")
    
    url = f.get("downloadUrl")
    name = f.get("name")
    
    if not url:
        raise RuntimeError("No download URL available")
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / name
    
    # Download if file doesn't exist
    if not out_file.exists():
        print(f"[*] Downloading: {name}")
        settings = get_download_settings()
        
        headers = {"Authorization": f"Bearer {settings['api_key']}"} if settings['api_key'] else {}
        
        with requests.get(url, headers=headers, stream=True, timeout=settings['timeout']) as rr:
            rr.raise_for_status()
            with open(out_file, "wb") as fp:
                for chunk in rr.iter_content(settings['chunk_size']):
                    if chunk:
                        fp.write(chunk)
        
        print(f"[OK] Downloaded: {out_file}")
    else:
        print(f"[SKIP] File already exists: {out_file}")
    
    return str(out_file)

def download_models_from_config():
    """Download models based on configuration"""
    print("[*] Starting model download from configuration...")
    
    # Create directories
    if not create_directories():
        print("[ERROR] Failed to create directories")
        return False
    
    # Get models to download
    models = get_models_to_download()
    if not models:
        print("[WARN] No models configured for download")
        return False
    
    paths = get_model_paths()
    if not paths:
        print("[ERROR] Failed to get model paths")
        return False
    
    # Download each model
    for model in models:
        try:
            if model['type'] == 'checkpoint':
                out_dir = paths['stable_diffusion']
            elif model['type'] == 'controlnet':
                out_dir = paths['controlnet']
            else:
                continue
            
            # Extract model name from URL for search
            model_name = model['name'].replace('.safetensors', '').replace('.ckpt', '')
            search_and_download(model_name, model['type'], out_dir)
            
        except Exception as e:
            print(f"[WARN] Failed to download {model['name']}: {e}")
    
    print("[*] Model download completed!")
    return True

def main():
    """Main function"""
    print("🚀 Portrait Enhancement Pipeline - Model Downloader")
    print("=" * 50)
    
    # Check if configuration exists
    if not (project_root / "config" / "models.yaml").exists():
        print("[ERROR] Configuration file config/models.yaml not found!")
        print("Please run this script from the project root directory.")
        return 1
    
    # Download models
    try:
        success = download_models_from_config()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n[INFO] Download interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
