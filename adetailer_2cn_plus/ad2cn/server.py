"""
Simple server module for ADetailer 2CN Plus.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ad2cn.config import Config
from ad2cn.pipeline.detect import DetectionPipeline


def main():
    """Simple server entry point."""
    print("ADetailer 2CN Plus Server")
    print("=" * 30)
    
    try:
        # Load configuration
        config_path = Path("config.yaml")
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            config = Config(**config_data)
            print("✓ Configuration loaded")
        else:
            # Use minimal config
            config_data = {
                "pipeline": {
                    "detectors": [{"name": "blazeface"}]
                }
            }
            config = Config(**config_data)
            print("✓ Using minimal configuration")
        
        # Initialize pipeline
        pipeline = DetectionPipeline(config.pipeline.dict())
        print("✓ Pipeline initialized")
        
        # Show status
        pipeline_info = pipeline.get_pipeline_info()
        print(f"✓ Pipeline ready: {pipeline_info['detectors']}")
        
        print("\nServer is ready!")
        print("Press Ctrl+C to exit")
        
        # Keep server running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
