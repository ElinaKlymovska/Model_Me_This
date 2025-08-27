"""Simple prompt generation."""

from typing import Dict, Any


class EnhancedPrompts:
    """Simple prompt generator."""
    
    def __init__(self):
        self.config = {}
        
    def get_prompt(self, character_type: str = "portrait") -> str:
        """Generate simple prompt."""
        return "beautiful portrait, high quality, detailed"
        
    def get_negative_prompt(self) -> str:
        """Get negative prompt."""
        return "blurry, low quality, distorted"


def get_enhanced_prompt_config(character_type: str) -> Dict[str, Any]:
    """Get simple prompt config."""
    return {
        "positive": "beautiful portrait, high quality, detailed",
        "negative": "blurry, low quality, distorted"
    }