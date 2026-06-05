"""
Quantum Alpha Intelligence Platform
Entry point for Hugging Face Spaces deployment.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.app import app

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
