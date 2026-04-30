#!/usr/bin/env python3
"""
Activation script for venv_complete environment.
Run this to ensure your terminal is using the complete environment.

Usage:
    # On Windows PowerShell:
    . .\activate_complete_env.ps1
    
    # On Windows CMD:
    venv_complete\Scripts\activate.bat
    
    # On Linux/Mac:
    source venv_complete/bin/activate
"""

import os
import sys
import subprocess
from pathlib import Path

def get_venv_python():
    """Get path to Python in venv_complete"""
    path = Path(__file__).parent / "venv_complete" / "Scripts" / "python.exe"
    if not path.exists():
        path = Path(__file__).parent / "venv_complete" / "bin" / "python"
    return str(path)

def main():
    """Activate the complete environment"""
    venv_python = get_venv_python()
    
    if not Path(venv_python).exists():
        print("ERROR: venv_complete not found!")
        print("Please create it first with: python -m venv venv_complete")
        sys.exit(1)
    
    print("=" * 80)
    print("COMPLETE ENVIRONMENT ACTIVATED")
    print("=" * 80)
    print()
    print("Python environment:")
    print(f"  Location: {Path(venv_python).parent.parent}")
    print(f"  Python:   {venv_python}")
    print()
    print("To run your project:")
    print(f"  {venv_python} main_complete.py")
    print(f"  {venv_python} comprehensive_analysis.py")
    print()
    print("To run with YAFS:")
    print(f"  {venv_python} comprehensive_analysis.py --use-yafs")
    print()
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
