#!/usr/bin/env python3
"""
Master runner for the entire Fog RL Medical project.
Runs both Hierarchical DQN training and comparative evaluation.

Usage: python run_all.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_comparison():
    """Run full comparison of all approaches."""
    print_header("PHASE 1: COMPARATIVE EVALUATION")
    print("Running: Hierarchical DQN vs 3 Baselines\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/run_comparison.py"],
            cwd=Path(__file__).parent,
            capture_output=False
        )
        
        if result.returncode != 0:
            print("❌ Comparison failed!")
            return False
        
        print("\n✅ Comparison completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error running comparison: {e}")
        return False

def main():
    """Main orchestration."""
    print_header("FOG RL MEDICAL - COMPLETE PROJECT RUNNER")
    print("This will execute the full pipeline:")
    print("  1. Train all approaches (10 episodes each)")
    print("  2. Generate performance comparisons")
    print("  3. Create visualization reports\n")
    
    start_time = time.time()
    
    # Run comparison (includes H-DQN + baselines)
    success = run_comparison()
    
    if success:
        elapsed = time.time() - start_time
        print_header("🎉 PROJECT COMPLETE")
        print(f"Total execution time: {elapsed:.1f} seconds\n")
        print("📊 Results Location: results/comparison/")
        print("   - multi_approach_comparison.png")
        print("   - improvement_analysis.png")
        print("   - final_metrics_table.png")
        print("   - COMPARISON_RESULTS.md\n")
        print("📈 To view results, open PNG files in results/comparison/\n")
        return 0
    else:
        print_header("❌ PROJECT FAILED")
        print("Check error messages above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
