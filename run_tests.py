#!/usr/bin/env python3
# ## File: run_tests.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Test runner script for Cortex Suite with various test configurations.

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    if description:
        print(f"🧪 {description}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Cortex Suite tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "api", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test type markers
    if args.type == "unit":
        cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
    elif args.type == "api":
        cmd.extend(["-m", "api"])
    elif args.type == "all":
        # Run all tests
        pass
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=cortex_engine",
            "--cov=api", 
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Add verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    # Skip slow tests if requested
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    # Run in parallel if requested
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Run specific file if specified
    if args.file:
        cmd.append(f"tests/{args.file}")
    else:
        cmd.append("tests/")
    
    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests")
    
    if success:
        print(f"\n🎉 All tests passed!")
        
        # Generate coverage report if coverage was enabled
        if args.coverage:
            print(f"\n📊 Coverage report generated in htmlcov/index.html")
            
        return 0
    else:
        print(f"\n💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())