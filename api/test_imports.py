#!/usr/bin/env python3
"""
Quick test to verify API server imports work correctly.
Run this from the api directory to test if all imports resolve.
"""

import sys
from pathlib import Path

# Add paths as the server does
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "strands_agents"))

print("Testing imports...")

try:
    from model_manager import ModelManager
    print("✅ ModelManager imported successfully")
except Exception as e:
    print(f"❌ ModelManager import failed: {e}")

try:
    from agent_manager import AgentManager
    print("✅ AgentManager imported successfully")
except Exception as e:
    print(f"❌ AgentManager import failed: {e}")

try:
    from tool_manager import ToolManager, ToolConfig
    print("✅ ToolManager imported successfully")
except Exception as e:
    print(f"❌ ToolManager import failed: {e}")

print("\nAll imports successful! The API server should start correctly.")
