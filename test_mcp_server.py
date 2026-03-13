#!/usr/bin/env python
"""
测试MCP服务器是否能正确加载工具列表
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_mcp_server import SimpleMCPServer

def main():
    print("Testing MCP server loading tools...")
    server = SimpleMCPServer("vqnet_all_tools.json")
    print(f"Server loaded {len(server.tools)} tools")

    # List first 5 tool names
    if server.tools:
        print("First 5 tools:")
        for i, tool in enumerate(server.tools[:5]):
            print(f"  {i+1}. {tool.get('name', 'Unknown')}")
    else:
        print("Error: No tools loaded")
        return False

    # Check if tool count matches expected (we saw 151 tools generated)
    expected_min = 150  # We expect at least 150
    if len(server.tools) >= expected_min:
        print(f"PASS: Tool count meets expectation ({len(server.tools)} >= {expected_min})")
        return True
    else:
        print(f"FAIL: Tool count below expectation ({len(server.tools)} < {expected_min})")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)