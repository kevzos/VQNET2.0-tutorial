#!/usr/bin/env python
"""
Test the newly added VQC interfaces to the MCP implementation.
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from simple_mcp_server import SimpleMCPServer

def test_new_vqc_interfaces():
    """Test the 26 newly added VQC interfaces"""
    print("Testing newly added VQC interfaces...")

    server = SimpleMCPServer()

    # Define the 26 new interfaces we added
    new_interfaces = [
        # Controlled gates
        "pyvqnet.qnn.vqc.crx",
        "pyvqnet.qnn.vqc.CRX",
        "pyvqnet.qnn.vqc.cry",
        "pyvqnet.qnn.vqc.CRY",
        "pyvqnet.qnn.vqc.crz",
        "pyvqnet.qnn.vqc.CRZ",

        # Advanced VQC functions
        "pyvqnet.qnn.vqc.VQC_AllSinglesDoubles",
        "pyvqnet.qnn.vqc.VQC_BasisRotation",
        "pyvqnet.qnn.vqc.VQC_QuantumPoolingCircuit",
        "pyvqnet.qnn.vqc.vqc_qft_add_to_register",
        "pyvqnet.qnn.vqc.VQC_FABLE",
        "pyvqnet.qnn.vqc.VQC_LCU",
        "pyvqnet.qnn.vqc.VQC_QSVT",

        # Quantum neural network models
        "pyvqnet.qnn.qcnn.Quanvolution",
        "pyvqnet.qnn.qdrl_vqc.QDRL",
        "pyvqnet.qnn.qgru.QGRU",
        "pyvqnet.qnn.qlstm.QLSTM",
        "pyvqnet.qnn.qrl.QRLModel",
        "pyvqnet.qnn.qrnn.QRNN",

        # Optimization and utilities
        "pyvqnet.qnn.vqc.wrapper_single_qubit_op_fuse",
    ]

    found_count = 0
    missing_count = 0
    missing_interfaces = []

    for interface in new_interfaces:
        # Find the tool by name
        tool = None
        for t in server.tools:
            if t.get('name') == interface:
                tool = t
                break

        if tool:
            found_count += 1
            print(f"[PASS] Found: {interface}")
        else:
            missing_count += 1
            missing_interfaces.append(interface)
            print(f"[FAIL] Missing: {interface}")

    print(f"\n=== Results ===")
    print(f"Total new interfaces to verify: {len(new_interfaces)}")
    print(f"Found: {found_count}")
    print(f"Missing: {missing_count}")

    if missing_interfaces:
        print(f"\nMissing interfaces:")
        for interface in missing_interfaces:
            print(f"  - {interface}")

    return found_count == len(new_interfaces)

def test_tool_functionality():
    """Test that some of the new tools can be called"""
    print("\n=== Testing Tool Functionality ===")

    server = SimpleMCPServer()

    # Test a few representative tools
    test_cases = [
        {
            "name": "pyvqnet.qnn.vqc.crx",
            "params": {"q_machine": "qm", "wires": "[0, 1]", "params": "np.pi/2"}
        },
        {
            "name": "pyvqnet.qnn.qcnn.Quanvolution",
            "params": {"params_shape": "[4, 4]", "strides": "(1, 1)"}
        },
        {
            "name": "pyvqnet.qnn.vqc.VQC_FABLE",
            "params": {"wires": "[0, 1, 2]"}
        }
    ]

    success_count = 0
    total_count = len(test_cases)

    for test_case in test_cases:
        tool_name = test_case["name"]
        params = test_case["params"]

        # Find the tool
        tool = None
        for t in server.tools:
            if t.get('name') == tool_name:
                tool = t
                break

        if tool:
            # Try to call the tool
            try:
                request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": params
                    }
                }

                response = server.handle_request(request)

                if "result" in response and "content" in response["result"]:
                    print(f"[PASS] {tool_name}: Success")
                    success_count += 1
                else:
                    print(f"[FAIL] {tool_name}: Failed - {response.get('error', {}).get('message', 'Unknown error')}")

            except Exception as e:
                print(f"[FAIL] {tool_name}: Exception - {str(e)}")
        else:
            print(f"[FAIL] {tool_name}: Tool not found")

    print(f"\nFunctionality test: {success_count}/{total_count} passed")
    return success_count == total_count

if __name__ == "__main__":
    print("=" * 60)
    print("Testing New VQC Interfaces Expansion")
    print("=" * 60)

    interface_test_passed = test_new_vqc_interfaces()
    functionality_test_passed = test_tool_functionality()

    print("\n" + "=" * 60)
    if interface_test_passed and functionality_test_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print("[FAILURE] Some tests failed")
    print("=" * 60)