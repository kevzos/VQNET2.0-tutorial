"""
Test for QNN_PQ3 MCP interface expansion
Tests for quantum layers, templates, gates, and measurement functions
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tools():
    """Load all MCP tool definitions"""
    tools_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vqnet_all_tools.json')
    with open(tools_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_quantum_layers():
    """Test quantum layer tools exist"""
    tools = load_tools()
    tool_names = [t['name'] for t in tools]

    expected_layers = [
        'pyvqnet.qnn.pq3.quantumlayer.QuantumLayer',
        'pyvqnet.qnn.pq3.quantumlayer.QpandaQProgVQCLayer',
        'pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint',
        'pyvqnet.qnn.pq3.quantumlayer.grad',
    ]

    for layer in expected_layers:
        assert layer in tool_names, f"Missing quantum layer: {layer}"
        print(f"OK: {layer}")

    print(f"\nAll {len(expected_layers)} quantum layers verified!")


def test_quantum_gates():
    """Test quantum gate and circuit template tools exist"""
    tools = load_tools()
    tool_names = [t['name'] for t in tools]

    expected_gates = [
        'pyvqnet.qnn.pq3.template.CSWAPcircuit',
        'pyvqnet.qnn.pq3.template.Controlled_Hadamard',
        'pyvqnet.qnn.pq3.template.CCZ',
    ]

    for gate in expected_gates:
        assert gate in tool_names, f"Missing quantum gate: {gate}"
        print(f"OK: {gate}")

    print(f"\nAll {len(expected_gates)} quantum gates verified!")


def test_ansatz_templates():
    """Test ansatz and template tools exist"""
    tools = load_tools()
    tool_names = [t['name'] for t in tools]

    expected_templates = [
        'pyvqnet.qnn.pq3.ansatz.HardwareEfficientAnsatz',
        'pyvqnet.qnn.pq3.template.BasicEntanglerTemplate',
        'pyvqnet.qnn.pq3.template.StronglyEntanglingTemplate',
        'pyvqnet.qnn.pq3.ComplexEntangelingTemplate',
        'pyvqnet.qnn.pq3.Quantum_Embedding',
        'pyvqnet.qnn.pq3.template.QuantumPoolingCircuit',
    ]

    for template in expected_templates:
        assert template in tool_names, f"Missing template: {template}"
        print(f"OK: {template}")

    print(f"\nAll {len(expected_templates)} ansatz templates verified!")


def test_fermionic_excitations():
    """Test fermionic excitation tools exist"""
    tools = load_tools()
    tool_names = [t['name'] for t in tools]

    expected_excitations = [
        'pyvqnet.qnn.pq3.template.FermionicSingleExcitation',
        'pyvqnet.qnn.pq3.template.FermionicDoubleExcitation',
    ]

    for exc in expected_excitations:
        assert exc in tool_names, f"Missing excitation: {exc}"
        print(f"OK: {exc}")

    print(f"\nAll {len(expected_excitations)} fermionic excitations verified!")


def test_measurement_functions():
    """Test measurement function tools exist"""
    tools = load_tools()
    tool_names = [t['name'] for t in tools]

    expected_measurements = [
        'pyvqnet.qnn.pq3.measure.ProbsMeasure',
        'pyvqnet.qnn.pq3.measure.DensityMatrixFromQstate',
        'pyvqnet.qnn.pq3.measure.VN_Entropy',
        'pyvqnet.qnn.pq3.measure.Mutal_Info',
        'pyvqnet.qnn.pq3.measure.Purity',
    ]

    for meas in expected_measurements:
        assert meas in tool_names, f"Missing measurement: {meas}"
        print(f"OK: {meas}")

    print(f"\nAll {len(expected_measurements)} measurement functions verified!")


def test_total_pq3_tools():
    """Test total number of QNN_PQ3 tools"""
    tools = load_tools()
    pq3_tools = [t for t in tools if 'pq3' in t['name']]

    print(f"\nTotal QNN_PQ3 tools: {len(pq3_tools)}")
    assert len(pq3_tools) >= 28, f"Expected at least 28 QNN_PQ3 tools, got {len(pq3_tools)}"

    # List all tools for verification
    print("\nAll QNN_PQ3 tools:")
    for i, t in enumerate(sorted([t['name'] for t in pq3_tools]), 1):
        print(f"  {i}. {t}")


def test_tool_schema():
    """Test tool schemas are valid"""
    tools = load_tools()

    for tool in tools:
        # Check required fields
        assert 'name' in tool, f"Tool missing 'name' field: {tool}"
        assert 'description' in tool, f"Tool missing 'description' field: {tool['name']}"

        # Check inputSchema exists
        assert 'inputSchema' in tool, f"Tool missing 'inputSchema' field: {tool['name']}"

    print(f"\nOK: All {len(tools)} tools have valid schemas")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing QNN_PQ3 MCP interface expansion")
    print("=" * 60)

    print("\n--- Test 1: Quantum Layers ---")
    test_quantum_layers()

    print("\n--- Test 2: Quantum Gates ---")
    test_quantum_gates()

    print("\n--- Test 3: Ansatz Templates ---")
    test_ansatz_templates()

    print("\n--- Test 4: Fermionic Excitations ---")
    test_fermionic_excitations()

    print("\n--- Test 5: Measurement Functions ---")
    test_measurement_functions()

    print("\n--- Test 6: Total QNN_PQ3 Tools ---")
    test_total_pq3_tools()

    print("\n--- Test 7: Tool Schema ---")
    test_tool_schema()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
