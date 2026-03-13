"""
Test for high-priority VQC interfaces
Tests for basic quantum gates (I, PauliX/Y/Z, T, S) and embedding circuits
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


def test_basic_gate_classes():
    """Test basic quantum gate class tools exist"""
    tools = load_tools()
    tool_names = [t['name'] for t in tools]

    # Basic gate classes (Y1, Z1 not in rst documentation)
    expected_gates = [
        'pyvqnet.qnn.vqc.I',
        'pyvqnet.qnn.vqc.Hadamard',
        'pyvqnet.qnn.vqc.T',
        'pyvqnet.qnn.vqc.S',
        'pyvqnet.qnn.vqc.PauliX',
        'pyvqnet.qnn.vqc.PauliY',
        'pyvqnet.qnn.vqc.PauliZ',
        'pyvqnet.qnn.vqc.X1',
    ]

    for gate in expected_gates:
        assert gate in tool_names, f"Missing gate class: {gate}"
        print(f"OK: {gate}")

    print(f"\nAll {len(expected_gates)} basic gate classes verified!")


def test_basic_gate_functions():
    """Test basic quantum gate function tools exist"""
    tools = load_tools()
    tool_names = [t['name'] for t in tools]

    # Basic gate functions (y1, z1 not in rst documentation)
    expected_functions = [
        'pyvqnet.qnn.vqc.hadamard',
        'pyvqnet.qnn.vqc.t',
        'pyvqnet.qnn.vqc.s',
        'pyvqnet.qnn.vqc.paulix',
        'pyvqnet.qnn.vqc.pauliy',
        'pyvqnet.qnn.vqc.pauliz',
        'pyvqnet.qnn.vqc.x1',
    ]

    for func in expected_functions:
        assert func in tool_names, f"Missing gate function: {func}"
        print(f"OK: {func}")

    print(f"\nAll {len(expected_functions)} basic gate functions verified!")


def test_embedding_circuits():
    """Test embedding circuit tools exist"""
    tools = load_tools()
    tool_names = [t['name'] for t in tools]

    expected_embeddings = [
        'pyvqnet.qnn.pq3.template.BasicEmbeddingCircuit',
        'pyvqnet.qnn.pq3.template.RotCircuit',
        'pyvqnet.qnn.pq3.template.CRotCircuit',
        'pyvqnet.qnn.pq3.template.IQPEmbeddingCircuits',
    ]

    for emb in expected_embeddings:
        assert emb in tool_names, f"Missing embedding circuit: {emb}"
        print(f"OK: {emb}")

    print(f"\nAll {len(expected_embeddings)} embedding circuits verified!")


def test_measurement_tools():
    """Test measurement tools exist"""
    tools = load_tools()
    tool_names = [t['name'] for t in tools]

    expected_measurements = [
        'pyvqnet.qnn.pq3.measure.expval',
        'pyvqnet.qnn.pq3.measure.QuantumMeasure',
        'pyvqnet.qnn.vqc.VQC_VarMeasure',
    ]

    for meas in expected_measurements:
        assert meas in tool_names, f"Missing measurement tool: {meas}"
        print(f"OK: {meas}")

    print(f"\nAll {len(expected_measurements)} measurement tools verified!")


def test_gate_code_generation():
    """Test code generation for gate tools"""
    tools = load_tools()
    tools_dict = {t['name']: t for t in tools}

    # Test I gate code generation
    i_gate = tools_dict.get('pyvqnet.qnn.vqc.I')
    assert i_gate is not None, "I gate not found"

    # Verify tool has inputSchema for parameters
    assert 'inputSchema' in i_gate, "I gate missing inputSchema"
    assert 'properties' in i_gate['inputSchema'], "I gate inputSchema missing properties"

    print("OK: I gate has code generation schema")

    # Test PauliX gate
    paulix = tools_dict.get('pyvqnet.qnn.vqc.PauliX')
    assert paulix is not None, "PauliX gate not found"
    assert 'inputSchema' in paulix, "PauliX gate missing inputSchema"
    print("OK: PauliX gate has code generation schema")

    # Test expval
    expval = tools_dict.get('pyvqnet.qnn.pq3.measure.expval')
    assert expval is not None, "expval not found"
    assert 'inputSchema' in expval, "expval missing inputSchema"
    print("OK: expval has code generation schema")


def test_tool_schema():
    """Test tool schemas are valid"""
    tools = load_tools()

    for tool in tools:
        # Check required fields
        assert 'name' in tool, f"Tool missing 'name' field: {tool}"
        assert 'description' in tool, f"Tool missing 'description' field: {tool['name']}"

        # Check inputSchema exists
        assert 'inputSchema' in tool, f"Tool missing 'inputSchema' field: {tool['name']}"

    print(f"OK: All {len(tools)} tools have valid schemas")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing high-priority VQC interfaces")
    print("=" * 60)

    print("\n--- Test 1: Basic Gate Classes ---")
    test_basic_gate_classes()

    print("\n--- Test 2: Basic Gate Functions ---")
    test_basic_gate_functions()

    print("\n--- Test 3: Embedding Circuits ---")
    test_embedding_circuits()

    print("\n--- Test 4: Measurement Tools ---")
    test_measurement_tools()

    print("\n--- Test 5: Code Generation ---")
    test_gate_code_generation()

    print("\n--- Test 6: Tool Schema ---")
    test_tool_schema()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
