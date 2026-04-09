"""
测试 ToolSearchTool 搜索功能
"""
import json
from vqnet_mcp_server.server import SimpleMCPServer

def test_search():
    """测试搜索功能"""
    server = SimpleMCPServer()

    print("=" * 60)
    print("测试 ToolSearchTool 搜索功能")
    print("=" * 60)

    # 测试用例
    test_cases = [
        ("adam", 3),          # 应该找到 Adam 优化器
        ("Hadamard", 3),      # 应该找到 Hadamard 门
        ("量子电路", 3),       # 中文搜索
        ("tensor", 5),        # 应该找到多个 tensor 相关工具
        ("优化器", 3),         # 中文搜索优化器
        ("Linear", 3),        # 应该找到 Linear 层
    ]

    for query, top_k in test_cases:
        print(f"\n搜索: '{query}' (top_k={top_k})")
        print("-" * 40)

        result = server.search_tools(query, top_k)

        if result["matched_tools"]:
            for tool in result["matched_tools"]:
                print(f"  Rank {tool['rank']}: {tool['name']}")
                print(f"         Score: {tool['score']}")
                print(f"         Description: {tool['description'][:60]}...")
                if tool['example_code']:
                    print(f"         Example Code (前50字符): {tool['example_code'][:50]}...")
        else:
            print("  未找到匹配工具")

        print(f"\n  总匹配数: {result['total_matches']}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_search()