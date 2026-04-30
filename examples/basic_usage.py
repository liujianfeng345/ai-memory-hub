"""
Memory Agent 基本使用示例。

演示完整的记忆管理流程：
配置 → 初始化 → 写入三种记忆 → 检索 → 整合 → 清理

运行前请确保已配置环境变量 DEEPSEEK_API_KEY::

    export DEEPSEEK_API_KEY="sk-your-key"  # Linux / macOS
    set DEEPSEEK_API_KEY=sk-your-key       # Windows cmd
    $env:DEEPSEEK_API_KEY="sk-your-key"    # Windows PowerShell

然后运行::

    python examples/basic_usage.py
"""

import asyncio
import logging
import os
import sys

# 将项目根目录加入 sys.path，方便在没有 pip install 时直接运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory_agent import ConsolidateResult, MemoryConfig, MemoryManager  # noqa: E402


def _check_api_key() -> None:
    """检查是否配置了有效的 API key，未配置时给出提示。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key or api_key == "sk-test000000000000":
        print("=" * 60)
        print("  [警告] DEEPSEEK_API_KEY 未配置或为测试占位值")
        print("  请先设置环境变量，否则涉及 LLM 的操作将失败：")
        print("    export DEEPSEEK_API_KEY='sk-your-key'")
        print("=" * 60)
        print()
    else:
        print(f"DEEPSEEK_API_KEY 已配置: {api_key[:8]}...")
        print()


async def main() -> None:
    """主函数：演示完整的记忆管理流程。"""
    # 配置日志，便于观察执行过程
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 检查 API key
    _check_api_key()

    # ===================================================================
    # 1. 加载配置
    # ===================================================================
    # MemoryConfig 会自动从环境变量和 .env 文件加载配置
    # 也可以直接传入参数覆盖默认值
    config = MemoryConfig(
        # chroma_persist_dir="./data/chroma",  # ChromaDB 持久化目录（默认值）
        # deepseek_model="deepseek-chat",       # LLM 模型名称（默认值）
    )
    print(f"[1] 配置加载完成: chroma_dir={config.chroma_persist_dir}")
    print(f"    模型: {config.deepseek_model}, 嵌入模型: {config.embedding_model_name}")
    print()

    # ===================================================================
    # 2. 初始化 MemoryManager
    # ===================================================================
    # MemoryManager 是唯一需要实例化的入口类，构造时会装配所有内部组件：
    # - InMemoryStore → WorkingMemory（工作记忆）
    # - ChromaStore → EpisodicMemory（情节记忆）
    # - ChromaStore → SemanticMemory（语义记忆）
    # - LocalEmbedder（嵌入模型）
    # - DeepSeekClient（LLM 客户端）
    manager = MemoryManager(config)
    print("[2] MemoryManager 初始化完成")
    print()

    session_id = "demo-session-001"

    # ===================================================================
    # 3. 写入工作记忆 —— 当前会话中的临时信息
    # ===================================================================
    # 工作记忆用于存储会话内的临时上下文，基于关键词检索，支持 TTL 自动过期。
    wm_item = await manager.remember(
        content="用户正在学习 Python 异步编程，已经掌握了 asyncio 的基础用法",
        memory_type="working",
        session_id=session_id,
    )
    print(f"[3] 写入工作记忆: id={wm_item.id}")
    print(f"    内容: {wm_item.content[:60]}...")
    print(f"    类型: {wm_item.memory_type}")
    print()

    # 再写入一条工作记忆
    await manager.remember(
        content="用户询问了关于 aiohttp 库的使用方法，想实现并发 HTTP 请求",
        memory_type="working",
        session_id=session_id,
    )
    print("    写入第二条工作记忆: aiohttp 相关")
    print()

    # ===================================================================
    # 4. 写入情节记忆 —— 已完结的对话片段
    # ===================================================================
    # 情节记忆用于长期存储对话或事件，基于语义向量检索。
    ep_item = await manager.remember(
        content=(
            "今天用户问了关于 asyncio 的问题，从基础概念到实际项目应用。"
            "用户之前有 C# 开发经验，对 .NET 的 async/await 比较熟悉，"
            "正在将知识迁移到 Python 的异步编程生态。"
        ),
        memory_type="episodic",
        session_id=session_id,
    )
    print(f"[4] 写入情节记忆: id={ep_item.id}")
    print(f"    内容: {ep_item.content[:80]}...")
    print()

    # 再写一条情节记忆（用于后续的 consolidate 演示）
    await manager.remember(
        content=(
            "用户偏好使用 VSCode 编辑器进行 Python 开发，已安装了 Python、"
            "Pylance、Black Formatter 等扩展。用户位于北京，从事金融科技行业。"
        ),
        memory_type="episodic",
        session_id=session_id,
    )
    print("    写入第二条情节记忆: 用户偏好信息")
    print()

    # ===================================================================
    # 5. 写入语义记忆 —— 持久化知识实体
    # ===================================================================
    # 语义记忆使用 LLM 从文本中提取实体、偏好、关系，并持久化存储。
    # 适合存储用户偏好、事实知识等需要长期保留的信息。
    sem_item = await manager.remember(
        content="用户偏好：Python 编程语言，日常使用 VSCode 编辑器，喜欢喝咖啡",
        memory_type="semantic",
    )
    print(f"[5] 写入语义记忆: id={sem_item.id}")
    print(f"    内容: {sem_item.content[:80]}...")
    print()

    # ===================================================================
    # 6. 检索工作记忆 —— 单类型检索
    # ===================================================================
    working_results = await manager.recall(
        query="异步编程",
        memory_type="working",
        session_id=session_id,
        top_k=5,
    )
    print(f"[6] 检索工作记忆（查询: '异步编程'）: 找到 {len(working_results)} 条")
    for i, item in enumerate(working_results):
        print(f"    [{i+1}] {item.content[:60]}...")
    print()

    # ===================================================================
    # 7. 跨类型检索 —— 并行搜索所有记忆类型
    # ===================================================================
    # memory_type=None 时，会并行检索 working、episodic、semantic 三种类型，
    # 并将工作记忆结果排在向量检索结果之前（更高优先级）。
    all_results = await manager.recall(
        query="编程相关",
        memory_type=None,  # 检索所有类型
        top_k=5,
        session_id=session_id,
    )
    print(f"[7] 跨类型检索（查询: '编程相关'）: 找到 {len(all_results)} 条")
    for i, item in enumerate(all_results):
        print(f"    [{i+1}] type={item.memory_type.value} | {item.content[:60]}...")
    print()

    # ===================================================================
    # 8. 触发记忆整合 —— 从情节记忆中提取知识更新语义记忆
    # ===================================================================
    # consolidate 会从近期（默认 24 小时）情节记忆中提取实体、偏好和关系，
    # 然后与语义记忆中的已有实体进行合并或新建。
    print("[8] 触发记忆整合...")
    consolidate_result: ConsolidateResult = await manager.consolidate(
        time_window_hours=24,
        dry_run=False,
    )
    print("    整合结果:")
    print(f"    - 处理情节数: {consolidate_result.episodes_processed}")
    print(f"    - 新建实体: {consolidate_result.new_entities}")
    print(f"    - 更新实体: {consolidate_result.updated_entities}")
    print(f"    - 新建偏好: {consolidate_result.new_preferences}")
    print(f"    - 更新偏好: {consolidate_result.updated_preferences}")
    print(f"    - 新建关系: {consolidate_result.new_relations}")
    if consolidate_result.errors:
        print(f"    - 错误: {consolidate_result.errors}")
    print()

    # 验证整合后可以检索到新创建的语义实体
    semantic_results = await manager.recall(
        query="用户偏好",
        memory_type="semantic",
        top_k=3,
    )
    print(f"    整合后检索语义记忆（查询: '用户偏好'）: 找到 {len(semantic_results)} 条")
    for i, item in enumerate(semantic_results):
        print(f"    [{i+1}] {item.content[:80]}...")
    print()

    # ===================================================================
    # 9. 清理会话 —— 清空当前会话的工作记忆
    # ===================================================================
    cleared = await manager.clear_session(session_id)
    print(f"[9] 清理会话 '{session_id}': 清除了 {cleared} 条工作记忆")
    print()

    # 验证清理后工作记忆不再可检索
    after_clear = await manager.recall(
        query="异步",
        memory_type="working",
        session_id=session_id,
    )
    print(f"    验证清理后工作记忆: 剩余 {len(after_clear)} 条")
    print()

    print("=" * 60)
    print("  示例执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
