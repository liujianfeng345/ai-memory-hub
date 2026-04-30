# ai-memory-hub 集成到 LangGraph 使用指南

## 概述

`ai-memory-hub` 可以与 LangGraph 无缝集成，为 Agent 提供持久的记忆能力。本指南提供三个层次的使用示例：基础集成、进阶记忆策略、以及完整的记忆感知 Agent。

## 安装

```bash
pip install ai-memory-hub langgraph langchain-core
```

## 架构总览

```
┌─────────────────────────────────────────────┐
│              LangGraph Agent                 │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ remember │  │  recall  │  │consolidate│  │
│  │   node   │  │   node   │  │   node    │  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
│       │              │              │        │
└───────┼──────────────┼──────────────┼────────┘
        ▼              ▼              ▼
┌──────────────────────────────────────────────┐
│          MemoryManager（ai-memory-hub）        │
│                                               │
│  ┌──────────────┐ ┌──────────┐ ┌───────────┐ │
│  │WorkingMemory │ │Episodic  │ │ Semantic  │ │
│  │ (短期/会话)   │ │ Memory   │ │ Memory    │ │
│  │              │ │ (事件流)  │ │ (知识图谱) │ │
│  └──────────────┘ └──────────┘ └───────────┘ │
└──────────────────────────────────────────────┘
```

---

## 示例一：基础集成 —— 带会话记忆的对话 Agent

这是最常用的模式：每次对话前检索相关历史，对话后存储新记忆。

```python
"""
langgraph_basic.py —— 带记忆的 LangGraph 对话 Agent

运行前设置环境变量：
    export DEEPSEEK_API_KEY="sk-your-key"
"""

import asyncio
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from memory_agent import MemoryManager, MemoryConfig


# ─── 1. 定义 Agent 状态 ─────────────────────────────────

class AgentState(TypedDict):
    messages: List[dict]       # 当前对话历史
    session_id: str            # 会话标识
    user_input: str            # 用户最新输入
    memories: Optional[str]     # 检索到的相关记忆（注入到 prompt）
    response: Optional[str]     # Agent 回复


# ─── 2. 初始化 MemoryManager ───────────────────────────

config = MemoryConfig()
memory = MemoryManager(config)


# ─── 3. 定义 LangGraph 节点 ─────────────────────────────

async def recall_memories(state: AgentState) -> dict:
    """节点 1：检索相关记忆，注入上下文"""
    result = await memory.recall(
        query=state["user_input"],
        memory_type=None,           # 检索所有类型记忆
        session_id=state["session_id"],
        top_k=5,
    )

    # 将记忆拼接为上下文文本
    context_parts = []
    for item in result:
        context_parts.append(f"[{item.memory_type.value}] {item.content}")

    memories_text = "\n".join(context_parts) if context_parts else "(无相关历史记忆)"
    return {"memories": memories_text}


async def generate_response(state: AgentState) -> dict:
    """节点 2：结合记忆生成回复（这里用 DeepSeek 演示）"""
    system_prompt = f"""你是一个有记忆的 AI 助手。以下是与你之前对话相关的记忆：

{state["memories"]}

请根据这些记忆和当前对话上下文，自然地回复用户。如果用户提到之前的对话内容，你应该能关联起来。"""

    messages = [
        {"role": "system", "content": system_prompt},
        *state["messages"],
        {"role": "user", "content": state["user_input"]},
    ]

    # 使用 MemoryManager 内置的 DeepSeek 客户端
    response_text = await memory._llm_client.chat(
        messages=messages,
        temperature=0.7,
    )
    return {"response": response_text}


async def remember_this_turn(state: AgentState) -> dict:
    """节点 3：将本轮对话存入记忆"""
    user_msg = state["user_input"]
    assistant_msg = state.get("response", "")

    # 工作记忆：短期记住本轮对话要点
    await memory.remember(
        content=f"用户: {user_msg}",
        memory_type="working",
        session_id=state["session_id"],
    )

    # 情景记忆：存储完整交互
    await memory.remember(
        content=f"用户说: {user_msg}\n助手回复: {assistant_msg}",
        memory_type="episodic",
        session_id=state["session_id"],
    )

    return {}


# ─── 4. 构建 Graph ──────────────────────────────────────

def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("recall", recall_memories)
    builder.add_node("generate", generate_response)
    builder.add_node("remember", remember_this_turn)

    builder.set_entry_point("recall")
    builder.add_edge("recall", "generate")
    builder.add_edge("generate", "remember")
    builder.add_edge("remember", END)

    # 无需 checkpointer：ai-memory-hub 自身管理持久化记忆，LangGraph 只负责流程编排
    return builder.compile()


# ─── 5. 运行 ────────────────────────────────────────────

async def main():
    agent = build_agent()
    session_id = "user-session-001"

    print("=" * 60)
    print("  带记忆的 LangGraph 对话 Agent")
    print("  输入 'quit' 退出，输入 'consolidate' 触发记忆整合")
    print("=" * 60)

    messages_history = []

    while True:
        user_input = input("\n👤 你: ").strip()

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "consolidate":
            result = await memory.consolidate(session_id=session_id)
            print(f"\n📦 记忆整合完成：新增 {result.new_entities} 实体, "
                  f"{result.new_preferences} 偏好, {result.new_relations} 关系")
            continue

        initial_state: AgentState = {
            "messages": messages_history[-10:],  # 保留最近 10 轮
            "session_id": session_id,
            "user_input": user_input,
            "memories": None,
            "response": None,
        }

        result = await agent.ainvoke(initial_state)

        # 更新对话历史
        messages_history.append({"role": "user", "content": user_input})
        messages_history.append({"role": "assistant", "content": result["response"]})

        print(f"\n🤖 助手: {result['response']}")

        # 显示检索到的记忆
        if result["memories"] and result["memories"] != "(无相关历史记忆)":
            print(f"\n📝 关联记忆:\n{result['memories'][:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 示例二：进阶 —— 多节点记忆策略

对于复杂 Agent，不同节点往往需要不同的记忆类型。下面的例子展示了如何在多节点流程中有针对性地使用三种记忆类型。

```python
"""
langgraph_advanced.py —— 多节点记忆策略

场景：客服 Agent，需要记住用户偏好（语义记忆）、本次会话上下文（工作记忆）、
以及历史工单（情景记忆）。
"""

import asyncio
from typing import TypedDict, List, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from memory_agent import MemoryManager, MemoryConfig, MemoryItem


class TicketState(TypedDict):
    user_id: str
    session_id: str
    user_input: str
    intent: Optional[str]         # 意图分类
    user_profile: Optional[str]   # 用户画像（来自语义记忆）
    recent_context: Optional[str] # 近期上下文（来自工作记忆）
    similar_cases: Optional[str]  # 相似工单（来自情景记忆）
    response: Optional[str]


config = MemoryConfig()
memory = MemoryManager(config)


# ─── 节点 1：并行检索所有记忆类型 ──────────────────────

async def gather_context(state: TicketState) -> dict:
    """从三种记忆来源搜集上下文"""
    query = state["user_input"]

    # 并行检索（MemoryManager 内部使用 asyncio.gather）
    all_memories = await memory.recall(
        query=query,
        memory_type=None,  # 全部类型
        session_id=state["session_id"],
        top_k=3,
    )

    # 按类型分类
    working_items = [it for it in all_memories if it.memory_type.value == "working"]
    episodic_items = [it for it in all_memories if it.memory_type.value == "episodic"]
    semantic_items = [it for it in all_memories if it.memory_type.value == "semantic"]

    return {
        "recent_context": "\n".join(it.content for it in working_items[:3]),
        "similar_cases": "\n".join(it.content for it in episodic_items[:2]),
        "user_profile": "\n".join(it.content for it in semantic_items),
    }


# ─── 节点 2：意图分类（利用语义记忆中的用户偏好） ───────

async def classify_intent(state: TicketState) -> dict:
    """根据用户画像和当前输入判断意图"""
    profile = state.get("user_profile", "")
    user_input = state["user_input"]

    prompt = f"""根据用户画像和输入判断意图。

用户画像：
{profile if profile else "新用户，无历史画像"}

当前输入：{user_input}

可能的意图：refund（退款）、technical_support（技术支持）、
complaint（投诉）、inquiry（咨询）、account_issue（账号问题）

只回复意图标签，不要解释。"""

    intent = await memory._llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=50,
    )
    return {"intent": intent.strip()}


# ─── 节点 3：生成回复 ──────────────────────────────────

async def handle_ticket(state: TicketState) -> dict:
    intent = state.get("intent", "inquiry")
    similar = state.get("similar_cases", "")

    prompt = f"""你是客服助手。当前意图：{intent}

历史相似工单参考：
{similar if similar else "无相似工单"}

用户近期上下文：
{state.get('recent_context', '无')}

请根据以上信息回复用户：{state['user_input']}"""

    response = await memory._llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return {"response": response}


# ─── 节点 4：写入记忆（根据意图选择不同策略） ───────────

async def save_context(state: TicketState) -> dict:
    user_input = state["user_input"]
    response = state.get("response", "")
    intent = state.get("intent", "")

    # 工作记忆：总是写入（短期会话上下文）
    await memory.remember(
        content=f"[{intent}] 用户: {user_input}",
        memory_type="working",
        session_id=state["session_id"],
    )

    # 情景记忆：仅写入有意义的交互
    if intent in ("complaint", "technical_support"):
        await memory.remember(
            content=f"工单类型={intent}\n用户输入: {user_input}\n回复: {response}\n时间: {datetime.utcnow().isoformat()}",
            memory_type="episodic",
            session_id=state["session_id"],
            metadata={"intent": intent, "status": "resolved"},
        )

    # 语义记忆：如果检测到用户偏好信息，提取为实体
    if any(kw in user_input for kw in ("喜欢", "偏好", "常用", "习惯")):
        await memory.remember(
            content=f"用户偏好提取: {user_input}",
            memory_type="semantic",
        )

    return {}


# ─── 节点 5：定期整合 ──────────────────────────────────

async def periodic_consolidate(state: TicketState) -> dict:
    """每 N 次对话后自动整合情景记忆到语义记忆"""
    # 检查是否需要整合（例如基于计数）
    working_count = await memory._working_memory.get_by_session(state["session_id"])
    if len(working_count) > 0 and len(working_count) % 10 == 0:
        result = await memory.consolidate(
            session_id=state["session_id"],
            time_window_hours=24,
            dry_run=False,
        )
        print(f"[自动整合] 新增实体 {result.new_entities}, "
              f"偏好 {result.new_preferences}")
    return {}


# ─── 构建 Graph ─────────────────────────────────────────

def build_ticket_agent():
    builder = StateGraph(TicketState)

    builder.add_node("gather_context", gather_context)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("handle_ticket", handle_ticket)
    builder.add_node("save_context", save_context)
    builder.add_node("periodic_consolidate", periodic_consolidate)

    builder.set_entry_point("gather_context")
    builder.add_edge("gather_context", "classify_intent")
    builder.add_edge("classify_intent", "handle_ticket")
    builder.add_edge("handle_ticket", "save_context")
    builder.add_edge("save_context", "periodic_consolidate")
    builder.add_edge("periodic_consolidate", END)

    return builder.compile()


async def main():
    agent = build_ticket_agent()
    session_id = f"ticket-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"

    test_inputs = [
        "我的账号登录不了了，一直提示密码错误",
        "我平时习惯用 Chrome 浏览器登录",
        "上次你们说会发邮件重置链接但我没收到",
        "算了我不投诉了，帮我退款就行",
    ]

    for user_input in test_inputs:
        state: TicketState = {
            "user_id": "user-001",
            "session_id": session_id,
            "user_input": user_input,
            "intent": None,
            "user_profile": None,
            "recent_context": None,
            "similar_cases": None,
            "response": None,
        }

        result = await agent.ainvoke(state)
        print(f"\n{'='*60}")
        print(f"👤 用户: {user_input}")
        print(f"🎯 意图: {result['intent']}")
        print(f"🤖 回复: {result['response'][:200]}...")

    # 最后执行一次整合
    consolidate_result = await memory.consolidate(session_id=session_id)
    print(f"\n📦 最终整合: {consolidate_result}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 示例三：Tool-calling Agent —— StateGraph + ToolNode 模式

现代的 LangGraph tool-calling 模式是 `StateGraph` + `ToolNode` + `tools_condition`。下例展示如何将记忆操作包装为工具，让 Agent 在对话中自主决定何时记住、何时检索。

```python
"""
langgraph_tools.py —— 使用 StateGraph + ToolNode 构建记忆感知 Agent

运行前设置环境变量：
    export DEEPSEEK_API_KEY="sk-your-key"
"""

import asyncio
import os
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from memory_agent import MemoryManager, MemoryConfig


# ─── 初始化 ─────────────────────────────────────────────

config = MemoryConfig()
memory = MemoryManager(config)
SESSION_ID = "tool-agent-session"


# ─── 1. 定义记忆工具 ────────────────────────────────────

@tool
async def remember_user_info(info: str) -> str:
    """记住用户告诉你的个人信息、偏好或重要事实。例如用户说"我叫张三"或"我喜欢喝咖啡"时调用。
    info: 需要记住的内容"""
    await memory.remember(
        content=info,
        memory_type="episodic",
        session_id=SESSION_ID,
    )
    return f"已记住: {info}"


@tool
async def recall_past_conversation(query: str) -> str:
    """检索与当前话题相关的历史对话。当用户问"我们之前聊过什么"或需要回顾上下文时调用。
    query: 检索关键词"""
    results = await memory.recall(
        query=query,
        memory_type=None,
        session_id=SESSION_ID,
        top_k=5,
    )
    if not results:
        return "没有找到相关历史记忆。"

    lines = []
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. [{item.memory_type.value}] {item.content}")
    return "\n".join(lines)


@tool
async def add_user_preference(preference: str) -> str:
    """提取并存储用户的长期偏好或习惯。例如"用户喜欢 Python"、"常用 VSCode"。
    preference: 偏好描述"""
    await memory.remember(
        content=preference,
        memory_type="semantic",
    )
    return f"已记录偏好: {preference}"


@tool
async def build_knowledge_graph() -> str:
    """将近期对话中的所有信息整合为结构化知识图谱。当你需要全面了解用户时调用。"""
    result = await memory.consolidate(session_id=SESSION_ID, time_window_hours=24)
    return (f"知识整合完成：新增 {result.new_entities} 个实体，"
            f"{result.new_preferences} 个偏好，{result.new_relations} 条关系。")


# ─── 2. 定义 State ─────────────────────────────────────

class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


# ─── 3. 构建 Tool-calling Agent（StateGraph + ToolNode） ─

def build_agent():
    """使用 StateGraph + ToolNode 构建 tool-calling agent"""
    tools = [
        remember_user_info,
        recall_past_conversation,
        add_user_preference,
        build_knowledge_graph,
    ]

    # 使用 ChatOpenAI 对接 DeepSeek API
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.7,
    ).bind_tools(tools)

    # 系统提示词
    system_prompt = SystemMessage(content=(
        "你是一个有长期记忆的 AI 助手。对话时请注意：\n"
        "1. 当用户分享个人信息或偏好时，调用 remember_user_info 或 add_user_preference 记录下来\n"
        "2. 当用户提到之前聊过的话题时，先调用 recall_past_conversation 检索历史\n"
        "3. 当你需要全面了解用户时，调用 build_knowledge_graph 整合知识\n"
        "4. 回复时自然地引用你记住的信息，让用户感受到你是'记得'他的"
    ))

    async def agent_node(state: MessagesState) -> dict:
        """LLM 决策节点：决定是否调用工具或直接回复"""
        response = await llm.ainvoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    # 构建 Graph
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("agent")

    # 条件路由：如果 LLM 要调用工具 → tools 节点；否则 → END
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")  # 工具结果返回 agent 继续决策

    return builder.compile(checkpointer=MemorySaver())


# ─── 4. 运行演示 ────────────────────────────────────────

async def main():
    agent = build_agent()

    # 配置：thread_id 作为 LangGraph 的会话标识
    config = {"configurable": {"thread_id": SESSION_ID}}

    print("=" * 60)
    print("  有记忆的 Tool-calling Agent")
    print("  Agent 会自主决定何时调用记忆工具")
    print("=" * 60)

    # 第一轮：用户介绍自己
    user_input = "你好！我叫李明，今年28岁，是一名后端工程师，平时喜欢用 Go 和 Rust"
    print(f"\n👤 用户: {user_input}")
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    )
    print(f"🤖 助手: {result['messages'][-1].content}")

    # 第二轮：用户询问偏好
    user_input = "帮我看看我常用的编程语言是什么？"
    print(f"\n👤 用户: {user_input}")
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    )
    print(f"🤖 助手: {result['messages'][-1].content}")

    # 第三轮：用户问之前聊过什么
    user_input = "我们之前都聊了什么？帮我总结一下"
    print(f"\n👤 用户: {user_input}")
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    )
    print(f"🤖 助手: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 最佳实践

### 1. 记忆类型的选择

| 场景 | 记忆类型 | 典型 TTL |
|------|---------|---------|
| 当前会话上下文（用户刚说的） | `working` | 会话结束即过期 |
| 对话历史、工单记录 | `episodic` | 持久化（向量检索） |
| 用户画像、偏好、长期知识 | `semantic` | 持久化（知识图谱） |

### 2. 整合时机

不要每轮对话都调用 `consolidate()`。建议：

- 每 10-20 轮对话触发一次
- 会话结束时触发一次
- 用户主动要求"总结"时触发

### 3. 性能建议

- `MemoryManager` 使用懒加载：模型在首次 `remember`/`recall` 时才加载，初始化本身很快
- `recall(memory_type=None)` 内部使用 `asyncio.gather` 并行检索三种记忆源
- ChromaDB 不支持多进程并发写入，同一实例不要在多进程中共享
- 如需高并发场景，考虑为每个 Agent 实例创建独立的 `MemoryManager`

### 4. 会话管理

```python
# 会话结束时清理工作记忆
await memory.clear_session(session_id)

# 定期清理过期工作记忆（避免内存泄漏）
await memory._working_memory._store.cleanup_expired()
```

---

## 环境变量参考

```bash
# 必需
DEEPSEEK_API_KEY=sk-your-key

# 可选（以下为默认值）
LOG_LEVEL=INFO
EMBEDDING_MODEL_NAME=models/bge-small-zh-v1.5
CHROMA_PERSIST_DIR=./data/chroma
DEFAULT_TTL_SECONDS=3600
MAX_CONTENT_LENGTH=50000
```

## 相关链接

- [PyPI - ai-memory-hub](https://pypi.org/project/ai-memory-hub/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
