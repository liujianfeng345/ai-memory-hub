# 开发环境准备清单

## 必须安装的软件

- [ ] **Python**：3.10 或更高版本
  - 验证：`python --version`
  - 下载：https://www.python.org/downloads/

- [ ] **pip**：23.0 或更高版本（Python 3.10+ 通常自带）
  - 验证：`pip --version`
  - 升级：`python -m pip install --upgrade pip`

- [ ] **Git**：2.30 或更高版本
  - 验证：`git --version`
  - 下载：https://git-scm.com/downloads

- [ ] **Visual C++ 构建工具**（Windows 必需，用于编译 ChromaDB 的 C 扩展）：
  - 下载：https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - 安装时勾选"Desktop development with C++"工作负载
  - 或者运行：`winget install Microsoft.VisualStudio.2022.BuildTools`

- [ ] **CUDA Toolkit**（可选，仅 GPU 加速时需要）：
  - 版本：CUDA 11.8 或 12.1（匹配 PyTorch 支持的版本）
  - 下载：https://developer.nvidia.com/cuda-downloads
  - 注意：需要 NVIDIA GPU（显存 >= 4GB 推荐）

## 环境变量（请保存于 `.env` 文件，切勿提交至版本库）

| 变量名 | 描述 | 如何获取 | 默认值 |
|--------|------|----------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 在 https://platform.deepseek.com/api_keys 创建 API Key；用户本机已配置此变量 | 无，必须配置 |
| `MEMORY_AGENT_DATA_DIR` | 记忆数据持久化目录 | 本地任意可写路径 | `./data` |
| `MEMORY_AGENT_LOG_LEVEL` | 日志级别 | 可选值：DEBUG / INFO / WARNING / ERROR | `INFO` |
| `MEMORY_AGENT_LOG_FORMAT` | 日志输出格式 | `json`（结构化）或 `text`（可读） | `text` |
| `EMBEDDING_MODEL_NAME` | 嵌入模型名称或路径 | 见下方说明 | `models/bge-small-zh-v1.5` |
| `EMBEDDING_DEVICE` | 嵌入模型推理设备 | `cpu` 或 `cuda` | `cpu` |
| `LLM_MODEL_NAME` | 大模型名称 | DeepSeek 支持的模型 | `deepseek-chat` |
| `LLM_BASE_URL` | LLM API 基础 URL | DeepSeek API 地址 | `https://api.deepseek.com/v1` |
| `LLM_TIMEOUT` | LLM 请求超时（秒） | 按需调整 | `30` |
| `LLM_MAX_RETRIES` | LLM 请求最大重试次数 | 按需调整 | `3` |
| `WORKING_MEMORY_TTL` | 工作记忆默认过期时间（秒） | 按需调整 | `3600` |

### 环境变量配置示例

在项目根目录创建 `.env` 文件：

```bash
# DeepSeek API（必需）
DEEPSEEK_API_KEY=sk-your-api-key-here

# 数据目录（可选，使用默认值即可）
MEMORY_AGENT_DATA_DIR=./data

# 日志配置（可选）
MEMORY_AGENT_LOG_LEVEL=INFO
MEMORY_AGENT_LOG_FORMAT=text

# 嵌入模型配置（可选，使用默认值即可）
EMBEDDING_MODEL_NAME=models/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu

# LLM 配置（可选，使用默认值即可）
LLM_MODEL_NAME=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TIMEOUT=30
LLM_MAX_RETRIES=3
```

## 第三方服务账号

- [ ] **DeepSeek**：需要有效的 API 密钥和可用额度
  - 注册地址：https://platform.deepseek.com
  - 获取 API Key：https://platform.deepseek.com/api_keys
  - 注意：用户本机环境变量中已配置 `DEEPSEEK_API_KEY`，如果本机已可用则无需额外操作

## Python 依赖

### 生产依赖（requirements.txt）

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `chromadb` | >=0.5.0, <1.0 | 向量数据库，本地嵌入式模式 |
| `sentence-transformers` | >=3.0.0 | 加载和运行 BGE 嵌入模型 |
| `httpx` | >=0.27.0 | 异步 HTTP 客户端，用于调用 DeepSeek API |
| `pydantic` | >=2.5.0 | 数据模型校验和配置管理 |
| `pydantic-settings` | >=2.1.0 | 从环境变量加载配置 |
| `python-dotenv` | >=1.0.0 | 加载 `.env` 文件 |
| `uuid6` | >=2024.0 | 生成 UUID（含时间戳信息，便于排序） |

### 开发依赖（requirements-dev.txt）

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `pytest` | >=8.0.0 | 测试框架 |
| `pytest-asyncio` | >=0.24.0 | 异步测试支持 |
| `pytest-cov` | >=5.0.0 | 测试覆盖率 |
| `ruff` | >=0.4.0 | 代码格式化和静态检查 |
| `mypy` | >=1.10.0 | 类型检查 |

### 关于 ChromaDB 的依赖说明

ChromaDB 在 Windows 环境下依赖 Microsoft Visual C++ 运行时。如果安装过程报错：
- 确保已安装 Visual C++ 构建工具（见上方"必须安装的软件"）
- 如果仍有问题，可尝试安装预编译的 wheel：`pip install chromadb --only-binary=:all:`

### 关于 sentence-transformers 的依赖说明

`sentence-transformers` 依赖 PyTorch。如果不需要 GPU 加速（推荐开发阶段使用 CPU）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 首次运行步骤

```bash
# 1. 克隆代码仓库
git clone <repository-url> memory-agent-2
cd memory-agent-2

# 2. 创建并激活 Python 虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. 安装生产依赖
pip install -r requirements.txt

# 4. 安装开发依赖（如需运行测试）
pip install -r requirements-dev.txt

# 5. 复制环境变量模板并填写
# Windows:
copy .env.example .env
# Linux/macOS:
cp .env.example .env
# 编辑 .env 文件，填入你的 DEEPSEEK_API_KEY

# 6. 验证安装
python -c "from memory_agent import MemoryManager; print('安装成功')"

# 7. 运行测试
pytest tests/ -v

# 8. 运行示例
python examples/basic_usage.py
```

## 数据目录结构

首次运行后，会在 `MEMORY_AGENT_DATA_DIR`（默认 `./data`）下自动创建以下目录：

```
data/
├── chroma/                      # ChromaDB 持久化数据
│   ├── chroma.sqlite3           # 元数据和索引
│   └── <uuid>/                  # 向量数据分片
├── models/                      # （可选）本地缓存的嵌入模型
└── logs/                        # （可选）日志文件
    └── memory_agent.log
```

注意：
- `data/chroma/` 目录包含所有记忆数据，删除该目录将丢失所有持久化记忆（工作记忆不受影响，因为工作记忆仅存内存）
- 建议定期备份 `data/chroma/` 目录
- 该目录已在 `.gitignore` 中排除，不会被提交到版本库

## 常见问题

### Q: 嵌入模型是否需要联网下载
A: 不需要。默认嵌入模型 `bge-small-zh-v1.5` 已预置在项目 `models/bge-small-zh-v1.5/` 目录下，`sentence-transformers` 可直接从本地路径加载。如需切换到其他模型（如 `bge-base-zh-v1.5`），可参考 7.4 节下载到 models/ 目录后修改 `EMBEDDING_MODEL_NAME` 配置。

### Q: 运行时报错 "DLL load failed"（Windows）
A: 缺少 Visual C++ 运行时。安装 [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)。

### Q: ChromaDB 数据损坏怎么办
A: 删除 `data/chroma/` 目录，重启应用会自动重建。注意这将丢失所有情景记忆和语义记忆数据。
