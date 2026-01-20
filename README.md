# GraphRAG 知识图谱检索系统

基于 Neo4j 图数据库和混合检索（向量搜索 + 图遍历）的知识图谱检索系统，集成 OpenAI/DeepSeek API 和 MCP 服务，支持多轮对话和智能问答。

## 功能特性

- 🔍 **混合检索**：结合向量搜索和图遍历，实现精准的实体定位和关系挖掘
- 🧠 **智能问答**：集成 OpenAI/DeepSeek API，支持多轮对话和函数调用
- 📊 **图数据库**：基于 Neo4j 存储和管理知识图谱
- 🚀 **MCP 集成**：通过 FastMCP 提供工具调用接口
- ⚡ **性能优化**：模型懒加载、缓存机制，提升响应速度
- 🎯 **NER 支持**：可选的自然语言实体识别功能

## 项目结构

```
chinadaily/
├── LLM.py                    # OpenAI 与 MCP 服务集成客户端
├── mcp_server.py             # MCP 服务器，提供图检索工具
├── graph_rag_retriever.py   # 混合图检索器核心实现
├── preload_model.py         # 模型预加载脚本
├── create/                  # 数据导入和索引构建
│   ├── ingest_to_neo4j.py  # 数据导入到 Neo4j
│   ├── build_index.py      # 构建向量索引
│   └── quick_start.py      # 快速启动脚本
├── spider/                  # 数据爬取模块
└── requirements.txt         # 项目依赖
```

## 环境要求

- Python 3.10+
- Neo4j 5.0+
- 8GB+ 内存（推荐）

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd chinadaily
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 spacy 模型（可选，用于 NER）

```bash
python -m spacy download en_core_web_sm
```

### 4. 配置 Neo4j

确保 Neo4j 服务正在运行，默认配置：
- URI: `bolt://localhost:7687`
- 用户名: `neo4j`
- 密码: `password123`（请根据实际情况修改）

### 5. 配置环境变量

创建 `.env` 文件：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

## 快速开始

### 1. 导入数据到 Neo4j

```bash
python create/ingest_to_neo4j.py
```

### 2. 构建索引

```bash
python create/build_index.py
```

### 3. 预加载模型（可选，提升首次查询速度）

```bash
python preload_model.py
```

### 4. 运行对话测试

```bash
python LLM.py
```

## 使用示例

### 基本检索

```python
from graph_rag_retriever import HybridGraphRetriever

retriever = HybridGraphRetriever(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password123",
    top_k_entities=10,
    graph_depth=2,
    lazy_load_model=True
)

result = retriever.retrieve("查询问题")
print(result['context'])
```

### 使用 MCP 客户端

```python
from LLM import MCPOpenAIClient
import asyncio

async def main():
    client = MCPOpenAIClient(
        openai_api_key="your_api_key",
        openai_base_url="https://api.deepseek.com",
        openai_model="deepseek-chat",
        mcp_server_path="mcp_server"
    )
    
    answer = await client.query("你的问题")
    print(answer)
    
    await client.close()

asyncio.run(main())
```

### 多轮对话

```python
history = [
    {"role": "system", "content": "你是一个专业的知识问答助手..."},
    {"role": "user", "content": "第一个问题"}
]

answer, history = await client.chat_with_history(history)
# 继续对话
history.append({"role": "user", "content": "第二个问题"})
answer, history = await client.chat_with_history(history)
```

## 核心组件说明

### HybridGraphRetriever

混合图检索器，实现三步检索流程：

1. **实体定位**：使用向量搜索和 NER 提取关键实体
2. **图扩展**：从实体出发，在图中扩展相关关系和属性
3. **上下文格式化**：对检索结果进行关联度排序和格式化

主要参数：
- `top_k_entities`: 向量搜索返回的实体数量（默认 20）
- `graph_depth`: 图遍历深度（默认 2）
- `lazy_load_model`: 是否懒加载模型（默认 True）
- `use_ner`: 是否使用 NER（默认 True）

### MCPOpenAIClient

OpenAI 与 MCP 服务集成客户端，支持：

- 自动工具发现和转换
- 函数调用循环处理
- 多轮对话历史管理
- 异步操作支持

## 配置说明

### Neo4j 连接配置

在 `mcp_server.py` 或 `graph_rag_retriever.py` 中修改：

```python
graph_rag_retriever = HybridGraphRetriever(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",
    ...
)
```

### 模型配置

默认使用 `sentence-transformers/all-MiniLM-L6-v2`，可在初始化时修改：

```python
retriever = HybridGraphRetriever(
    embedding_model_name="your-model-name",
    ...
)
```
### 实际运行示例

运行 `python LLM.py` 后，系统会启动交互式对话界面。以下是一个实际运行示例：
<img width="1480" height="1346" alt="image" src="https://github.com/user-attachments/assets/2d6c5bd5-03e6-4d21-897e-6e18e2917be4" />


## 性能优化

1. **模型缓存**：相同模型只加载一次，多个实例共享
2. **懒加载**：模型在首次使用时才加载，加快启动速度
3. **关键词过滤**：在 Cypher 查询层面过滤无关边，减少数据传输
4. **稀有实体优先**：优先匹配度数小的实体，避免超级节点

