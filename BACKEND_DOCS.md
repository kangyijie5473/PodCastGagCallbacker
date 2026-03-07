# PodCastGagCallbacker 后端技术文档

本文档详细介绍了 `src` 目录下后端代码的系统架构、核心模块功能及数据流向。

## 1. 系统概述

本项目是一个针对播客内容的智能检索与问答系统（RAG）。后端基于 Python 开发，集成了音频下载、语音识别（ASR）、说话人分离（Diarization）、向量索引与检索、以及大语言模型（LLM）增强问答功能。

### 核心能力
- **播客获取**: 支持小宇宙（XiaoYuZhou）链接解析及标准 RSS 订阅源下载。
- **音频处理**: 使用 FunASR 模型进行高精度中文语音识别，并支持多人说话人区分。
- **智能索引**: 对转录文本进行语义切片（Windowing）和向量化（Embedding）。
- **语义检索**: 支持基于语义的跨播客、跨单集搜索。
- **AI 问答 (RAG)**: 基于检索结果，利用 LLM 生成针对用户问题的精准回答。

---

## 2. 系统架构

系统采用分层架构设计，主要包含以下层级：

- **接口层 (Interface Layer)**:
  - `cli.py`: 命令行工具，提供下载、索引、搜索等离线操作入口。
  - `server.py`: FastAPI Web 服务，提供 RESTful API，支持前端交互、文件上传及后台任务管理。

- **服务层 (Service Layer)**:
  - `PodcastDownloader`: 负责播客音频及元数据的爬取与下载。
  - `IndexingService`: 核心处理流水线（音频 -> 文本 -> 向量）。
  - `SearchService`: 负责向量索引的加载与检索。
  - `RAGService`: 负责检索增强生成，组装上下文并调用 LLM。
  - `CollectorService`: 辅助服务，用于管理已索引的数据集合。

- **模型层 (Model Layer)**:
  - `FasterWhisperASR`: 封装 Faster Whisper 框架进行 ASR，集成 Pyannote.audio 进行说话人分离。
  - `LocalEmbedding`: 封装本地 Embedding 模型（如 BGE-Small）。
  - `OpenAILLM`: 封装 OpenAI 兼容格式的 LLM 接口。

- **数据层 (Data Layer)**:
  - 基于文件系统的存储结构，无需额外数据库。
  - 目录结构：`data/{podcast_name}/{audio_id}/`。

---

## 3. 核心模块详解

### 3.1 Server (`src/server.py`)
基于 FastAPI 构建的 Web 服务器，主要职责包括：
- **生命周期管理**: 在启动时加载 ASR、Embedding 和 LLM 模型，初始化各服务单例。
- **API 路由**:
  - `POST /api/search`: 搜索接口，支持普通向量检索和 RAG 问答模式。
  - `POST /api/upload`: 处理用户上传的音频文件，并触发后台索引任务。
  - `POST /api/podcast/submit`: 接收播客链接提交，触发后台下载任务。
  - `GET /api/podcasts`: 返回当前系统内已索引的播客与单集列表（层级结构）。
  - `GET /api/tasks/{task_id}`: 查询后台任务（如上传处理）的进度与状态。
- **后台任务**: 利用 `BackgroundTasks` 处理耗时的音频转录与索引工作，避免阻塞 API 响应。

### 3.2 IndexingService (`src/services/indexer.py`)
音频处理的核心流水线，处理流程如下：
1.  **预估**: 计算音频时长，预估处理耗时（基于 RTF ~ 0.2）。
2.  **转录 (Transcribe)**: 调用 `FunASR` 模型，输出包含时间戳、文本和说话人 ID 的原始片段 (`segments`)。
3.  **切片 (Windowing)**: 使用滑动窗口算法（默认窗口大小 20，步长 5）将原始片段组合成语义完整的文本块 (`windows`)。
    - 每个 Window 包含合并后的文本、起止时间、说话人集合。
4.  **LLM 润色 (Refinement)** (可选): 调用 LLM 修复转录文本中的标点、错别字及语气词，提升可读性。
5.  **向量化 (Embedding)**: 对 Window 文本进行向量编码。
6.  **存储**: 将结果保存为 `segments.json` (原始), `windows.json` (切片), `embeddings.npy` (向量)。

### 3.3 SearchService (`src/services/searcher.py`)
负责索引数据的检索：
- **索引加载**: 按需加载 `data` 目录下的索引文件到内存。
- **向量检索**:
  1. 将用户 Query 转换为向量。
  2. 计算 Query 向量与所有加载 Window 向量的余弦相似度。
  3. 排序并返回 Top-K 结果。
- **过滤**: 支持按 `podcast_name` 和 `audio_id` 进行精确过滤。

### 3.4 RAGService (`src/services/rag.py`)
实现检索增强生成：
1.  **检索**: 调用 `SearchService` 获取相关上下文片段。
2.  **上下文组装**: 将片段格式化为带有元数据（播客名、时间、说话人）的文本块。
    - *优化*: 如果可能，会回溯原始 `segments` 以提供更精确的逐句对话还原。
3.  **Prompt 构建**: 构建包含 System Prompt 和 User Prompt 的指令，要求 LLM 仅依据上下文回答，并引用来源。
4.  **生成**: 调用 LLM 生成最终答案。

### 3.5 PodcastDownloader (`src/services/downloader.py`)
- **小宇宙解析**: 针对 `xiaoyuzhoufm.com` 链接，通过正则提取 `__NEXT_DATA__` 中的 JSON 数据，直接获取高清音频地址和元数据，无需依赖外部 RSSHub。
- **RSS 解析**: 使用 `feedparser` 解析标准播客 RSS Feed。
- **下载管理**: 自动去重（检查文件是否存在及大小），支持断点续传（通过 HTTP Range 实际上主要由 `requests` 流式处理），显示实时下载进度条。

### 3.6 Models (`src/models/`)
- **funasr.py**:
  - 集成 ModelScope 的 `AutoModel`。
  - 启用 `merge_vad_diar=True`，实现 VAD 切分与声纹聚类的端到端输出。
  - 参数配置：`batch_size_s=300` (推理批次时长), `max_speakers=15` (最大说话人数预设)。

---

## 4. 数据存储结构

系统所有数据存储在 `data` 目录下，结构如下：

```text
data/
├── {podcast_name}/           # 播客栏目名称 (归一化后)
│   ├── {audio_id}/           # 单集 ID (通常为 日期_标题)
│   │   ├── segments.json     # [关键] ASR 原始转录结果 (含说话人、精确时间戳)
│   │   ├── windows.json      # [关键] 语义切片结果 (含合并文本、Start/End)
│   │   └── embeddings.npy    # [关键] 向量数据 (NumPy 数组)
│   └── ...
└── ...
```

用户上传的文件会默认归类为 `user_uploads` 栏目（未来可扩展为 `user_{id}`）。

## 5. 依赖环境

- **Python**: 3.9+ (推荐 3.10/3.11)
- **核心库**:
  - `fastapi`, `uvicorn`: Web 服务
  - `funasr`, `modelscope`, `torch`: 语音识别与深度学习
  - `sentence-transformers` (或类似): 向量化
  - `numpy`: 数值计算
  - `librosa`: 音频处理
  - `requests`, `feedparser`: 网络请求与解析
