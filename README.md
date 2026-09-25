# 技术支持 AI 智能体（RAG 实现）

> **Technical Support AI Agent: A RAG-based Implementation**

一个可直接运行的 **检索增强生成（Retrieval-Augmented Generation, RAG）** 技术支持智能体，
面向企业 IT 服务台（IT Service Desk）场景。核心交付物是一个包含完整英文理论说明与
可运行代码的 Jupyter Notebook。

**需要通过 `.env` 配置一个 OpenAI 兼容端点。本项目没有离线模式。**

---

## 1. 交付物清单

| 文件 | 说明 |
|---|---|
| `Technical_Support_AI_Agent_RAG.ipynb` | **主交付物**。13 个章节、52 个单元格（34 个代码单元格），英文理论 + 可运行代码 |
| `Technical_Support_AI_Agent_RAG.html` | 同一份 Notebook 的 HTML 导出，双击即可在浏览器阅读，无需 Jupyter |
| `.env.example` | **配置模板** —— 复制为 `.env` 后填写 Key 与端点 |
| `tech_support.txt` | 技术支持知识库（10 个章节的 IT 服务台文档，8,732 字符 → 25 个检索块） |
| `tools/ask.py` | **命令行入口**：不打开 Jupyter 也能直接向智能体提问 |
| `tools/webui.py` | **浏览器测试界面**：本地聊天页，不依赖 Jupyter 内核 |
| `tools/run_notebook.py` | 无头执行脚本：命令行重跑 Notebook 并断言零报错 |
| `tools/build_notebook.py` | Notebook 生成脚本（声明式构建，改这里再重新生成） |
| `tools/setup_wsl.sh` | 一键复现 WSL 环境（建 venv + 装依赖 + 注册内核 + 生成 `.env`） |
| `requirements.txt` | 依赖清单 |
| `artifacts/` | Notebook 运行时生成的 5 张图表与持久化的 FAISS 向量库 |

## 2. 先配置 `.env`（必做）

```bash
cp .env.example .env
```

然后编辑 `.env`：

| 变量 | 是否必填 | 默认值 | 说明 |
|---|---|---|---|
| `OPENAI_API_KEY` | **必填** | — | 端点凭据。缺了这一项 Notebook 会在第 3 节直接停下并给出修复指引 |
| `OPENAI_BASE_URL` | 选填 | `https://api.openai.com/v1` | **指向任意 OpenAI 兼容网关**：官方 API、企业代理、自建 vLLM / Ollama / LiteLLM、第三方中转 |
| `OPENAI_EMBED_MODEL` | 选填 | `text-embedding-3-small` | 仅当 `EMBED_BACKEND=api` 时使用；必须是该端点提供的 **embeddings** 模型 |
| `EMBED_BACKEND` | 选填 | `local` | `api` = 走端点嵌入；`local` = **进程内**用 `sentence-transformers`（当前默认。不花钱，也不依赖端点的 `/embeddings` 路由） |
| `LOCAL_EMBED_MODEL` | 选填 | `BAAI/bge-m3` | 仅当 `EMBED_BACKEND=local` 时使用。**多语言模型**：中/俄/英映射到同一向量空间，所以能用中文或俄文提问英文知识库。首次自动下载约 2.3 GB，之后完全离线 |
| `OPENAI_CHAT_MODEL` | 选填 | `gpt-4o-mini` | 必须是该端点提供的 **chat** 模型 |
| `OPENAI_TEMPERATURE` | 选填 | `0` | `0` 才能保证技术支持回答可复现 |

真实环境变量（shell export、CI secret）**优先级高于 `.env`**，不会被本地文件覆盖。

### 两个必须知道的坑

1. **很多第三方中转只代理 chat，`/embeddings` 返回 404。** 第 3.5 节的预检会把这种情况
   单独识别出来并明确告诉你，而不是等到第 6 节抛一个看不懂的栈。
2. **`OPENAI_EMBED_MODEL` 在建索引和查询时必须完全一致。** 换了模型之后，查询向量和
   库里的分块向量不在同一个空间里，检索会静默失效。改了就删掉 `artifacts/vector_store/` 重跑。

## 3. 如何使用

本项目有五种用法，按"介入程度"从低到高排列。

### 3.0 运行位置：WSL（推荐）

项目已部署在 WSL 里，**推荐在那儿跑** —— 环境是干净的 Ubuntu 24.04 + Python 3.12，
不受 Windows 侧各种环境变量、代理和编码问题干扰：

```bash
# 在 WSL 终端里
cd ~/rag-tech-support-agent
.venv/bin/python -V                     # Python 3.12.3
.venv/bin/python tools/run_notebook.py  # 无头重跑，应输出 errors=0
```

| 位置 | 路径 |
|---|---|
| WSL 项目目录 | `/home/jichi/rag-tech-support-agent` |
| WSL 虚拟环境 | `/home/jichi/rag-tech-support-agent/.venv` |
| Windows 项目目录 | `C:\Users\Administrator\WorkBuddy\2026-09-14-15-41-25\rag-tech-support-agent` |
| Windows 虚拟环境 | `C:\Users\Administrator\.workbuddy\binaries\python\envs\default` |

从零复现 WSL 环境：`bash tools/setup_wsl.sh`（会建 venv、装依赖、注册内核，并在没有
`.env` 时从模板复制一份）。

> 下文命令给的是 **WSL（bash）** 写法。Windows 侧把 `.venv/bin/python` 换成下面这个即可：
> ```powershell
> $PY   = "C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
> $PROJ = "C:\Users\Administrator\WorkBuddy\2026-09-14-15-41-25\rag-tech-support-agent"
> ```
> `.env` 两套环境各放一份（内容可以相同），因为它是按项目目录解析的。

### 3.1 只想读一遍 —— 直接打开

双击 `Technical_Support_AI_Agent_RAG.html` —— 用浏览器看，最省事；
或用 VS Code / JupyterLab 打开 `.ipynb`，能看源码也能改。

### 3.2 直接向它提问（命令行 或 浏览器）

**命令行**：

```bash
cd ~/rag-tech-support-agent
.venv/bin/python tools/ask.py "how do I reset the office router"
.venv/bin/python tools/ask.py -v "what does a 429 error mean"   # 附带 coverage / 相似度 / 引用来源
.venv/bin/python tools/ask.py -k 6 "vpn tunnel establishment timeout"
.venv/bin/python tools/ask.py --json "the vpn says E-2011"      # 机器可读输出
.venv/bin/python tools/ask.py                                   # 不带问题 → 进入交互式会话
```

配置有问题时**不抛栈**，而是打印一段可执行的诊断并以退出码 `3` 结束：

```
======================================================================
CONFIGURATION ERROR - OPENAI_API_KEY is not set
======================================================================
The agent has no offline mode: it cannot run without credentials.

Expected file : /home/jichi/rag-tech-support-agent/.env
File found    : NO - .env does not exist
Variable      : OPENAI_API_KEY

To fix:
  1. cp .env.example .env        (creates the file from the template)
  2. open .env and set OPENAI_API_KEY
  3. re-run this cell
======================================================================
```

**浏览器** —— 想点着试、看引用和置信度，用本地聊天页：

```bash
cd ~/rag-tech-support-agent
.venv/bin/python tools/webui.py --port 8777
# 然后打开 http://127.0.0.1:8777
```

页面上会显示当前**端点地址**、每次回答的 **coverage（接地性覆盖率）/ top 相似度 / 耗时**，
并附一个必然被拒答的样例问题，方便你验证"不知道就说不知道"这条行为。
它**完全不依赖 Jupyter 内核**，只把 `ask.py` 的管线套了一层网页外壳。

> `ask.py` 不复制任何算法 —— 它把 Notebook 里的代码单元格提取出来就地执行，
> 所以命令行行为与 Notebook **永远一致**，不会两边跑偏。

### 3.3 交互式运行 Notebook（JupyterLab）

```bash
cd ~/rag-tech-support-agent
.venv/bin/python -m jupyter lab --no-browser --port 8888 --IdentityProvider.token=atlas
# 浏览器打开 http://127.0.0.1:8888/lab?token=atlas
```

打开 `Technical_Support_AI_Agent_RAG.ipynb`，菜单 **Run → Run All Cells** 即可。
WSL 用的是 mirrored 网络模式，**WSL 里起的服务在 Windows 浏览器里可直接访问**，
不需要任何端口转发。

**一个已知的显示怪癖（不影响使用）**：内核刚启动的那一瞬间，Jupyter 报出的
`execution_state` 可能是 `starting`，界面上的内核指示器显示 "Starting"。
原因是内核在 iopub 上发布启动 `status: idle` 的时机与 jupyter_server 订阅建立之间存在竞态
（ZMQ PUB/SUB 的 slow-joiner 语义），这条状态消息会被丢掉。
**执行一次单元格后状态就恢复为 `idle`，功能完全正常。**

另外两个容易踩的点：

- **命令是 `jupyter lab`，不是 `jupyter notebook`** —— 环境里装的是 `jupyterlab`，
  `notebook` 包没装，`jupyter notebook` 会报命令不存在。想用经典界面就
  `.venv/bin/python -m pip install notebook`。
- **内核不用手动选**：`.venv` 里已注册 `ragenv313` 内核，`python3` 也指向该 venv，
  `import faiss` / `import langchain` 都正常。

### 3.4 无头重跑（验证 / CI）

```bash
cd ~/rag-tech-support-agent
.venv/bin/python tools/run_notebook.py                      # 执行并写入 _executed.ipynb
RAG_PUBLISH=1 .venv/bin/python tools/run_notebook.py        # 把结果发布回主 Notebook
```

脚本会**先检查有没有凭据**。没有就直接退出（码 `3`）并告诉你怎么办，
而不是先花时间启动一个内核再失败。成功时输出：`OK - 52 cells (34 code), errors=0`。

### 3.5 换成你自己的知识库

⚠️ **直接编辑 `tech_support.txt` 是无效的**：Notebook 第 4.1 节内嵌了一份 `KB_TEXT` 常量，
启动时会比对，只要磁盘上的文件与它不一致就**重新写回内嵌版本**，你的修改会被静默覆盖。

正确做法是改 Notebook 里的 `KB_TEXT`（第 4.1 节，`KB_TEXT = r""" ... """` 之间），
再重跑。文档需保持 `## 标题` 形式的二级标题 —— 章节解析与引用溯源都依赖它。
`tools/build_notebook.py` 是这些单元格的生成源，改完它再执行即可重新产出 Notebook。

⚠️ **换语料还有一个更容易踩的坑**：评测集（§11.1 `EVAL_SET`，12 条）与训练对
（§12.2 `TRAIN_PARAPHRASES`，20 条）的标签都是**写死的 section 名字符串**，命中判定靠
`metadata["section"] == expected` 精确匹配。只换 `KB_TEXT` 而不改这两处，**不会报任何错**，
但 Hit@k / MRR 会直接塌成 0（标签全部对不上）——属于静默失效，比报错更危险。
所以换语料必须**三处同步改**：`KB_TEXT` + `EVAL_SET` + `TRAIN_PARAPHRASES`，
改完再 `rm -rf artifacts/vector_store` 重建索引。

想调参就看这几处：

| 参数 | 位置 | 默认值 |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 第 5.1 节 | 500 / 50 字符 |
| `TOP_K` | 第 8.1 节 | 4 |
| `COVERAGE_THRESHOLD` | 第 9.2 节 | 0.50（低于此值拒答） |

## 4. Notebook 章节结构

| § | 章节 | 内容 |
|---|---|---|
| 1 | Project Title & Introduction | 背景、LLM 在私有知识上的三个结构性缺陷 |
| 2 | Theoretical Background | Self-Attention、RAG 三阶段、向量嵌入与余弦相似度、RAG vs 微调、**Mermaid 架构图** |
| 3 | Configuration & Setup | `.env` 加载、依赖安装、**端点预检**、组件报告 |
| 4 | Data Preparation | 知识库写入、章节解析、语料探索性可视化 |
| 5 | Text Splitting | 500 字符 / 50 重叠的递归切分，切分质量验证 |
| 6 | Vector Embeddings | 嵌入模型（默认多语言 `bge-m3`），L2 归一化 + 内积=余弦的自检 |
| 7 | Vector Store | FAISS `IndexFlatIP` 精确内积检索 + 磁盘持久化 |
| 8 | Semantic Retrieval | Top-k 语义检索、相似度可视化 |
| 9 | Agent Logic & Chain | 受限提示词、LCEL 链、**接地性门控（abstention gate）**、`RetrievalQA` 弃用说明 |
| 10 | Usage & Testing | 域内问题（含引用）+ 域外问题（拒答并转人工） |
| 11 | Evaluation | Hit@k / MRR / 排名分布 + 嵌入空间二维投影 + **§11.5 跨语言检索评测（中文/俄文提问、英文知识库）** |
| 12 | Embedder Fine-tuning | 对比学习微调嵌入模型（CPU 全流程）、重建索引、微调前后指标对比 |
| 13 | Conclusion | 结果汇总、已知局限、后续改进方向 |

## 5. 关键设计决策

### 5.1 为什么彻底删掉了离线模式

早先的版本有一条"无 Key 也能跑"的降级路径：TF-IDF + TruncatedSVD 做嵌入，
确定性抽取式生成器做生成。它在演示时很方便，但代价是**交付物可能与生产系统不是同一个东西** ——
你在本地看到的漂亮指标，来自一个生产上根本不会用到的检索器。

现在只有一条路径。配置不全、端点不通、模型名写错，都会在第 3 节**立刻停下**，
并给出针对性的诊断（见 §2 的"两个坑"）。这比"看起来能跑"更有价值。

### 5.2 为什么 `OPENAI_BASE_URL` 是一等公民

支持第三方端点不是加个参数那么简单，有三处必须一起处理，否则换端点必然出问题：

| 问题 | 处理 |
|---|---|
| `tiktoken` 不认识第三方模型名，按上下文长度预切分时会报错 | `OpenAIEmbeddings(check_embedding_ctx_length=False)`，跳过 tiktoken，直接把原文发出去 |
| 向量维度随模型变化，硬编码 1536 会崩 | 维数由预检探针返回，FAISS 索引按矩阵形状创建 —— 换 768 维 / 1024 维模型不用改代码 |
| 中转站往往只代理 chat，`/embeddings` 返回 404 | 预检**分别探测** embeddings 和 chat，并把 404 解释成"该中转不提供嵌入路由" |

chat 侧的 `ChatOpenAI` 同样透传 `base_url` 与 `api_key`，两条链路走同一个端点。

### 5.3 为什么用 LCEL 而不是 `RetrievalQA`？

原方案里的 `RetrievalQA.from_chain_type(...)` 是**已弃用的遗留 API**：
在 LangChain 0.2 被 LCEL 取代，1.x 中已移除。本 Notebook 在第 9.4 节实际探测其可用性
并如实报告（实测结论：`RetrievalQA is NOT importable`），主体使用现代 LCEL：

```python
chain = (
    {"context":  RunnableLambda(lambda q: format_context(retriever.invoke(q))),
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

### 5.4 接地性门控（防幻觉的核心）

拒答判定由**两个信号取 OR**决定，只有**两者都说"没有证据"**时才拒答：

| 信号 | 判据 | 阈值 | 特点 |
|---|---|---|---|
| 词法 | IDF 加权词汇覆盖率 | `coverage < 0.50` | 便宜、确定性、**不依赖嵌入模型**；但跨语言时必然失效 |
| 语义 | Top-1 余弦相似度 | `top_sim < 0.40` | 语言无关 —— 多语言嵌入把三种语言映到同一空间 |

- 语料中**从未出现**的词被赋予**最大 IDF**，因此问一个文档完全没提过的话题时覆盖率趋近 0；
- 罕见的高信息词（如错误码 `429`）权重高，而 *support* 这类无处不在的词几乎不贡献。
- **为什么必须是 OR**：中文提问和英文文档**没有任何字面重合**，词汇覆盖率恒为 0。
  只看词法的话，中文/俄文提问会被 100% 误拒。加上语义信号后，跨语言问题靠"意思接近"
  放行（实测 0.58–0.67），而真正的域外问题两种信号**同时**失败（相似度仅 0.28）。
- 顺带修掉一个真实缺陷：*"What does error E-2011 mean and how do I fix it?"* 的词汇覆盖率
  只有 0.41（`mean` / `fix` 在语料里从不作为独立词出现），原本被误拒；现在语义分 0.59 足以放行。

两者同时失败时，智能体**在调用大模型之前**就确定性拒答 —— 既省钱又可靠，而且这一行为在
`ask()` 里是显式短路，不依赖模型是否听话。返回值里的 `gated=True` 表示"这次是本地拦下的，没花钱"。

### 5.5 分段切分保留章节元数据

切分在**每个章节内部**进行，每个 chunk 继承章节标题作为 metadata，
这使得"引用来源"成为可能（如 `[Source: Router Reset and Network Troubleshooting]`）。

## 6. 测试与评估

Notebook 内建评测：

- **10 个域内问题**（7 个交互式 + 批处理）：验证准确性与引用正确性
- **3 个域外问题**（"法国首都是哪里？"等）：验证礼貌拒答与转人工
- **12 条带标注的评测集**（§11），覆盖 10 个不同章节：计算 Hit@k、MRR、排名分布
- **10 条口语化改写问法**（§12.4，held-out）：专门用来检验微调带来的泛化改善
- **10 条跨语言问题**（§11.5：5 条中文 + 5 条俄文）：验证中文/俄文提问能否检索到英文文档
- **同一条域外问题用三种语言各问一遍**：验证拒答文案是否跟随提问语言

### 当前验证状态

| 项目 | 状态 |
|---|---|
| Notebook 结构（58 单元格 / 39 代码） | ✅ 已校验 |
| 全部代码单元格语法（剥离魔法行后 `compile()`） | ✅ 0 错误 |
| 缺 Key 时的失败路径与诊断文案 | ✅ 已实测，退出码 3 |
| 假 Key / 不可达端点的预检诊断 | ✅ 已实测，401 与连接失败均能正确分类 |
| **真实端点下的端到端指标（Hit@k / MRR / 拒答率）** | ✅ 已实测 |
| **嵌入微调全流程（§12，CPU 可跑）** | ✅ 已实跑，结果可复现（`FT_SEED = 42`） |
| **跨语言检索（中文/俄文提问、英文知识库）** | ✅ §11.5 实测 **Hit@4 = 1.000 / MRR = 1.000**（10/10 排名第 1） |
| **三语言门控与拒答文案** | ✅ 已实测：中/俄/英域外问题全部正确拒答，且文案随语言切换 |

实测配置：`EMBED_BACKEND=local`（`BAAI/bge-m3`，1024 维，本地编码，**多语言**）+ 远端生成。

| 指标 | 基线 | 微调后（4 epochs） |
|---|---|---|
| §11 英文评测集 Hit@4 / MRR | 1.000 / 1.000 | 1.000 / 1.000 |
| **§11.5 跨语言（中/俄）Hit@4 / MRR** | **1.000 / 1.000** | 不适用 |
| held-out 改写问法 MRR | 0.717 | **0.783** |
| held-out 平均 margin | 0.0259 | **0.0412** |

> §12 的微调**故意不微调检索骨干**：训练对象由 `FT_BASE_MODEL` 单独指定（默认
> `bge-small-en-v1.5`）。bge-m3 有 568M 参数，全参数微调的 Adam 状态是模型体积的数倍，
> CPU 内存吃不消；33M 的小模型几秒就能训完，而**方法本身与骨干无关**。

> §11 的 12 题**基线即满分**，说明该集合已经饱和，只能当作"不退化"的守门指标；
> 真正的改善体现在 held-out 改写问法上。另外这 12 题在实测中暴露了词法门控的
> 误拒问题，详见 §7 第 1 条。

## 7. 已知局限

1. **词法覆盖率在跨语言时完全失效，只能靠语义信号兜住**。中文提问与英文文档零字面
   重合，`coverage` 恒为 0，放行与否**完全取决于 `top_sim ≥ 0.40`**。这意味着跨语言
   路径没有第二道独立防线：一旦嵌入模型退化，误放行的风险会直接暴露。
   *（单语言场景下的误拒问题已由 OR 门控修复 —— E-2011 问句现在以语义分 0.59 放行。）*
2. **检索是单轮的**，没有对话式查询改写，追问（"那 Mac 客户端呢？"）会丢失指代。
3. **多跳问题**只依赖 Top-k，无法跨章节组合推理；需要多轮检索的 Agent 循环。
4. **现在只能在线跑**。每次执行都需要可达的凭据与网络，没有任何气隙（air-gapped）路径 ——
   这是主动删掉离线模式换来的代价。
5. **持久化的索引与嵌入模型绑定**。改变 `OPENAI_EMBED_MODEL` 之后必须重建索引，
   否则查询向量与库中向量不在同一空间，检索会静默失效。
6. 语料规模很小（25 个块），Hit@k 高只能说明**流程正确**，不能外推到大语料。
   真实知识库（数万块）需要 ANN 索引（`IndexIVFFlat` / HNSW）而非 `IndexFlatIP`。
7. **§12 的微调属于领域适应（domain adaptation），不是泛化能力的证明**。训练用的
   20 组改写问法，目标正是它随后要检索的那 25 个块。真实语料应按**文档**切分留出
   评测集，并随语料增长持续重训。
8. **多语言只覆盖了提问侧，知识库本身仍是英文**。跨语言检索靠多语言嵌入，回答由模型
   用提问语言复述英文原文。若需要知识库本身中文化 / 俄文化，必须换语料并重做评测集
   （`EVAL_SET` 与 `TRAIN_PARAPHRASES` 的标签都是英文 section 名，见 §3.5）。

## 8. 后续改进方向

- 在生成前加入 **cross-encoder 重排序**
- 把检索器包装成 tool，让 LLM 自行决定**何时检索、检索什么**
- 加入**对话记忆**与查询改写，支持多轮支持会话
- 记录 `(问题, 检索上下文, 回答, 覆盖率)` 四元组 —— 真实用户措辞配上检索到的章节，
  正是 §12 下一轮微调所需的弱标签
- **按 `(模型名, 分块哈希)` 缓存嵌入向量**，避免开发期反复重跑时重复计费

---

*内容由 AI 生成仅供参考*
# rag-tech-support-agent
