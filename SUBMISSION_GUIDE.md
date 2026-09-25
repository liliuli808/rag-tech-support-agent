# 作业交付指南 — Technical Support AI Agent (RAG)

> 配套作业要求（俄语原文）：
> *Разработать Юпитер ноутбук демонстрирующий работу чата технической поддержки
> с использованием ИИ с применением любого сервиса. Привести теорию и работающий
> код с обучением и использованием на английском языке.*

---

## 0. 结论先行

**作业本身已经做完了。** 剩下的全是「交付动作」，预计 30 分钟内可收尾：

1. 换成你自己的 API Key；
2. 跑一次一键复现，确认 `errors=0`；
3. 选要交的文件（并确认**不要**把 `.env` 交出去）；
4. 过一遍答辩要点。

---

## 1. 作业要求 → 交付物对照表

| 俄语要求 | 含义 | 落在哪里 | 状态 |
|---|---|---|---|
| Юпитер ноутбук | Jupyter Notebook | `Technical_Support_AI_Agent_RAG.ipynb`（57 单元格 / 38 代码块） | ✅ |
| демонстрирующий работу чата технической поддержки с ИИ | 演示 AI 技术支持聊天 | §9 Agent Logic、§10 Usage and Testing；另有浏览器聊天界面 `tools/webui.py` | ✅ |
| с применением любого сервиса | 使用任意（第三方）服务 | §3：OpenAI 兼容端点做生成 + 本地多语言模型 `BAAI/bge-m3` 做嵌入 | ✅ |
| привести теорию | 包含理论 | §2 Theoretical Background（含 §2.6 端到端架构图） | ✅ |
| работающий код | 可运行的代码 | 38 个代码块全部已执行，`errors=0` | ✅ |
| **с обучением** | **含训练** | **§12 Fine-tuning the Embedder**（CPU 上真实微调，非骨架） | ✅ |
| на английском языке | 英文 | Notebook 全文英文（理论、注释、输出） | ✅ |

**"обучение"（训练）这一条**是之前唯一的硬伤——原 §12 只是个 `RUN_FINE_TUNING = False` 的 LoRA 骨架，标注 *not executed*。现已整块替换为真实执行的嵌入微调，这是本次交付的关键补强。

---

## 2. 你要做的四件事

### ① 换成自己的 API Key（必做）

当前 `.env` 里的 Key 是从 `~/qqbot/.env.prod` 借来的，**不能这样交作业**。

```bash
cd ~/rag-tech-support-agent
nano .env        # 改这三行：
#   OPENAI_API_KEY=<你自己的 Key>
#   OPENAI_BASE_URL=<服务商的 /v1 地址>
#   OPENAI_CHAT_MODEL=<模型名>
```

改完先验证端点，再跑 Notebook：

```bash
.venv/bin/python tools/check_endpoint.py
```

该脚本会分别探测 `/embeddings` 与 `/chat/completions`，直接告诉你这个中转到底支持哪条路由。

> 备注：`EMBED_BACKEND=local` 时只需要 chat 路由可用，嵌入在本地跑，所以**对中转的要求很低**——只要它能对话即可。这也是当前配置能成立的唯一原因。

### ② 一键复现验证（必做）

```bash
bash tools/setup_wsl.sh              # 建 venv + 装依赖 + 注册内核
.venv/bin/python tools/run_notebook.py   # 无头执行全部单元格
```

期望输出末行为 `errors=0`。这一步同时证明「работающий код」名副其实。

### ③ 决定交什么

| 文件 | 说明 | 建议 |
|---|---|---|
| `Technical_Support_AI_Agent_RAG.ipynb` | 主交付物，含全部输出 | **必交** |
| `Technical_Support_AI_Agent_RAG.html` | 同一份的 HTML 导出，无需 Jupyter 即可阅读 | 建议一并交，方便老师批阅 |
| `tech_support.txt` | 知识库语料（虚构的 Northwind 公司 IT 手册，英文） | 建议交，否则 Notebook 无法复现 |
| `tools/` 中的 `build_notebook.py` / `run_notebook.py` | Notebook 的生成与执行脚本 | 可选，能证明工程可复现 |
| `.env` | **含密钥** | ❌ **绝对不要交** |
| `README.md` | 使用说明（中文） | 可选 |

**关于语料：不需要你提供私有知识。** Notebook 的知识库是随代码内嵌的**合成语料**
（`*.example` 保留域名、虚构公司 Northwind），不是外部下载的，也不涉及真实客户数据。
作业只要求「演示」技术支持聊天，并未要求真实业务文档——用虚构语料反而更干净：
无隐私、无版权风险，可以放心交给老师或放到公开仓库。答辩时直接这样说明即可，是加分项。

> 如果你**确实想**换成自己的文档（例如老师明确要求真实场景），代价比看上去大：
> 必须同步改**三处**——`KB_TEXT`（§4.1）+ `EVAL_SET`（§11.1）+ `TRAIN_PARAPHRASES`（§12.2），
> 因为评测与训练的标签都是**写死的 section 名**。只换语料不改标签**不会报任何错**，
> 但 Hit@k / MRR 会静默塌成 0。详见 README §3.5。

### ④ 准备答辩

见第 4 节。重点是把「为什么这样设计」讲清楚，而不是背代码。

---

## 3. 三个风险点（提前处理）

1. **没有离线模式。** 生成端必须联网。若演示／批改环境**不能上外网**，Notebook 到 §9 之后的生成环节会失败。
   → 预案：提前跑好并交**已执行的 `.ipynb` / `.html`**（输出已固化为单元格结果），现场只做讲解、不重跑。

2. **本地嵌入模型首次需要联网下载，而且不小。** 默认的 `BAAI/bge-m3` 约 **2.3 GB**，
   首次加载还要 2–3 分钟（之后走 HF 缓存会快很多）。国内网络建议加
   `HF_ENDPOINT=https://hf-mirror.com`。
   → 预案：**提前跑通一次，把缓存预热好**，别到演示现场才下载。
   → 如果只需要英文、想省时间和磁盘：把 `LOCAL_EMBED_MODEL` 换成 `BAAI/bge-small-en-v1.5`
     （约 130 MB，秒级加载，但**中文/俄文提问会失效**）。换模型后必须删除
     `artifacts/vector_store` 重建索引。

3. **`README.md` 是中文的**，而作业要求英文。
   → Notebook 本身是纯英文，满足要求；若老师要求**全部材料**英文，用第 5 节附的英文说明替代 README 作为交付文档。

---

## 4. 答辩要点（大概率会被问到）

| 问题 | 应答要点 |
|---|---|
| 为什么用 RAG，不直接问大模型？ | 直接问会出现幻觉，且不懂企业私有知识。RAG 把「事实」交给检索到的文档，模型只负责组织语言。 |
| 检索是怎么实现的？ | 切分（500 字符 / 50 重叠）→ 嵌入（384 维，归一化）→ FAISS `IndexFlatIP` 精确内积（=余弦）→ top-4。 |
| 你**训练**了什么？为什么训练它？ | 微调**嵌入模型**（不是生成模型）。理由：生成模型的微调只改变「怎么表达」，而检索微调直接改善「能不能找到」；RAG 的瓶颈在检索，且只需 `(问句, 正确段落)` 配对即可训练——这正是 `MultipleNegativesRankingLoss` 的输入形式。 |
| 训练配置？ | 20 组改写问法（与评测集零重叠）、4 epochs、batch 8、lr 2e-5、CPU 分钟级完成；损失函数用 in-batch 负样本。 |
| 效果如何？ | 评测集（§11, 12 题）基线已满分 1.000，无提升空间；另建 held-out 改写问法集才有区分度——**MRR 0.717 → 0.783**，margin 0.0259 → 0.0412。固定种子可复现。 |
| 怎么防止幻觉？ | **接地性门控**：回答前先用 IDF 加权字面覆盖率检查「检索到的材料是否真的支持这个问题」，低于阈值直接拒答，不调用模型（22ms、0 token）。 |
| 支持多语言吗？ | 支持中/俄/英三种。检索用多语言嵌入 `bge-m3`，把三种语言映到同一向量空间，所以中文提问能命中英文文档（§11.5 实测 Hit@4 = 1.000）。回答语言由 prompt 规则跟随提问语言，连拒答文案都有三套。 |
| 有什么已知缺陷？ | 跨语言路径**只有语义一道防线**：中文提问与英文文档零字面重合，词汇覆盖率恒为 0，放行与否完全取决于 `top_sim ≥ 0.40`。单语言下的误拒问题已由 OR 门控修复（E-2011 问句现在以语义分 0.59 放行）。详见 §13 已知局限第 1、8 条。 |

**主动讲出第 7 条（缺陷）通常比隐藏它更好** —— 它展示的是你理解了系统的失效模式，而不是背诵了一个 demo。

---

## 5. 可附给老师的英文说明

> **Deliverable**: `Technical_Support_AI_Agent_RAG.ipynb` — a Jupyter Notebook that
> implements an AI technical-support chat agent using a Retrieval-Augmented
> Generation (RAG) pipeline.
>
> **External service used**: any OpenAI-compatible chat endpoint (configured via
> `OPENAI_BASE_URL` / `OPENAI_CHAT_MODEL` in `.env`), plus a local multilingual
> `sentence-transformers` model (`BAAI/bge-m3`) for embeddings. The agent accepts
> questions in **English, Chinese or Russian** and answers in the language it was
> asked in; the knowledge base itself stays in English, which the multilingual
> embedder bridges. No proprietary or offline
> fallback stack is used — the pipeline is exactly the RAG architecture
> described in Section 2.
>
> **Theory** is in Section 2 (chunking, embeddings and vector space, cosine
> similarity, FAISS indexing, groundedness gating, LCEL composition), with the
> end-to-end architecture diagram in Section 2.6.
>
> **Running code**: all 38 code cells execute end-to-end with zero errors
> (reproduce with `python tools/run_notebook.py`).
>
> **Training** (Section 12, `с обучением`): an embedding model is fine-tuned
> in-process on CPU using `MultipleNegativesRankingLoss` with in-batch negatives,
> on 20 paraphrase pairs disjoint from the evaluation set. The fine-tuning
> backbone is configured independently through `FT_BASE_MODEL` (default
> `BAAI/bge-small-en-v1.5`): the 568M-parameter multilingual retrieval model is
> too large for full fine-tuning in CPU RAM, and the method itself is
> backbone-agnostic. Held-out paraphrase
> MRR improves from 0.717 to 0.783 and the retrieval margin from 0.0259 to
> 0.0412. Training is seeded for reproducibility. Note that the Section 11
> benchmark is already saturated at 1.000 at baseline, so the held-out set is
> where the improvement is actually measurable — this is stated explicitly in
> the notebook.
>
> **Knowledge base**: a synthetic IT support handbook for a fictional company
> (all domains use the reserved `.example` TLD). It contains no real data.
>
> **Interface**: `tools/webui.py` serves a browser chat UI at
> `http://127.0.0.1:8777`; `tools/ask.py` offers the same pipeline as a CLI.

---

## 6. 从零复现的完整命令

```bash
# 1) 准备环境（建 venv、装依赖、注册 ragenv313 内核）
bash tools/setup_wsl.sh

# 2) 配置凭据（必做）
cp .env.example .env && nano .env

# 3) 校验端点是否同时支持 chat / embeddings
.venv/bin/python tools/check_endpoint.py

# 4) 无头执行整个 Notebook —— 期望 errors=0
.venv/bin/python tools/run_notebook.py

# 5) 直接提问（命令行）
.venv/bin/python tools/ask.py -v "How do I reset the office router?"

# 6) 浏览器聊天界面
.venv/bin/python tools/webui.py --port 8777      # http://127.0.0.1:8777

# 7) 交互式打开 Notebook
.venv/bin/python -m jupyter lab --no-browser --port 8888 \
    --IdentityProvider.token=atlas                 # http://127.0.0.1:8888/lab?token=atlas

# 8) 导出 HTML（交作业用）
.venv/bin/jupyter nbconvert --to html _executed.ipynb \
    --output Technical_Support_AI_Agent_RAG.html
```

> 换嵌入模型后**必须先删掉 `artifacts/vector_store/` 再重建索引**，
> 否则查询向量与库中向量不在同一空间，检索会静默失效（不报错，只是结果全错）。
