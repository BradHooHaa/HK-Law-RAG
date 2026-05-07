# 香港法例 XML RAG 起步版

这个目录里是一套最小可运行的 RAG 前置流程：

1. 从你下载的香港法例 zip 里直接读取 XML。
2. 解析每条法例的章节号、状态、日期、标题、正文。
3. 按 section / regulation / rule 等法律结构切成 chunk。
4. 写入 SQLite FTS5 全文检索库。
5. 用命令行先验证“能不能找准资料”。

这一步还没有接大模型回答问题。原因是 RAG 最容易翻车的地方通常不是 LLM，而是前面的解析、切块、引用和检索。如果检索结果靠谱，下一步接 OpenAI / local LLM 会简单很多。

## 1. 建索引

先跑小样本，确认流程正常：

```bash
python3 scripts/build_index.py \
  --zip "/Users/brad/Downloads/香港法律/香港法例1-300章(英文).zip" \
  --limit 20
```

确认没问题后，建立完整索引：

```bash
python3 scripts/build_index.py \
  --zip "/Users/brad/Downloads/香港法律/香港法例1-300章(英文).zip"
```

默认会生成：

```text
data/hk_legislation_fts.db
```

## 2. 搜索

```bash
python3 scripts/search.py "water authority licensed plumber"
```

也可以限制返回数量：

```bash
python3 scripts/search.py "dangerous goods licence" --limit 10
```

## 3. 问答

先安装 OpenAI Python 包：

```bash
python3 -m pip install -r requirements.txt
```

然后设置 API Key。如果用 GLM，在 `.env` 里放：

```bash
GLM_API_KEY="your_glm_api_key_here"
```

如果用 OpenAI，则设置：

```bash
export OPENAI_API_KEY="your_api_key_here"
```

先只看系统会找出哪些法例片段，不调用模型：

```bash
python3 scripts/ask.py "What licence is required for dangerous goods?" --context-only
```

确认检索片段靠谱后，生成答案：

```bash
python3 scripts/ask.py "What licence is required for dangerous goods?" --show-context
```

默认会优先使用 GLM，模型是 `glm-4.7-flash`。如果你想手动指定：

```bash
python3 scripts/ask.py "What licence is required for dangerous goods?" --provider glm --model glm-4.7-flash
```

如果你想换成 OpenAI：

```bash
python3 scripts/ask.py "What licence is required for dangerous goods?" --provider openai --model gpt-4.1-mini
```

注意：这个工具只用于法律资料检索，不是法律意见。

## 4. 下一步怎么继续优化

推荐路线：

1. 继续保留现在的 SQLite FTS，作为关键词检索。
2. 加 embedding 检索，用向量数据库或 SQLite 向量扩展保存 chunk embedding。
3. 查询时同时跑关键词检索和向量检索，合并 top results。
4. 把结果连同 `citation`、`date`、`source_url` 一起交给 LLM。
5. 回答时强制引用来源，并提示“这不是法律意见”。

一个很实用的 LLM prompt 形状：

```text
You are helping search Hong Kong legislation.
Answer only from the provided context.
If the context is insufficient, say what is missing.
Always cite Cap/section/date/source_url.
Do not provide legal advice.

Question:
{question}

Context:
{retrieved_chunks}
```

## 5. 对新手友好的理解

可以把 RAG 想成三层：

- 数据整理：把 XML 变成一小段一小段能引用的文本。
- 检索：用户问问题时，先找出最相关的法例片段。
- 生成：把这些片段交给大模型，让它基于片段回答。

现在这个项目已经完成了前两层的本地雏形。
