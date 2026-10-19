# 数字电子技术学习助手（AI 教学智能体 · 毕设项目骨架）

面向《数字电子技术基础》课程的 AI 教学智能体。当前为**阶段 1~2** 骨架：
**DeepSeek 对话问答 + RAG 教材知识库**。后续阶段将扩展电路分析工具与作业批改引擎。

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. 配置 DeepSeek API Key
copy .env.example .env         # 然后编辑 .env 填入 DEEPSEEK_API_KEY
# 申请地址: https://platform.deepseek.com/

# 3. 启动服务
uvicorn app.main:app --reload
```

浏览器打开 http://127.0.0.1:8000 即可对话。

## 项目结构

```
├── app/
│   ├── main.py            # FastAPI 主程序（/api/chat、/api/health）
│   ├── config.py          # 配置（读取 .env）
│   ├── llm_client.py      # DeepSeek 客户端（OpenAI 兼容）
│   ├── knowledge.py       # RAG：文档分块 + jieba 分词 + BM25 检索
│   └── static/index.html  # 前端聊天页面
├── data/knowledge/        # 教材知识文档（.md/.txt，启动时自动建索引）
└── requirements.txt
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/chat | `{message, history}` → `{reply, sources}` |
| GET  | /api/health | 服务与知识库状态 |

## 使用说明

- 把教材/讲义/PPT 导出的文本放入 `data/knowledge/`（UTF-8 或 GBK 编码），重启服务即自动重建检索索引
- 回答会附带命中的"参考资料"出处，方便核对与演示 RAG 效果
- 未配置 API Key 时服务可正常启动，调用聊天接口会返回明确提示

## 下一步规划（毕设阶段 3~5）

- **阶段 3 · 电路分析引擎**：接入 `sympy`（布尔化简、真值表）、`PySpice`（仿真）、`matplotlib`（画波形），让智能体通过"工具调用"完成电路计算
- **阶段 4 · 作业批改引擎**：作业上传 → 内容解析（含 PaddleOCR 手写识别）→ 对照标准答案/评分规则逐题判分 → 生成批改报告
- **阶段 5 · 整合验收**：统一界面（学生问答区 + 教师批改区），编写演示用例与论文
