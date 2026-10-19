"""数字电子技术学习助手 —— FastAPI 主程序。

启动: uvicorn app.main:app --reload
"""
import base64
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, knowledge
from .llm_client import chat_completion
from .tools import circuit

logger = logging.getLogger("uvicorn.error")

# 全局知识库（启动时构建索引）
kb = knowledge.KnowledgeBase()

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    n = kb.build()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("知识库加载完成，共 %d 个分块（来源: %s）", n, config.KNOWLEDGE_DIR)
    yield


app = FastAPI(title="数字电子技术学习助手", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "app" / "static"), name="static")
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=config.UPLOAD_DIR), name="uploads")


# ---------- 数据模型 ----------
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # 对话历史 [{"role": "user"|"assistant", "content": "..."}]
    images: list[str] = []    # 用户本次上传图片的访问 URL 列表


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict]          # 命中的参考资料
    analysis: dict | None = None  # 电路精确分析结果（供前端展示卡片）


# ---------- 接口 ----------
@app.get("/")
async def index():
    return FileResponse(config.BASE_DIR / "app" / "static" / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "kb_chunks": len(kb.chunks)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """接收图片并保存到 data/uploads/，返回可访问的 URL。"""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400,
                            detail=f"仅支持图片格式: {', '.join(sorted(ALLOWED_EXT))}")
    content = await file.read()
    if len(content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"图片超过 {config.MAX_UPLOAD_MB} MB 限制")
    name = uuid.uuid4().hex + ext
    (config.UPLOAD_DIR / name).write_bytes(content)
    return {"url": f"/uploads/{name}"}


def _build_system_prompt(question: str, context: str, analysis: dict | None) -> str:
    parts = [
        "你是「数字电子技术基础」课程的 AI 助教，面向本科学生答疑。\n"
        "要求：\n"
        "1. 优先依据下面提供的参考资料作答，引用时说明出处；\n"
        "2. 解题要分步骤、写清楚推理过程；\n"
        "3. 资料不足以回答时，明确说明，再基于专业知识谨慎补充；\n"
        "4. 语气耐心，适合教学场景；\n"
        "5. 公式一律用纯文本表达，严禁使用 LaTeX 或 Markdown 数学语法："
        "不要出现 $...$、\\overline、\\sum、\\frac、\\geq 等任何转义符号；\n"
        "   非运算写成撇号或“非”字（如 A'、A′、非A），与运算写成 ·（如 A·B 或 AB），"
        "或运算写成 +（如 A+B），整体取反写成括号加撇号（如 (AB)'）；\n"
        "6. 真值表用普通文本表格（表头列名用 | 分隔，表身逐行列出），不要用图片或复杂符号。",
        f"【参考资料】\n{context}",
    ]
    if analysis:
        parts.append(_analysis_prompt(analysis))
    return "\n\n".join(parts)


def _build_image_prompt(context: str, analysis: dict | None) -> str:
    """图片识别场景的专用提示词：先识别、再分析，提高读图准确率。"""
    parts = [
        "你是「数字电子技术基础」课程的 AI 助教，正在查看学生上传的图片"
        "（可能是电路截图、手绘逻辑图、作业照片或教材页面）。\n"
        "回答分两大部分：先给【识别结果】，再给【分析与解答】。\n\n"
        "识别要求：\n"
        "1. 先判断图片类型：电路截图 / 手绘逻辑图 / 作业或题目照片 / 教材页，并如实报告。\n"
        "2. 若是电路图：依次识别输入输出变量、每个逻辑门的类型"
        "（与/或/非/与非/或非/异或/同或）及引脚连接，按信号流向逐步写出各级输出表达式，"
        "最后汇总成整体逻辑表达式。任何看不清的门或连线都要明确标注“看不清”，"
        "绝不猜测编造，并建议用户用文字补充说明电路结构。\n"
        "3. 若是文字图片：逐字抄录题目内容，严格区分题目正文与“提示/注释/答案”等附加文字，"
        "不要把提示或答案当作题目；抄录时保留原符号（如 A'、AB、·、⊕）。\n"
        "4. 若图片中既有题目又有参考答案：先独立解题，不要照抄图片里的答案，"
        "若发现图片中答案有错要明确指出。\n"
        "5. 手写内容不确定时，给出你认为最可能的内容并注明“（不确定）”。\n\n"
        "解题要求：\n"
        "6. 公式一律用纯文本表达，严禁 LaTeX：A' 表示非，A·B 或 AB 表示与，A+B 表示或，"
        "(AB)' 表示整体取反；真值表用文本表格（| 分隔列名）。\n"
        "7. 解题分步骤讲解，优先依据下面参考资料中的知识点。",
    ]
    if context:
        parts.append(f"【参考资料】\n{context}")
    if analysis:
        parts.append(_analysis_prompt(analysis))
    return "\n\n".join(parts)


def _analysis_prompt(analysis: dict) -> str:
    """把 sympy 精确计算结果转成给大模型的提示（要求以工具结果为准）。"""
    lines = ["【电路分析工具输出（sympy 精确计算，讲解时必须以此为准，不要自行改动）】"]
    for e in analysis["expressions"]:
        lines.append(f"- 表达式 {e['expr_str']}：最简与或式 = {e['simplified']}")
        lines.append(f"  真值表：{'；'.join(f'{k}={v}' for k, v in e['truth_table'][0]['inputs'].items())}"
                     f" -> F={e['truth_table'][0]['f']}"
                     + (f"，共 {e['n_rows']} 行（详见前端卡片）" if e["n_rows"] > 1 else ""))
        lines.append(f"  最小项（F=1 的下标）: m{', m'.join(map(str, e['minterms']))}")
    eq = analysis.get("equivalence")
    if eq:
        if eq["equivalent"]:
            lines.append(f"- 等价性判断：{eq['left']} 与 {eq['right']} 等价 ✓")
        else:
            lines.append(f"- 等价性判断：{eq['left']} 与 {eq['right']} 不等价 ✗，"
                         f"反例：{eq['counterexample']} 时两式不同，请指出学生错在哪里并给出正确结果")
    return "\n".join(lines)


def _clean_reply(text: str) -> str:
    """兜底清理回复中残留的 LaTeX 记号，避免浏览器显示成乱码。"""
    s = text
    s = re.sub(r"\\overline\{([^}]*)\}", r"\1'", s)   # \overline{AB} -> AB'
    s = re.sub(r"\\sum\s*", "", s)                    # \sum m(1,3) -> m(1,3)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\cdot\s*", "·", s)
    s = re.sub(r"\\times\s*", "×", s)
    s = re.sub(r"\\(geq|leq|equiv|ne|in)\s*", "", s)  # 删掉 \geq \leq 等符号命令
    s = s.replace("$$", "").replace("$", "").replace("\\", "")
    return s


def _extract_image_expressions(urls: list[str]) -> tuple[list[str], list[str]]:
    """阶段A：让视觉模型只提取图片中的逻辑表达式（不解题），供 sympy 精确校验。

    返回 (题目表达式列表, 作答/参考答案表达式列表)。任何失败都降级为空，不影响主流程。
    """
    system = (
        "你是数字电路题目的文字提取器，只看图提取内容，不解题、不解释、不化简。\n"
        "只提取形如 F=... / Y=... / L=... 的布尔逻辑表达式；"
        "题目描述、提示、注释、说明等其余文字一律忽略。\n"
        "按下面格式逐行输出，没有对应内容就不输出该行，不要输出其他任何文字：\n"
        "【题目表达式】F=...\n"
        "【作答表达式】F=...（仅当图片中出现学生作答或参考答案时才输出）\n"
        "严格保留原符号（'、·、+、⊕、(AB)' 等），不要化简、不要改写。"
    )
    content: list[dict] = [{"type": "text", "text": "请提取图片中的逻辑表达式。"}]
    for url in urls[:4]:
        fp = config.UPLOAD_DIR / Path(url).name
        if not fp.exists():
            continue
        b64 = base64.b64encode(fp.read_bytes()).decode()
        mime = MIME.get(fp.suffix.lower(), "image/png")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}",
                                      "detail": "original"}})
    try:
        reply = chat_completion(
            [{"role": "system", "content": system},
             {"role": "user", "content": content}],
            model=config.DEEPSEEK_VISION_MODEL, temperature=0.0)
    except Exception as e:
        logger.warning("图片表达式提取失败，跳过精确校验: %s", e)
        return [], []
    prompts = re.findall(r"【题目表达式】\s*([^\n【]+)", reply)
    answers = re.findall(r"【作答表达式】\s*([^\n【]+)", reply)
    return ([p.strip() for p in prompts], [a.strip() for a in answers])


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip() and not req.images:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 1) 检索相关资料（RAG）
    hits = kb.retrieve(req.message)
    context = "\n\n".join(
        f"【{c['title']} · {c['section']}】\n{c['text']}" for c in hits
    )

    # 2) sympy 精确分析用户消息中的电路表达式
    raw_analysis = circuit.analyze_message(req.message)
    analysis = circuit.to_public(raw_analysis) if raw_analysis else None

    # 3) 组装消息：图片场景用专门的识别提示词
    model = config.DEEPSEEK_MODEL
    if req.images:
        # 阶段A：从图片提取表达式并用 sympy 精确校验（失败则降级，不影响回答）
        img_prompt, img_answer = _extract_image_expressions(req.images)
        img_analysis = circuit.build_image_analysis(img_prompt, img_answer)
        if img_analysis:
            analysis = img_analysis
        system_prompt = _build_image_prompt(context, analysis)
        text = req.message.strip() or (
            "请先识别这张图片的内容（是电路图还是题目文字），"
            "再分析其中的电路：写出逻辑表达式、判断是否有错误并说明理由。")
        if len(req.images) > 1:
            text += f"\n（共 {len(req.images)} 张图片，按上传顺序编号为图1~图{len(req.images)}）"
        content: list[dict] = [{"type": "text", "text": text}]
        for url in req.images[:4]:  # 单轮最多带 4 张图
            fname = Path(url).name
            fp = config.UPLOAD_DIR / fname
            if not fp.exists():
                continue
            b64 = base64.b64encode(fp.read_bytes()).decode()
            mime = MIME.get(fp.suffix.lower(), "image/png")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}",
                                          "detail": "original"}})  # 保留原图细节，利于读小字/手写
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(req.history[-6:])
        messages.append({"role": "user", "content": content})
        model = config.DEEPSEEK_VISION_MODEL
    else:
        system_prompt = _build_system_prompt(req.message, context, analysis)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(req.history[-6:])  # 只保留最近 6 轮
        messages.append({"role": "user", "content": req.message})

    # 4) 调用 DeepSeek
    try:
        reply = _clean_reply(chat_completion(messages, model=model))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("调用 DeepSeek 失败")
        raise HTTPException(status_code=502, detail=f"调用大模型失败: {e}")

    return ChatResponse(reply=reply, sources=hits, analysis=analysis)
