"""全局配置：从 .env 读取环境变量。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（app/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
# 多模态模型（支持图片输入），用于解析用户上传的电路图/作业照片
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-flash")

# 知识库
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
CACHE_PATH = BASE_DIR / "data" / "kb_cache.pkl"
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# 图片上传
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
MAX_UPLOAD_MB = 10
