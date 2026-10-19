"""知识库 RAG：读取教材文档 -> 分块 -> 中文分词 -> BM25 检索。

- 文档来源: data/knowledge/ 下的 .md / .txt 文件（UTF-8 或 GBK）
- 检索策略: BM25（离线、零依赖大模型，毕设演示够用；
  后续可无缝替换为向量检索，见 README「下一步」）
"""
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from . import config


class KnowledgeBase:
    def __init__(self):
        self.chunks: list[dict] = []      # [{"title", "section", "text"}]
        self._bm25: BM25Okapi | None = None

    # ---------- 构建索引 ----------
    def build(self, force: bool = False) -> int:
        """从 KNOWLEDGE_DIR 读取文档并构建 BM25 索引，返回分块数量。"""
        if not force and self.chunks and self._bm25 is not None:
            return len(self.chunks)

        raw_docs = self._load_files()
        self.chunks = self._chunk_docs(raw_docs)
        corpus = [[w for w in jieba.cut(c["text"]) if w.strip()] for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)
        return len(self.chunks)

    # ---------- 检索 ----------
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """检索与 query 最相关的 top_k 个知识分块。"""
        top_k = top_k or config.TOP_K
        if self._bm25 is None:
            self.build()

        tokens = [w for w in jieba.cut(query) if w.strip()]
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k] if scores[i] > 0]

    # ---------- 内部实现 ----------
    def _load_files(self) -> list[tuple[str, str]]:
        if not config.KNOWLEDGE_DIR.exists():
            return []
        docs = []
        for fp in sorted(config.KNOWLEDGE_DIR.iterdir()):
            if fp.suffix.lower() not in (".md", ".txt"):
                continue
            text = self._read_text(fp)
            if text.strip():
                docs.append((fp.stem, text))
        return docs

    @staticmethod
    def _read_text(fp: Path) -> str:
        for encoding in ("utf-8", "gbk"):
            try:
                return fp.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return ""

    def _chunk_docs(self, docs: list[tuple[str, str]]) -> list[dict]:
        """按「段落 + 章节标题」分块，约 400 字一块，保留出处信息。"""
        chunks: list[dict] = []
        for title, text in docs:
            heading = ""
            buffer: list[str] = []
            for line in text.splitlines():
                line = line.strip()
                # 章节标题行
                if re.match(r"^#{1,4}\s", line):
                    self._flush(chunks, title, heading, buffer)
                    heading = re.sub(r"^#{1,4}\s*", "", line).strip()
                    continue
                # 空行 => 段落分隔
                if not line:
                    self._flush(chunks, title, heading, buffer)
                    continue
                buffer.append(line)
                if len("".join(buffer)) >= 400:
                    self._flush(chunks, title, heading, buffer)
            self._flush(chunks, title, heading, buffer)
        return chunks

    @staticmethod
    def _flush(chunks: list[dict], title: str, heading: str, buffer: list[str]) -> None:
        text = "\n".join(buffer).strip()
        if len(text) >= 20:  # 丢弃过短的碎片
            chunks.append({"title": title, "section": heading or "未分类", "text": text})
        buffer.clear()
