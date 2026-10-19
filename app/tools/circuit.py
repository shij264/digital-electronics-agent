"""数字电路分析工具：布尔表达式解析、化简、真值表、等价性判断与纠错。

使用 sympy 做精确计算（不是大模型“猜”），返回结构化结果，
供大模型讲解（以工具结果为准）和前端表格展示。

支持的输入写法示例：
    AB + A'C        与或式，并列字母表示“与”，' 表示“非”
    A·B ∨ ¬C        也可用 · ∧ ∨ ¬ ! ⊕ 等符号
    F=AB+(A^B)'     可用 F= 前缀，^ 表示异或
"""
from __future__ import annotations

import re
from itertools import product

from sympy import And, Not, Or, Symbol, Xor, simplify_logic
from sympy.logic.boolalg import BooleanFalse, BooleanTrue, truth_table

# 识别“等价 / 对不对”等纠错意图的关键词
_EQUIV_KEYWORDS = ("等价", "对不对", "一样", "相同", "一样吗", "比较", "验证", "检查",
                   "判断", "化简对", "正确", "对吗", "等不等于", "是否一样")

_VAR = re.compile(r"[A-Za-z]")


# ---------- 表达式解析（文本 -> sympy 表达式） ----------

class _Parser:
    """递归下降解析布尔表达式。

    文法（优先级从低到高）:
        expr    := xor_expr (('|'|'+'|'∨') xor_expr)*
        xor_expr:= term (('^'|'⊕') term)*
        term    := factor (factor | '&'|'·'|'∧'|'*')*      # 并列即“与”
        factor  := primary ("'")*                          # 后缀 ' 表示非
        primary := '(' expr ')' | '~' primary | 变量 | 0 | 1
    """

    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def parse(self):
        e = self.parse_expr()
        if self.i < len(self.s):
            raise ValueError(f"无法解析的字符: {self.s[self.i:]!r}")
        return e

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def parse_expr(self):
        left = self.parse_xor()
        while self.peek() in ("|", "+", "∨"):
            self.i += 1
            left = Or(left, self.parse_xor())
        return left

    def parse_xor(self):
        left = self.parse_term()
        while self.peek() in ("^", "⊕"):
            self.i += 1
            left = Xor(left, self.parse_term())
        return left

    def parse_term(self):
        left = self.parse_factor()
        while True:
            c = self.peek()
            if c in ("", "|", "+", "∨", "^", "⊕", ")"):
                break
            if c in ("&", "·", "∧", "*"):
                self.i += 1
            left = And(left, self.parse_factor())  # 相邻项默认是“与”
        return left

    def parse_factor(self):
        f = self.parse_primary()
        while self.peek() == "'":
            self.i += 1
            f = Not(f)
        return f

    def parse_primary(self):
        c = self.peek()
        if c == "(":
            self.i += 1
            e = self.parse_expr()
            if self.peek() != ")":
                raise ValueError("缺少右括号")
            self.i += 1
            return e
        if c == "~":
            self.i += 1
            return Not(self.parse_primary())
        if c == "0":
            self.i += 1
            return BooleanFalse()
        if c == "1":
            self.i += 1
            return BooleanTrue()
        if _VAR.match(c):
            self.i += 1
            return Symbol(c.upper())
        raise ValueError(f"无法解析的字符: {c!r}")


def _preprocess(text: str) -> str:
    """去掉 F= / F(A,B)= 前缀、统一符号、补省略的“与号”。"""
    s = text.strip()
    s = re.sub(r"^[A-Za-z]\s*\([^)]*\)\s*=\s*", "", s)  # F(A,B)=...
    s = re.sub(r"^[A-Za-z]\s*=\s*", "", s)              # F=...
    s = (s.replace("∧", "&").replace("·", "&").replace("*", "&")
          .replace("∨", "|").replace("＋", "|")
          .replace("¬", "~").replace("！", "~").replace("!", "~")
          .replace("⊕", "^").replace("⊻", "^")
          .replace("（", "(").replace("）", ")"))
    s = re.sub(r"\s+", "", s)
    # 补上省略的“与号”：变量相邻、变量后跟 '('、')' 后跟变量或 '('
    s = re.sub(r"([A-Za-z])([A-Za-z])", r"\1&\2", s)
    s = re.sub(r"([A-Za-z])(\()", r"\1&\2", s)   # 如 "A(B+C)" -> "A&(B+C)"
    s = re.sub(r"(\))([A-Za-z])", r"\1&\2", s)   # 如 "(A+B)C" -> "(A+B)&C"
    s = re.sub(r"(\))(\()", r"\1&\2", s)         # 如 "(A)(B)"  -> "(A)&(B)"
    return s


# ---------- 对外工具函数 ----------

def parse(expr_str: str):
    """解析文本为 sympy 表达式，失败抛 ValueError。"""
    return _Parser(_preprocess(expr_str)).parse()


def analyze(expr_str: str) -> dict:
    """分析一个布尔表达式，返回结构化结果。

    返回 {ok, error} 或 {ok, expr_str, simplified, variables,
         truth_table, minterms, n_rows, expr(内部)}
    """
    try:
        expr = parse(expr_str)
    except (ValueError, TypeError) as e:
        return {"ok": False, "error": str(e), "expr_str": expr_str}

    symbols = sorted(expr.free_symbols, key=str)          # 按名字排序，保证顺序稳定
    if not symbols:
        return {"ok": False, "error": "表达式中没有变量", "expr_str": expr_str}

    simplified = simplify_logic(expr, form="dnf", force=True)

    rows = []
    minterms = []
    n = len(symbols)
    for bits, val in truth_table(expr, symbols, input=True):
        # 教材惯例：第 1 个变量（如 A）是最高位
        idx = sum(b << (n - 1 - i) for i, b in enumerate(bits))
        if val:
            minterms.append(idx)
        rows.append({"inputs": {str(s): b for s, b in zip(symbols, bits)}, "f": int(bool(val))})

    return {
        "ok": True,
        "expr_str": expr_str,
        "simplified": str(simplified),
        "variables": [str(s) for s in symbols],
        "truth_table": rows,
        "minterms": minterms,
        "n_rows": len(rows),
        "_expr": expr,
        "_symbols": symbols,
    }


def compare(expr_a: str | dict, expr_b: str | dict) -> dict:
    """判断两个表达式是否等价；不等价时给出反例（哪组输入下不同）。"""
    a = expr_a if isinstance(expr_a, dict) else analyze(expr_a)
    b = expr_b if isinstance(expr_b, dict) else analyze(expr_b)
    for d in (a, b):
        if not d["ok"]:
            return {"ok": False, "error": f"无法解析: {d.get('expr_str')}"}

    diff = simplify_logic(Xor(a["_expr"], b["_expr"]))
    equivalent = diff == BooleanFalse()
    result = {
        "ok": True,
        "left": a["simplified"],
        "right": b["simplified"],
        "equivalent": bool(equivalent),
        "counterexample": None,
    }
    if not equivalent:
        symbols = sorted(set(a["_symbols"]) | set(b["_symbols"]), key=str)
        for bits, val in truth_table(diff, symbols, input=True):
            if val:
                result["counterexample"] = {str(s): int(bool(b)) for s, b in zip(symbols, bits)}
                break
    return result


def extract_expressions(text: str) -> list[str]:
    """从一句问话里提取疑似布尔表达式。

    支持三类写法：
    - F=... / F(A,B)=...
    - “化简成 X / 得到 X / 结果是 X” 这类第二表达式
    - “X 和 Y 等价吗 / 一样吗” 这类并列比较
    """
    cands: list[str] = []
    for m in re.finditer(r"[A-Za-z]\s*\([^)]*\)\s*=\s*([^，。；！？\n]+)", text):
        cands.append(m.group(1).strip())
    for m in re.finditer(r"[A-Za-z]\s*=\s*([^，。；！？\n]+)", text):
        cands.append(m.group(1).strip())
    # 等价类问法：提取“化简成/得到/结果是 X”形式的第二表达式
    for m in re.finditer(r"(化简成|化简为|化简到|得到|结果是|结果为|算成|化成|我的答案|我算得|写成)\s*([^，。；！？\n]+)",
                         text):
        cands.append(m.group(2).strip())
    # 并列比较：“X 和 Y 等价吗 / 一样吗 / 对不对”
    for m in re.finditer(r"([A-Za-z][^，。；！？\n]{0,40}?)\s*(?:和|与|跟)\s*([A-Za-z][^，。；！？\n]{0,40}?)\s*(?:等价|一样|相同|等不等于)",
                         text):
        cands.extend([m.group(1).strip(), m.group(2).strip()])
    # 没有“X=”形式且整句较短时，尝试整句作为表达式
    if not cands and len(text) < 60:
        cands.append(text)
    out = []
    for c in cands:
        c = c.strip("。！？,.； ")
        if c and _VAR.search(c):
            out.append(c)
    return out


def build_image_analysis(prompt_exprs: list[str], answer_exprs: list[str]) -> dict | None:
    """根据从图片中提取的题目式与作答式，构造可校验的分析结果。

    与 analyze_message 输出结构一致（expressions + equivalence），
    可直接用于前端分析卡片和提示词注入。
    """
    expressions = [a for e in prompt_exprs if (a := analyze(e))["ok"]]
    equivalence = None
    if prompt_exprs and answer_exprs:
        a, b = analyze(prompt_exprs[0]), analyze(answer_exprs[0])
        if a["ok"] and b["ok"]:
            equivalence = compare(a, b)
    if not expressions and equivalence is None:
        return None
    return to_public({"expressions": expressions, "equivalence": equivalence})


def to_public(result: dict) -> dict:
    """去掉内部 sympy 对象（_expr/_symbols），转成可 JSON 序列化的纯数据。"""
    out: dict = {}
    for k, v in result.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            out[k] = to_public(v)
        elif isinstance(v, list):
            out[k] = [to_public(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def analyze_message(text: str) -> dict | None:
    """从用户消息中提取并分析电路表达式（含等价性纠错判断）。

    返回 None 表示没有可分析的电路内容；
    否则返回 {expressions: [...], equivalence: {...} | None}。
    """
    cands = extract_expressions(text)
    valid = [analyze(c) for c in cands]
    valid = [a for a in valid if a["ok"]]
    if not valid:
        return None

    result: dict = {"expressions": valid}
    equivalence = None
    if len(valid) >= 2 and any(k in text for k in _EQUIV_KEYWORDS):
        equivalence = compare(valid[0], valid[1])
    elif len(valid) == 1 and any(k in text for k in _EQUIV_KEYWORDS):
        # 只有一个表达式，仅报告化简结果（交给 analyze 的 simplified 字段）
        pass
    result["equivalence"] = equivalence
    return result
