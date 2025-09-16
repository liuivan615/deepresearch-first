# control_server.py  —— 实时会话 + 事件流 + 导出
import os, asyncio, json, uuid, re
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse, FileResponse

from client import MCPClient  # 复用你的 MCP 客户端

LOG_PATH = os.getenv("LOG_PATH", "test.log")
EXPORT_DIR = os.getenv("EXPORT_DIR", "outputs")
os.makedirs(EXPORT_DIR, exist_ok=True)

app = FastAPI(title="MCP Controller", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)

_client: Optional[MCPClient] = None
_client_lock = asyncio.Lock()

class LiveSession:
    def __init__(self, sid:str):
        self.id = sid
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None
        self.active: bool = False
        self.final_md: str = ""
        self.graph_nodes: List[Dict[str,Any]] = []
        self.graph_edges: List[Dict[str,Any]] = []

_sessions: Dict[str, LiveSession] = {}

# --------- 工具函数 ---------
async def ensure_client_started():
    global _client
    async with _client_lock:
        if _client is None:
            _client = MCPClient()
            await _client.connect_to_server("./search_mcp.py")

def sse_format(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# --------- 控制器原有 ---------
@app.post("/api/start")
async def start_controller():
    await ensure_client_started()
    return {"status": "started"}

@app.post("/api/stop")
async def stop_controller():
    global _client
    async with _client_lock:
        if _client:
            await _client.close()
        _client = None
    return {"status":"stopped"}

@app.get("/api/status")
async def status():
    running = _client is not None
    provider = _client.llm and ("ollama" if _client.llm.ollama else "api")
    model = _client.llm.model_name if (_client and _client.llm) else None
    ollama_models = _client.llm.ollama.models if (_client and _client.llm and _client.llm.ollama) else []
    return {"running": bool(running), "provider": provider, "model": model, "ollama_models": ollama_models}

@app.get("/api/logs/tail", response_class=PlainTextResponse)
async def logs_tail(lines: int = 400):
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
        return "".join(data[-abs(int(lines)):])
    except FileNotFoundError:
        return ""
    except Exception as e:
        return PlainTextResponse(f"[error reading log] {e}", status_code=500)

# --------- 新：会话 / 事件流 / 交互 ----------
@app.post("/api/session")
async def create_session():
    sid = uuid.uuid4().hex
    _sessions[sid] = LiveSession(sid)
    return {"session_id": sid}

@app.post("/api/session/{sid}/clarify")
async def clarify(sid: str, payload: dict = Body(...)):
    await ensure_client_started()
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error":"empty_question"}, status_code=400)
    prompt = f"请针对下面的用户问题提出 3-5 条澄清问题，编号列出，尽量覆盖范围、目标、限制、偏好等关键信息。\n\n用户问题：{question}"
    resp = await _client.llm.create_chat_completion([
        {"role":"system","content":"你是严谨的研究助理。只输出问题列表，不要多余说明。"},
        {"role":"user","content": prompt}
    ])
    text = resp.choices[0].message.content
    questions = re.split(r'^\s*\d+[\.\)]\s*', text, flags=re.M)[1:] or [text]
    return {"clarifying": [q.strip() for q in questions if q.strip()]}

@app.post("/api/session/{sid}/start")
async def start_research(sid:str, payload: dict = Body(...)):
    await ensure_client_started()
    sess = _sessions.get(sid)
    if not sess: return JSONResponse({"error":"session_not_found"}, status_code=404)

    query = (payload.get("query") or "").strip()
    if not query: return JSONResponse({"error":"empty_query"}, status_code=400)

    async def sink(ev: dict):
        await sess.queue.put(sse_format(ev))
        t = ev.get("type")
        if t == "tool_call":
            nid = f"call_{len(sess.graph_nodes)}"
            sess.graph_nodes.append({"id": nid, "label": f"🔧 {ev.get('name')}"})
            if sess.graph_nodes:
                prev = sess.graph_nodes[-2]["id"] if len(sess.graph_nodes) >= 2 else "root"
                sess.graph_edges.append({"source": prev, "target": nid})
        elif t == "tool_result":
            nid = f"res_{len(sess.graph_nodes)}"
            sess.graph_nodes.append({"id": nid, "label": f"📄 {ev.get('name')} 结果"})
            prev = sess.graph_nodes[-2]["id"] if len(sess.graph_nodes) >= 2 else "root"
            sess.graph_edges.append({"source": prev, "target": nid})
        elif t == "final_report" and ev.get("format") == "markdown":
            sess.final_md = ev.get("content","")

    async def runner():
        sess.active = True
        await sess.queue.put(sse_format({"type":"phase","phase":"research","progress":0.15}))
        try:
            await _client.process_query_stream(query, event_cb=sink)
        except Exception as e:
            await sess.queue.put(sse_format({"type":"error","message":str(e)}))
        finally:
            sess.active = False
            await sess.queue.put(sse_format({"type":"phase","phase":"done","progress":1.0}))

    if sess.task and not sess.task.done():
        return {"status":"already_running"}
    sess.task = asyncio.create_task(runner())
    return {"status":"started"}

@app.get("/api/session/{sid}/events/stream")
async def events_stream(sid: str):
    sess = _sessions.get(sid)
    if not sess: return JSONResponse({"error":"session_not_found"}, status_code=404)

    async def gen():
        yield sse_format({"type":"graph_init","root":{"id":"root","label":"🧭 研究会话"}})
        while True:
            msg = await sess.queue.get()
            yield msg
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/api/session/{sid}/report")
async def get_report(sid:str):
    sess = _sessions.get(sid)
    if not sess: return JSONResponse({"error":"session_not_found"}, status_code=404)
    return {"markdown": sess.final_md or ""}

# --------- 导出（DOCX / PDF / Markdown） ----------
def export_docx(md: str, filepath: str):
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    lines = md.splitlines()
    in_code = False
    buf: List[str] = []

    def flush_buf():
        nonlocal buf
        if buf:
            doc.add_paragraph("\n".join(buf))
            buf.clear()

    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code:
                doc.add_paragraph("\n".join(buf)).style = 'Intense Quote'
                buf.clear(); in_code=False
            else:
                flush_buf(); in_code=True
            continue
        if in_code:
            buf.append(ln); continue
        if ln.startswith("# "): flush_buf(); doc.add_heading(ln[2:].strip(), level=1); continue
        if ln.startswith("## "): flush_buf(); doc.add_heading(ln[3:].strip(), level=2); continue
        if ln.startswith("### "): flush_buf(); doc.add_heading(ln[4:].strip(), level=3); continue
        if re.match(r"^\s*[-*]\s+", ln):
            doc.add_paragraph(re.sub(r"^\s*[-*]\s+","",ln), style='List Bullet'); continue
        if re.match(r"^\s*\d+\.\s+", ln):
            doc.add_paragraph(re.sub(r"^\s*\d+\.\s+","",ln), style='List Number'); continue
        doc.add_paragraph(ln)
    flush_buf()
    doc.save(filepath)

def export_pdf_simple(md: str, filepath: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_path = os.getenv("PDF_FONT_TTF")
    if font_path and os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont("DocFont", font_path))
        base_font = "DocFont"
    else:
        base_font = "Helvetica"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body", fontName=base_font, fontSize=10.5, leading=14))
    styles.add(ParagraphStyle(name="H1", parent=styles['Heading1'], fontName=base_font))
    styles.add(ParagraphStyle(name="H2", parent=styles['Heading2'], fontName=base_font))
    styles.add(ParagraphStyle(name="H3", parent=styles['Heading3'], fontName=base_font))

    doc = SimpleDocTemplate(filepath, pagesize=A4)
    story: List[Any] = []
    lines = md.splitlines()
    in_code, codebuf = False, []

    def flush_code():
        nonlocal codebuf
        if codebuf:
            story.append(Paragraph("<br/>".join([x.replace("<","&lt;").replace(">","&gt;") for x in codebuf]), styles["Body"]))
            story.append(Spacer(1,6))
            codebuf=[]
    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code: flush_code(); in_code=False
            else: in_code=True
            continue
        if in_code:
            codebuf.append(ln); continue
        if ln.startswith("# "): story.append(Paragraph(ln[2:].strip(), styles["H1"])); story.append(Spacer(1,8)); continue
        if ln.startswith("## "): story.append(Paragraph(ln[3:].strip(), styles["H2"])); story.append(Spacer(1,6)); continue
        if ln.startswith("### "): story.append(Paragraph(ln[4:].strip(), styles["H3"])); story.append(Spacer(1,4)); continue
        if re.match(r"^\s*[-*]\s+", ln):
            story.append(Paragraph("• "+re.sub(r"^\s*[-*]\s+","",ln), styles["Body"])); continue
        story.append(Paragraph(ln.replace("  ", "&nbsp;&nbsp;"), styles["Body"]))
    flush_code()
    doc.build(story)

@app.post("/api/session/{sid}/export")
async def export_report(sid: str, payload: dict = Body(...)):
    sess = _sessions.get(sid)
    if not sess: return JSONResponse({"error":"session_not_found"}, status_code=404)
    md = (sess.final_md or "").strip()
    if not md: return JSONResponse({"error":"no_report"}, status_code=400)

    fmt = (payload.get("format") or "md").lower()
    base = os.path.join(EXPORT_DIR, f"deepresearch_{sid}")
    if fmt == "md":
        path = base + ".md"
        with open(path, "w", encoding="utf-8") as f: f.write(md)
        return FileResponse(path, filename=os.path.basename(path), media_type="text/markdown")
    elif fmt == "docx":
        path = base + ".docx"
        export_docx(md, path)
        return FileResponse(path, filename=os.path.basename(path), media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    elif fmt == "pdf":
        path = base + ".pdf"
        export_pdf_simple(md, path)
        return FileResponse(path, filename=os.path.basename(path), media_type="application/pdf")
    return JSONResponse({"error":"unsupported_format"}, status_code=400)

# 允许直接 python control_server.py 启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8787)
