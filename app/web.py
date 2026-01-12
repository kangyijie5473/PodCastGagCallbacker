import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", static_url_path="/static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
TASKS = {}

def _resolve_upload_path(name):
    if not name:
        return None
    p = name
    if not os.path.isabs(p):
        p = os.path.join(UPLOAD_DIR, p)
    p = os.path.abspath(p)
    try:
        if os.path.commonpath([p, os.path.abspath(UPLOAD_DIR)]) != os.path.abspath(UPLOAD_DIR):
            return None
    except Exception:
        return None
    if os.path.exists(p) and os.path.isfile(p):
        return p
    return None

def _start_task(fn, *args, **kwargs):
    import threading, uuid, time
    tid = uuid.uuid4().hex
    TASKS[tid] = {"status": "pending", "result": None, "error": None, "start": time.time()}
    def run():
        TASKS[tid]["status"] = "running"
        try:
            r = fn(*args, **kwargs)
            TASKS[tid]["result"] = r
            TASKS[tid]["status"] = "done"
        except Exception as e:
            TASKS[tid]["error"] = str(e)
            TASKS[tid]["status"] = "error"
    th = threading.Thread(target=run, daemon=True)
    th.start()
    TASKS[tid]["thread"] = th
    return tid

def process_audio(audio_path, index_dir):
    from .audio_processing import transcribe_audio, assign_speakers, detect_laughter, build_index
    segs = transcribe_audio(audio_path)
    spks = assign_speakers(audio_path, segs)
    laughs = detect_laughter(audio_path, segs)
    return build_index(audio_path, segs, spks, laughs, index_dir)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/media/<path:fn>")
def media(fn):
    return send_from_directory(UPLOAD_DIR, fn)

@app.route("/api/index", methods=["POST"])
def api_index():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    fn = secure_filename(f.filename)
    ap = os.path.join(UPLOAD_DIR, fn)
    f.save(ap)
    id_dir = os.path.join(DATA_DIR, os.path.splitext(fn)[0])
    idx_json, idx_emb = process_audio(ap, id_dir)
    return jsonify({"index": id_dir, "index_json": idx_json, "index_emb": idx_emb, "audio_url": f"/media/{fn}"})

@app.route("/api/test_index")
def api_test_index():
    name = request.args.get("name")
    timeout_s = request.args.get("timeout", type=float)
    ap = None
    if name:
        p = _resolve_upload_path(name)
        if p:
            ap = p
    if ap is None:
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOAD_DIR, f)), reverse=True)
        ap = os.path.join(UPLOAD_DIR, files[0]) if files else None
    if ap is None:
        return jsonify({"error": "no local file"}), 404
    fn = os.path.basename(ap)
    rel = os.path.relpath(ap, UPLOAD_DIR)
    id_dir = os.path.join(DATA_DIR, os.path.splitext(fn)[0])
    if timeout_s and timeout_s > 0:
        tid = _start_task(process_audio, ap, id_dir)
        th = TASKS[tid]["thread"]
        th.join(timeout=timeout_s)
        st = TASKS[tid]["status"]
        if st == "done":
            idx_json, idx_emb = TASKS[tid]["result"]
            return jsonify({"index": id_dir, "index_json": idx_json, "index_emb": idx_emb, "audio_url": f"/media/{rel}", "task": tid, "status": st})
        return jsonify({"task": tid, "status": st}), 202
    idx_json, idx_emb = process_audio(ap, id_dir)
    return jsonify({"index": id_dir, "index_json": idx_json, "index_emb": idx_emb, "audio_url": f"/media/{rel}"})

@app.route("/api/test_index_async")
def api_test_index_async():
    name = request.args.get("name")
    ap = None
    if name:
        p = _resolve_upload_path(name)
        if p:
            ap = p
    if ap is None:
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOAD_DIR, f)), reverse=True)
        ap = os.path.join(UPLOAD_DIR, files[0]) if files else None
    if ap is None:
        return jsonify({"error": "no local file"}), 404
    fn = os.path.basename(ap)
    id_dir = os.path.join(DATA_DIR, os.path.splitext(fn)[0])
    tid = _start_task(process_audio, ap, id_dir)
    return jsonify({"task": tid, "file": fn})

@app.route("/api/task_status")
def api_task_status():
    tid = request.args.get("id")
    if not tid or tid not in TASKS:
        return jsonify({"error": "no such task"}), 404
    t = TASKS[tid]
    r = {"status": t["status"], "error": t["error"]}
    if t["status"] == "done" and t["result"]:
        idx_json, idx_emb = t["result"]
        r.update({"index_json": idx_json, "index_emb": idx_emb})
    return jsonify(r)

@app.route("/api/split")
def api_split():
    name = request.args.get("name")
    sec = request.args.get("sec", type=int) or 300
    overlap = request.args.get("overlap", type=int) or 0
    ap = None
    if name:
        p = _resolve_upload_path(name)
        if p:
            ap = p
    if ap is None:
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOAD_DIR, f)), reverse=True)
        ap = os.path.join(UPLOAD_DIR, files[0]) if files else None
    if ap is None:
        return jsonify({"error": "no local file"}), 404
    fn = os.path.basename(ap)
    base = os.path.splitext(fn)[0]
    out_dir = os.path.join(UPLOAD_DIR, "chunks", base)
    from .split_audio import split_audio
    parts = split_audio(ap, out_dir, sec, overlap)
    items = []
    for it in parts:
        rel = os.path.relpath(it["file"], UPLOAD_DIR)
        items.append({"url": f"/media/{rel}", "start": it["start"], "end": it["end"]})
    return jsonify({"count": len(items), "items": items})

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    id_name = request.args.get("id")
    if not id_name:
        dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        if not dirs:
            return jsonify([])
        id_name = dirs[0]
    from .search_service import SearchService
    svc = SearchService(os.path.join(DATA_DIR, id_name))
    res = svc.search(q)
    return jsonify(res)

def run():
    app.run(host="127.0.0.1", port=8000, debug=True)

if __name__ == "__main__":
    run()
