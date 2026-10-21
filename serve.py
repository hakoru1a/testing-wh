#!/usr/bin/env python3
"""
Dùng: python serve.py  →  mở http://localhost:3456
Bỏ cặp file cùng tên (vd: bai1.mp3 + bai1.srt) vào thư mục input/.
"""
import os, re, json, mimetypes, urllib.parse, threading, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BASE  = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, 'input')
PORT  = 3456
DATA  = os.path.join(BASE, 'data')
DB_PATH = os.path.join(DATA, 'database.json')
DB_LOCK = threading.Lock()


def db_read():
    try:
        with open(DB_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def db_write(data):
    os.makedirs(DATA, exist_ok=True)
    tmp = DB_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DB_PATH)
AUDIO_EXTS = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.webm'}
NET_ERRORS = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)


def list_pairs():
    if not os.path.isdir(INPUT):
        return []
    pairs = []
    for f in os.listdir(INPUT):
        name, ext = os.path.splitext(f)
        if ext.lower() in AUDIO_EXTS and os.path.exists(os.path.join(INPUT, name + '.srt')):
            pairs.append({
                'name': name,
                'audio': '/input/' + urllib.parse.quote(f),
                'srt': '/input/' + urllib.parse.quote(name + '.srt'),
                'mtime': os.path.getmtime(os.path.join(INPUT, f)),
            })
    pairs.sort(key=lambda p: p['name'])
    return pairs


# ── Upload + transcribe (job chạy nền, client poll /api/job) ──
JOBS = {}
MODELS = {}
MODEL_LOCK = threading.Lock()   # 1 job transcribe tại một thời điểm (CPU)


def safe_name(name):
    name = os.path.basename(name or '')
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip() or 'audio.mp3'


def run_job(job_id, fpath, model_name, lang):
    job = JOBS[job_id]
    try:
        from faster_whisper import WhisperModel
        from transcribe import fmt
        with MODEL_LOCK:
            job['status'] = 'loading'
            if model_name not in MODELS:
                MODELS[model_name] = WhisperModel(model_name, device='cpu', compute_type='int8')
            job['status'] = 'running'
            segs, info = MODELS[model_name].transcribe(
                fpath, language=(lang or None), beam_size=5)
            dur = info.duration or 1
            base = os.path.splitext(os.path.basename(fpath))[0]
            srt_path = os.path.join(INPUT, base + '.srt')
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(segs, 1):
                    f.write(f"{i}\n{fmt(seg.start)} --> {fmt(seg.end)}\n{seg.text.strip()}\n\n")
                    job['progress'] = min(99, round(seg.end / dur * 100, 1))
                    job['segments'] = i
        job.update(status='done', progress=100,
                   srt='/input/' + urllib.parse.quote(base + '.srt'))
    except Exception as e:
        job.update(status='error', error=str(e))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_data(self, code, ctype, body):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', len(body))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except NET_ERRORS:
            pass

    def send_file(self, fpath, mime):
        size = os.path.getsize(fpath)
        start, end = 0, size - 1
        rng = self.headers.get('Range')
        m = re.match(r'bytes=(\d*)-(\d*)', rng or '')
        # Trình duyệt cần Range (206) để tua audio đến đúng câu
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            else:
                start = max(0, size - int(m.group(2)))
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        else:
            self.send_response(200)
        length = end - start + 1
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', length)
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        try:
            with open(fpath, 'rb') as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except NET_ERRORS:
            pass

    def do_GET(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)

        if path in ('/', '/index.html'):
            self.send_file(os.path.join(BASE, 'dictation.html'), 'text/html; charset=utf-8')
            return

        if path == '/upload':
            self.send_file(os.path.join(BASE, 'upload.html'), 'text/html; charset=utf-8')
            return

        if path == '/api/job':
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            job = JOBS.get((q.get('id') or [''])[0])
            if not job:
                self.send_data(404, 'application/json', '{"error":"job not found"}')
            else:
                self.send_data(200, 'application/json', json.dumps(job))
            return

        if path == '/api/db':
            with DB_LOCK:
                self.send_data(200, 'application/json', json.dumps(db_read(), ensure_ascii=False))
            return

        if path == '/api/list':
            self.send_data(200, 'application/json', json.dumps(list_pairs()))
            return

        if path.startswith('/input/'):
            fpath = os.path.normpath(os.path.join(INPUT, path[len('/input/'):]))
            if fpath.startswith(INPUT + os.sep) and os.path.isfile(fpath):
                mime = mimetypes.guess_type(fpath)[0] or 'application/octet-stream'
                if fpath.endswith('.srt'):
                    mime = 'text/plain; charset=utf-8'
                self.send_file(fpath, mime)
                return

        self.send_data(404, 'text/plain', 'Not found')


    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == '/api/db':
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            except json.JSONDecodeError:
                self.send_data(400, 'application/json', '{"error":"invalid json"}')
                return
            with DB_LOCK:
                data = db_read()
                if 'clear' in body:
                    for k in [k for k in data if k.startswith(str(body['clear']))]:
                        del data[k]
                elif 'key' in body:
                    if body.get('value') is None:
                        data.pop(body['key'], None)
                    else:
                        data[body['key']] = body['value']
                db_write(data)
            self.send_data(200, 'application/json', '{"ok":true}')
            return
        if u.path != '/api/upload':
            self.send_data(404, 'text/plain', 'Not found')
            return
        q = urllib.parse.parse_qs(u.query)
        name = safe_name((q.get('name') or [''])[0])
        if os.path.splitext(name)[1].lower() not in AUDIO_EXTS:
            self.send_data(400, 'application/json', '{"error":"unsupported file type"}')
            return
        os.makedirs(INPUT, exist_ok=True)
        fpath = os.path.join(INPUT, name)
        remaining = int(self.headers.get('Content-Length') or 0)
        with open(fpath, 'wb') as f:
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {'id': job_id, 'name': name, 'status': 'queued', 'progress': 0, 'segments': 0}
        threading.Thread(target=run_job, daemon=True, args=(
            job_id, fpath, (q.get('model') or ['medium'])[0], (q.get('lang') or ['ja'])[0])).start()
        self.send_data(200, 'application/json', json.dumps(JOBS[job_id]))


if __name__ == '__main__':
    print(f'\n✅  Server chạy tại: http://localhost:{PORT}')
    print(f'    Bỏ cặp file .mp3 + .srt cùng tên vào: {INPUT}\n')
    ThreadingHTTPServer(('', PORT), Handler).serve_forever()
