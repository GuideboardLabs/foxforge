"""App generation pool — Flask API + Vue 3 (CDN, no build step) + SQLite.

Mirrors the Foxforge project pattern: Python Flask backend, single-page
Vue 3 frontend served as static HTML + JS, SQLite database via stdlib sqlite3.

Pipeline (sequential — each step receives all prior outputs as context):
    1. db_architect         — SQLite schema + Flask db-helper module
    2. api_implementer      — Complete Flask app.py with all routes
                              → py_compile check + 2 fix cycles
    3. vue_architect        — Component/store plan given the API shape
    4. vue_implementer      — index.html + app.js (Vue 3 CDN, no build step)
    5. integration_check    — Flags Flask route / Vue fetch() mismatches
    6. integration_fixer    — Applies the flagged fixes to app.py + app.js
                              → py_compile check on fixed Flask code
    7. css_writer           — Generates real styles.css from HTML selectors
    8. readme_writer        — Setup + run instructions

Output: actual .py / .sql / .html / .js / .md files written to
    Projects/{slug}/implementation/{ts}_app/

Extend mode (automatic): if a prior {ts}_app/ build exists for the project slug,
every agent receives the existing files and is instructed to add/update only
what the new request requires — keeping all prior working code intact.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from shared_tools.feedback_learning import FeedbackLearningEngine
from shared_tools.model_routing import lane_model_config
from shared_tools.ollama_client import OllamaClient


# ---------------------------------------------------------------------------
# SQLite canonical patterns — injected into every Python-generating agent
# ---------------------------------------------------------------------------

_SQLITE_PATTERNS = """\
SQLite canonical patterns — follow these exactly:

1. Connection lifecycle using flask.g:
   import sqlite3
   from flask import g

   DATABASE = 'app.db'

   def get_db():
       if 'db' not in g:
           g.db = sqlite3.connect(DATABASE)
           g.db.row_factory = sqlite3.Row   # dict-like row access
           g.db.execute("PRAGMA foreign_keys = ON")
       return g.db

   @app.teardown_appcontext
   def close_db(e=None):
       db = g.pop('db', None)
       if db is not None:
           db.close()

2. Schema initialization:
   def init_db():
       with app.app_context():
           db = get_db()
           with open('schema.sql', 'r') as f:
               db.executescript(f.read())
           db.commit()

3. Always use parameterized queries — never f-strings or % in SQL:
   db.execute("INSERT INTO items (name, value) VALUES (?, ?)", (name, value))
   db.execute("SELECT * FROM items WHERE id = ?", (item_id,))

4. Schema conventions:
   - Every table gets: id INTEGER PRIMARY KEY AUTOINCREMENT
   - Timestamps: created_at TEXT DEFAULT (datetime('now', 'utc'))
   - Booleans: INTEGER (0/1)
   - Foreign keys: REFERENCES parent(id) ON DELETE CASCADE
   - Always add indexes on foreign key columns

5. Row to dict:
   row = db.execute("SELECT * FROM items WHERE id = ?", (id,)).fetchone()
   if row is None:
       return jsonify({"error": "not found"}), 404
   return jsonify(dict(row))

6. Batch fetch:
   rows = db.execute("SELECT * FROM items ORDER BY created_at DESC").fetchall()
   return jsonify([dict(r) for r in rows])
"""

# ---------------------------------------------------------------------------
# Vue 3 CDN canonical patterns — injected into frontend-generating agents
# ---------------------------------------------------------------------------

_VUE3_PATTERNS = """\
Vue 3 CDN pattern — no build step, loaded via unpkg CDN:

1. HTML entry point structure:
   <!DOCTYPE html>
   <html lang="en">
   <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <title>App Title</title>
     <link rel="stylesheet" href="/static/styles.css">
   </head>
   <body>
     <div id="app"><!-- Vue mounts here --></div>
     <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
     <script src="/static/app.js"></script>
   </body>
   </html>

2. app.js structure using Vue 3 global build:
   const { createApp, ref, reactive, computed, onMounted, watch } = Vue;

   const app = createApp({
     setup() {
       // state
       const items = ref([]);
       const loading = ref(false);
       const error = ref('');

       // API client — use fetch(), no axios
       async function fetchItems() {
         loading.value = true;
         try {
           const res = await fetch('/api/items');
           if (!res.ok) throw new Error(await res.text());
           items.value = await res.json();
         } catch(e) {
           error.value = e.message;
         } finally {
           loading.value = false;
         }
       }

       async function createItem(data) {
         const res = await fetch('/api/items', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify(data),
         });
         if (!res.ok) throw new Error(await res.text());
         return res.json();
       }

       onMounted(fetchItems);
       return { items, loading, error, fetchItems, createItem };
     },

     // Template: use inline template or <template> in HTML
   });

   app.mount('#app');

3. Always use v-bind, v-on, v-for with :key, v-if/v-else.
4. For forms: use v-model on ref() values.
5. No build step means no SFC (.vue files) — all in one app.js.
"""

# ---------------------------------------------------------------------------
# Flask API canonical patterns
# ---------------------------------------------------------------------------

_FLASK_PATTERNS = """\
Flask API canonical patterns:

1. App factory with Blueprint:
   from flask import Flask, jsonify, request, g
   app = Flask(__name__)
   app.config['JSON_SORT_KEYS'] = False

2. Standard CRUD route shapes:
   GET    /api/items          → list all
   POST   /api/items          → create (body: JSON)
   GET    /api/items/<int:id> → get one
   PUT    /api/items/<int:id> → update (body: JSON)
   DELETE /api/items/<int:id> → delete

3. Always enable CORS for frontend dev:
   from flask_cors import CORS
   CORS(app)
   (or manually add Access-Control-Allow-Origin header)

4. Error handling:
   @app.errorhandler(404)
   def not_found(e):
       return jsonify({"error": "not found"}), 404

   @app.errorhandler(400)
   def bad_request(e):
       return jsonify({"error": str(e)}), 400

5. Input validation — always validate before writing:
   data = request.get_json(silent=True)
   if not data or 'name' not in data:
       return jsonify({"error": "name required"}), 400

6. Always return JSON. Never return HTML from API routes.
"""


# ---------------------------------------------------------------------------
# Zero-model-call quality checkers
# ---------------------------------------------------------------------------

# Known stdlib top-level package names for Python < 3.10 (no sys.stdlib_module_names)
_STDLIB_EXTRAS: frozenset[str] = frozenset({
    "abc", "argparse", "ast", "asyncio", "base64", "builtins", "cgi",
    "collections", "contextlib", "copy", "csv", "dataclasses", "datetime",
    "decimal", "email", "enum", "functools", "glob", "gzip", "hashlib",
    "hmac", "html", "http", "importlib", "inspect", "io", "itertools",
    "json", "logging", "math", "mimetypes", "multiprocessing", "operator",
    "os", "pathlib", "pickle", "platform", "pprint", "queue", "random",
    "re", "secrets", "shutil", "signal", "socket", "sqlite3", "ssl",
    "stat", "string", "struct", "subprocess", "sys", "tempfile", "textwrap",
    "threading", "time", "traceback", "typing", "unicodedata", "unittest",
    "urllib", "uuid", "warnings", "weakref", "xml", "xmlrpc", "zipfile",
    "zlib", "__future__",
})

_IMPORT_RE = re.compile(r"^(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE)


def _check_dependencies(flask_code: str) -> tuple[list[str], list[str]]:
    """Parse imports in flask_code, return (present_third_party, missing_third_party)."""
    raw_packages = set(_IMPORT_RE.findall(flask_code))
    try:
        stdlib_names: frozenset[str] = sys.stdlib_module_names  # type: ignore[attr-defined]
    except AttributeError:
        stdlib_names = _STDLIB_EXTRAS
    third_party = [p for p in sorted(raw_packages) if p not in stdlib_names]
    present: list[str] = []
    missing: list[str] = []
    for pkg in third_party:
        try:
            if importlib.util.find_spec(pkg) is not None:
                present.append(pkg)
            else:
                missing.append(pkg)
        except (ModuleNotFoundError, ValueError):
            missing.append(pkg)
    return present, missing


class _HTMLChecker(HTMLParser):
    """Tracks structural invariants that would break Vue template compilation."""

    _BLOCK_TAGS: frozenset[str] = frozenset({
        "div", "section", "article", "main", "header", "footer", "nav",
        "form", "table", "thead", "tbody", "tr", "ul", "ol",
    })

    def __init__(self) -> None:
        super().__init__()
        self._stack: list[str] = []
        self.issues: list[str] = []
        self.has_app_div = False
        self.has_vue_cdn = False
        self.has_app_js = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        if tag == "div" and attr_dict.get("id") == "app":
            self.has_app_div = True
        if tag == "script":
            src = attr_dict.get("src") or ""
            if "vue" in src.lower():
                self.has_vue_cdn = True
            if "app.js" in src:
                self.has_app_js = True
        if tag in self._BLOCK_TAGS:
            self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag not in self._BLOCK_TAGS:
            return
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        elif tag in self._stack:
            # Drain stack to find matching open tag
            while self._stack and self._stack[-1] != tag:
                self.issues.append(f"Unclosed <{self._stack.pop()}> before </{tag}>")
            if self._stack:
                self._stack.pop()


def _check_html_structure(index_html: str) -> list[str]:
    """Check index.html for structural issues that would break Vue. Zero model calls."""
    issues: list[str] = []
    checker = _HTMLChecker()
    try:
        checker.feed(index_html)
    except Exception as exc:
        return [f"HTML parse error: {exc}"]

    if not checker.has_app_div:
        issues.append('Missing <div id="app"> — Vue has no mount target.')
    if not checker.has_vue_cdn:
        issues.append("Missing Vue 3 CDN <script> tag.")
    if not checker.has_app_js:
        issues.append("Missing /static/app.js <script> tag.")
    for unclosed in checker._stack:
        issues.append(f"Unclosed block tag: <{unclosed}>")
    issues.extend(checker.issues)
    # v-for without :key causes Vue warnings and potential rendering bugs
    vfor_lines = [
        l.strip() for l in index_html.splitlines()
        if "v-for" in l and ":key" not in l and "key" not in l
    ]
    for line in vfor_lines[:5]:
        issues.append(f"v-for without :key: {line[:120]}")
    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _trim(text: str, max_chars: int) -> str:
    body = str(text or "").strip()
    if len(body) <= max_chars:
        return body
    cut = body[:max_chars].rsplit("\n", 1)[0].strip()
    return cut or body[:max_chars]


_CODE_FENCE_RE = re.compile(r"```(?:python|sql|html|javascript|js|vue)?\n(.*?)```", re.DOTALL)


def _extract_largest_block(text: str) -> str:
    blocks = _CODE_FENCE_RE.findall(str(text or ""))
    if not blocks:
        return str(text or "").strip()
    return max(blocks, key=len).strip()


def _extract_named_block(text: str, extensions: tuple[str, ...]) -> str:
    """Extract first code block matching any of the given language hints."""
    pattern = re.compile(
        r"```(?:" + "|".join(extensions) + r")?\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    blocks = pattern.findall(str(text or ""))
    if not blocks:
        return _extract_largest_block(text)
    return max(blocks, key=len).strip()


def _py_compile_check(code: str) -> tuple[bool, str]:
    """Run py_compile on code string. Returns (ok, error_text)."""
    tmp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout or "syntax error").strip()
    except subprocess.TimeoutExpired:
        return False, "[py_compile timed out]"
    except Exception as exc:
        return False, str(exc)
    finally:
        if tmp:
            try:
                Path(tmp).unlink()
            except Exception:
                pass


def _fix_python(
    client: OllamaClient,
    code: str,
    error: str,
    question: str,
    cancel_checker: Callable[[], bool] | None,
) -> str:
    if callable(cancel_checker):
        try:
            if cancel_checker():
                return code
        except Exception:
            pass
    system_prompt = (
        "You are a Python debugging agent. "
        "Fix the syntax or import error in the code. "
        "Return the complete corrected Python file in a ```python block. "
        "Do not truncate. No explanations outside the code block."
    )
    user_prompt = (
        f"App request: {question}\n\n"
        f"Code with error:\n```python\n{code}\n```\n\n"
        f"Error:\n{error}\n\n"
        "Return the complete corrected Python code."
    )
    try:
        result = client.chat(
            model="qwen2.5-coder:7b",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            num_ctx=20480,
            think=False,
            timeout=300,
            retry_attempts=3,
            retry_backoff_sec=1.5,
        )
        fixed = _extract_named_block(str(result or ""), ("python",))
        return fixed if fixed.strip() else code
    except Exception:
        return code


def _chat(
    client: OllamaClient,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    num_ctx: int = 16384,
    timeout: int = 360,
) -> str:
    try:
        result = client.chat(
            model="qwen2.5-coder:7b",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            num_ctx=num_ctx,
            think=False,
            timeout=timeout,
            retry_attempts=4,
            retry_backoff_sec=1.5,
        )
        return str(result or "").strip()
    except Exception as exc:
        return f"[Model call failed: {exc}]"


# ---------------------------------------------------------------------------
# Extend mode: find an existing build to extend rather than rebuild from scratch
# ---------------------------------------------------------------------------

def _find_existing_app(repo_root: Path, project_slug: str) -> dict[str, str]:
    """Find the most recent _app/ build for this project and read key source files.

    Returns a dict of {relative_path: content}. Empty dict means no prior build found.
    Files read: app.py, schema.sql, db.py, templates/index.html, static/app.js, static/styles.css.
    """
    impl_dir = repo_root / "Projects" / project_slug / "implementation"
    if not impl_dir.exists():
        return {}
    app_dirs = sorted(
        [d for d in impl_dir.iterdir() if d.is_dir() and d.name.endswith("_app")],
        key=lambda d: d.name,
        reverse=True,  # most recent first (ISO timestamp prefix sorts lexicographically)
    )
    if not app_dirs:
        return {}
    latest = app_dirs[0]
    found: dict[str, str] = {}
    for rel in ("app.py", "schema.sql", "db.py", "templates/index.html", "static/app.js", "static/styles.css"):
        path = latest / rel
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    found[rel] = content
            except Exception:
                pass
    if found:
        found["__source_dir__"] = latest.name  # record which build we're extending
    return found


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_db_architect(
    client: OllamaClient,
    question: str,
    project_context: str,
    existing_schema: str = "",
    existing_db_py: str = "",
) -> tuple[str, str]:
    """Returns (schema_sql, db_helpers_py)."""
    if existing_schema:
        system_prompt = (
            f"Today: {_today()}. "
            "You are a SQLite database architect EXTENDING an existing app. "
            "Review the existing schema and add only the tables or columns the new feature requires. "
            "Do NOT remove or rename existing tables or columns — only add. "
            "Follow these patterns exactly:\n\n" + _SQLITE_PATTERNS + "\n\n"
            "Output TWO code blocks:\n"
            "1. A ```sql block with the COMPLETE updated schema.sql (existing + new tables).\n"
            "2. A ```python block with the COMPLETE updated db.py (keep existing helpers, add new ones only if needed).\n"
            "Nothing else."
        )
        user_prompt = (
            f"New feature to add: {question}\n\n"
            f"Existing schema.sql (keep all — only add what the new feature requires):\n"
            f"```sql\n{_trim(existing_schema, 3000)}\n```\n\n"
            f"Existing db.py (keep unchanged unless a new helper is needed):\n"
            f"```python\n{_trim(existing_db_py, 2000)}\n```\n\n"
            f"Project context:\n{_trim(project_context, 1500)}\n\n"
            "Output the complete updated schema.sql and db.py."
        )
    else:
        system_prompt = (
            f"Today: {_today()}. "
            "You are a SQLite database architect. Design a complete schema and Flask db-helper module. "
            "Follow these patterns exactly:\n\n" + _SQLITE_PATTERNS + "\n\n"
            "Output TWO code blocks:\n"
            "1. A ```sql block with schema.sql — CREATE TABLE statements only.\n"
            "2. A ```python block with db.py — get_db(), close_db(), init_db() only.\n"
            "Nothing else."
        )
        user_prompt = (
            f"App to build: {question}\n\n"
            f"Project context:\n{_trim(project_context, 2000)}\n\n"
            "Design a minimal but complete SQLite schema for this app."
        )
    raw = _chat(client, system_prompt, user_prompt, temperature=0.15, num_ctx=16384)

    sql_match = re.search(r"```sql\n(.*?)```", raw, re.DOTALL)
    py_match = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
    schema_sql = sql_match.group(1).strip() if sql_match else _extract_largest_block(raw)
    db_py = py_match.group(1).strip() if py_match else ""

    if not db_py:
        db_py = (
            "import sqlite3\nfrom flask import g\n\nDATABASE = 'app.db'\n\n"
            "def get_db():\n    if 'db' not in g:\n        g.db = sqlite3.connect(DATABASE)\n"
            "        g.db.row_factory = sqlite3.Row\n        g.db.execute('PRAGMA foreign_keys = ON')\n"
            "    return g.db\n\n"
            "@app.teardown_appcontext\ndef close_db(e=None):\n    db = g.pop('db', None)\n"
            "    if db is not None:\n        db.close()\n\n"
            "def init_db():\n    with app.app_context():\n        db = get_db()\n"
            "        with open('schema.sql') as f:\n            db.executescript(f.read())\n"
            "        db.commit()\n"
        )
    return schema_sql, db_py


def _step_api_implementer(
    client: OllamaClient,
    question: str,
    schema_sql: str,
    db_py: str,
    project_context: str,
    cancel_checker: Callable[[], bool] | None,
    progress_callback: Callable | None,
    existing_flask_code: str = "",
) -> str:
    def _prog(stage: str, detail: dict | None = None) -> None:
        if callable(progress_callback):
            try:
                progress_callback(stage, detail or {})
            except Exception:
                pass

    if existing_flask_code:
        system_prompt = (
            f"Today: {_today()}. "
            "You are a Flask API developer EXTENDING an existing app. "
            "Add or modify ONLY the routes needed for the new feature. "
            "Keep ALL existing working routes exactly as-is — do not delete or rename them. "
            "Follow these patterns:\n\n" + _FLASK_PATTERNS + "\n\n" + _SQLITE_PATTERNS + "\n\n"
            "Output ONE complete ```python block — the full updated app.py ready to run."
        )
        user_prompt = (
            f"New feature to add: {question}\n\n"
            f"Existing app.py (keep all existing routes — add/modify ONLY what the new feature needs):\n"
            f"```python\n{_trim(existing_flask_code, 5000)}\n```\n\n"
            f"Updated schema (schema.sql):\n```sql\n{_trim(schema_sql, 2000)}\n```\n\n"
            f"DB helper module (db.py):\n```python\n{_trim(db_py, 1500)}\n```\n\n"
            f"Project context:\n{_trim(project_context, 1000)}\n\n"
            "Output the complete updated app.py with all existing + new routes."
        )
    else:
        system_prompt = (
            f"Today: {_today()}. "
            "You are a Flask API implementer. Write a complete, runnable Flask app.py. "
            "Import and use get_db() from db.py — do not redefine database functions. "
            "Follow these patterns:\n\n" + _FLASK_PATTERNS + "\n\n" + _SQLITE_PATTERNS + "\n\n"
            "Output ONE complete ```python block — the full app.py file ready to run. "
            "Include if __name__ == '__main__': app.run(debug=True) at the bottom."
        )
        user_prompt = (
            f"App to build: {question}\n\n"
            f"Database schema (schema.sql):\n```sql\n{_trim(schema_sql, 3000)}\n```\n\n"
            f"DB helper module (db.py — import from this, don't redefine):\n```python\n{_trim(db_py, 2000)}\n```\n\n"
            f"Project context:\n{_trim(project_context, 1500)}\n\n"
            "Write the complete Flask app.py with all routes implemented."
        )
    code = _extract_named_block(
        _chat(client, system_prompt, user_prompt, temperature=0.15, num_ctx=20480, timeout=480),
        ("python",),
    )

    # py_compile check + fix loop
    ok, err = _py_compile_check(code)
    if not ok:
        for cycle in range(1, 3):
            _prog("app_api_fix_cycle", {"cycle": cycle, "error": err[:200]})
            if callable(cancel_checker):
                try:
                    if cancel_checker():
                        break
                except Exception:
                    pass
            code = _fix_python(client, code, err, question, cancel_checker)
            ok, err = _py_compile_check(code)
            if ok:
                _prog("app_api_fix_passed", {"cycle": cycle})
                break
    return code


def _step_vue_architect(
    client: OllamaClient,
    question: str,
    flask_code: str,
) -> str:
    """Plan Vue 3 component structure based on actual Flask routes."""
    system_prompt = (
        f"Today: {_today()}. "
        "You are a Vue 3 frontend architect. Given a Flask backend, plan the frontend. "
        "List every Flask API route, then map each to a Vue interaction (fetch call, form, list render). "
        "Output a concise component plan — no code yet, just structure and data flow.\n\n"
        + _VUE3_PATTERNS
    )
    # Extract just the route definitions from flask_code to save context
    route_lines = [l for l in flask_code.splitlines() if "@app.route" in l or "def " in l]
    route_summary = "\n".join(route_lines[:40])
    user_prompt = (
        f"App: {question}\n\n"
        f"Flask routes defined:\n{route_summary}\n\n"
        "Plan the Vue 3 component structure and data flow. Be specific about which API calls go where."
    )
    return _chat(client, system_prompt, user_prompt, temperature=0.2, num_ctx=16384)


def _step_vue_implementer(
    client: OllamaClient,
    question: str,
    schema_sql: str,
    flask_code: str,
    vue_plan: str,
    existing_app_js: str = "",
    existing_index_html: str = "",
) -> tuple[str, str]:
    """Returns (index_html, app_js)."""
    route_lines = [l for l in flask_code.splitlines() if "@app.route" in l or "def " in l]
    route_summary = "\n".join(route_lines[:40])

    # app.js
    if existing_app_js:
        system_prompt_js = (
            f"Today: {_today()}. "
            "You are a Vue 3 frontend developer EXTENDING an existing app.js. "
            "Add new state, methods, and template sections for the new feature. "
            "Keep ALL existing state and methods unchanged — do not remove or rename them. "
            "Follow these patterns:\n\n" + _VUE3_PATTERNS + "\n\n"
            "Output ONE complete ```javascript block — the full updated app.js."
        )
        user_prompt_js = (
            f"New feature to add: {question}\n\n"
            f"Extension plan:\n{_trim(vue_plan, 1500)}\n\n"
            f"All Flask API routes (including new ones):\n{route_summary}\n\n"
            f"Existing app.js (keep all existing code — add new feature sections):\n"
            f"```javascript\n{_trim(existing_app_js, 4000)}\n```\n\n"
            "Output the complete updated app.js."
        )
    else:
        system_prompt_js = (
            f"Today: {_today()}. "
            "You are a Vue 3 frontend implementer. Write a complete app.js file. "
            "Follow these patterns exactly:\n\n" + _VUE3_PATTERNS + "\n\n"
            "Rules:\n"
            "- Use Vue 3 global build (CDN) — const { createApp, ref, reactive, computed, onMounted } = Vue;\n"
            "- All API calls use fetch() — no axios.\n"
            "- Handle loading and error states for every fetch.\n"
            "- Mount to #app.\n"
            "- Output ONE complete ```javascript block."
        )
        user_prompt_js = (
            f"App: {question}\n\n"
            f"Frontend plan:\n{_trim(vue_plan, 2000)}\n\n"
            f"Flask API routes available:\n{route_summary}\n\n"
            "Write the complete app.js — all state, API calls, and template logic."
        )
    app_js_raw = _chat(client, system_prompt_js, user_prompt_js, temperature=0.3, num_ctx=20480, timeout=480)
    app_js = _extract_named_block(app_js_raw, ("javascript", "js"))

    # index.html
    if existing_index_html:
        system_prompt_html = (
            f"Today: {_today()}. "
            "You are a Vue 3 HTML template developer EXTENDING an existing index.html. "
            "Add new template markup inside #app for the new feature. "
            "Keep ALL existing markup, CDN script tags, and stylesheet links unchanged. "
            "Output ONE complete ```html block — the full updated index.html."
        )
        user_prompt_html = (
            f"New feature to add: {question}\n\n"
            f"Updated app.js (for reference — use the same state variables and methods):\n"
            f"{_trim(app_js, 2000)}\n\n"
            f"Existing index.html (keep all existing markup — add new sections for the feature):\n"
            f"```html\n{_trim(existing_index_html, 3000)}\n```\n\n"
            "Output the complete updated index.html."
        )
    else:
        system_prompt_html = (
            f"Today: {_today()}. "
            "You are a Vue 3 HTML template writer. Write the index.html entry point. "
            "Follow these patterns:\n\n" + _VUE3_PATTERNS + "\n\n"
            "Rules:\n"
            "- Load Vue 3 from CDN: https://unpkg.com/vue@3/dist/vue.global.js\n"
            "- Load /static/app.js after Vue.\n"
            "- Link /static/styles.css.\n"
            "- The #app div contains the full template markup (inline, not in app.js).\n"
            "- Use v-bind, v-on shorthand (: and @).\n"
            "- Output ONE complete ```html block."
        )
        user_prompt_html = (
            f"App: {question}\n\n"
            f"Vue app.js structure:\n{_trim(app_js, 3000)}\n\n"
            "Write the complete index.html with all template markup inside #app."
        )
    index_html = _extract_named_block(
        _chat(client, system_prompt_html, user_prompt_html, temperature=0.25, num_ctx=20480, timeout=360),
        ("html",),
    )
    return index_html, app_js


def _step_integration_check(
    client: OllamaClient,
    question: str,
    flask_code: str,
    app_js: str,
    index_html: str,
) -> str:
    system_prompt = (
        f"Today: {_today()}. "
        "You are an integration checker. Compare a Flask backend and Vue 3 frontend. "
        "Find SPECIFIC mismatches only:\n"
        "- API routes in Flask not called in app.js\n"
        "- fetch() calls in app.js to routes that don't exist in Flask\n"
        "- JSON field names that differ between Flask response and Vue template\n"
        "- Missing CORS setup\n"
        "- Missing error handling\n"
        "List each issue as: [FILE] ISSUE: fix instruction. "
        "If no issues, say 'Integration looks clean.' and stop."
    )
    # Compress for context
    route_lines = [l for l in flask_code.splitlines() if "@app.route" in l or "return jsonify" in l]
    fetch_lines = [l for l in app_js.splitlines() if "fetch(" in l or "await " in l]
    user_prompt = (
        f"App: {question}\n\n"
        f"Flask routes/returns:\n{chr(10).join(route_lines[:50])}\n\n"
        f"Vue fetch calls:\n{chr(10).join(fetch_lines[:50])}\n\n"
        f"HTML template (first 2000 chars):\n{index_html[:2000]}"
    )
    return _chat(client, system_prompt, user_prompt, temperature=0.1, num_ctx=16384)


def _step_integration_fixer(
    client: OllamaClient,
    question: str,
    flask_code: str,
    app_js: str,
    integration_notes: str,
    cancel_checker: Callable[[], bool] | None,
) -> tuple[str, str]:
    """Apply integration_check findings to actual code. Returns (fixed_flask_code, fixed_app_js)."""

    def _is_cancelled() -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                pass
        return False

    # Flask fixer pass
    if not _is_cancelled():
        system_prompt = (
            f"Today: {_today()}. "
            "You are a Flask integration fixer. You will receive a Flask app.py and a list of "
            "integration issues found by an automated checker. "
            "Fix ALL listed issues in the Flask code. "
            "Do not change working code — only fix what is listed.\n\n"
            + _FLASK_PATTERNS + "\n\n" + _SQLITE_PATTERNS + "\n\n"
            "Return the complete corrected app.py in ONE ```python block. Do not truncate."
        )
        user_prompt = (
            f"App: {question}\n\n"
            f"Integration issues to fix:\n{_trim(integration_notes, 2000)}\n\n"
            f"Current app.py:\n```python\n{_trim(flask_code, 6000)}\n```\n\n"
            "Return the complete corrected app.py."
        )
        fixed_raw = _chat(client, system_prompt, user_prompt, temperature=0.1, num_ctx=20480, timeout=480)
        candidate = _extract_named_block(fixed_raw, ("python",))
        if candidate.strip():
            ok, err = _py_compile_check(candidate)
            if ok:
                flask_code = candidate
            else:
                # One more fix attempt
                flask_code = _fix_python(client, candidate, err, question, cancel_checker) or flask_code

    # Frontend fixer pass — runs after Flask so it can match any Flask changes
    if not _is_cancelled():
        route_lines = [l for l in flask_code.splitlines() if "@app.route" in l or "return jsonify" in l]
        route_summary = "\n".join(route_lines[:50])
        system_prompt = (
            f"Today: {_today()}. "
            "You are a Vue 3 frontend integration fixer. You will receive an app.js and a list of "
            "integration issues found by an automated checker. "
            "Fix ALL listed issues in the JavaScript. "
            "Ensure every fetch() URL matches an actual Flask route. "
            "Do not change working code — only fix what is listed.\n\n"
            + _VUE3_PATTERNS + "\n\n"
            "Return the complete corrected app.js in ONE ```javascript block. Do not truncate."
        )
        user_prompt = (
            f"App: {question}\n\n"
            f"Integration issues to fix:\n{_trim(integration_notes, 2000)}\n\n"
            f"Updated Flask routes (after backend fix):\n{route_summary}\n\n"
            f"Current app.js:\n```javascript\n{_trim(app_js, 6000)}\n```\n\n"
            "Return the complete corrected app.js."
        )
        fixed_raw = _chat(client, system_prompt, user_prompt, temperature=0.1, num_ctx=20480, timeout=480)
        candidate = _extract_named_block(fixed_raw, ("javascript", "js"))
        if candidate.strip():
            app_js = candidate

    return flask_code, app_js


def _step_css_writer(
    client: OllamaClient,
    question: str,
    index_html: str,
    existing_css: str = "",
) -> str:
    """Generate (or extend) styles.css from HTML structure. Returns css_str."""
    # Extract class names and IDs from the (updated) HTML
    classes = re.findall(r'class="([^"]+)"', index_html)
    all_classes = set()
    for cls_attr in classes:
        for cls in cls_attr.split():
            all_classes.add(f".{cls}")
    ids = re.findall(r'id="([^"]+)"', index_html)
    all_ids = {f"#{i}" for i in ids if i != "app"}
    selectors = sorted(all_classes | all_ids)
    selector_list = "\n".join(selectors[:80]) if selectors else "(no classes/IDs found)"

    if existing_css:
        system_prompt = (
            f"Today: {_today()}. "
            "You are a CSS developer EXTENDING an existing stylesheet. "
            "Add new selectors for the new feature's HTML elements. "
            "Keep ALL existing CSS rules exactly as-is — only append new rules. "
            "Match the existing color scheme and spacing conventions. "
            "Return ONE complete ```css block — existing rules first, new rules appended."
        )
        user_prompt = (
            f"New feature added: {question}\n\n"
            f"All selectors in updated index.html (new ones need styles):\n{selector_list}\n\n"
            f"Existing styles.css (keep all — add new selectors for the new feature):\n"
            f"```css\n{_trim(existing_css, 4000)}\n```\n\n"
            f"Updated HTML structure (for layout reference):\n{_trim(index_html, 2000)}\n\n"
            "Output the complete updated styles.css."
        )
    else:
        system_prompt = (
            f"Today: {_today()}. "
            "You are a CSS designer. Write a complete, professional styles.css for a Flask + Vue 3 web app. "
            "Requirements:\n"
            "1. CSS custom properties at :root for color scheme (primary, secondary, background, surface, text, border, error).\n"
            "2. Basic reset: *, *::before, *::after { box-sizing: border-box; } body margin: 0.\n"
            "3. Layout: flexbox or grid matching the HTML structure.\n"
            "4. Navigation/header styles if present.\n"
            "5. Form styles: input, select, textarea, button — clean and usable.\n"
            "6. Table styles if tables are used.\n"
            "7. Loading spinner or skeleton state for .loading class.\n"
            "8. Error message style for .error class.\n"
            "9. Responsive breakpoint at 768px using @media.\n"
            "10. Style every class and ID selector listed — do not leave them unstyled.\n"
            "Return ONE complete ```css block. Professional quality, ready to use."
        )
        user_prompt = (
            f"App: {question}\n\n"
            f"Selectors found in index.html:\n{selector_list}\n\n"
            f"HTML structure (for layout reference):\n{_trim(index_html, 4000)}\n\n"
            "Write the complete styles.css."
        )
    raw = _chat(client, system_prompt, user_prompt, temperature=0.3, num_ctx=16384, timeout=360)
    css_block = re.search(r"```(?:css)?\n(.*?)```", raw, re.DOTALL)
    if css_block:
        return css_block.group(1).strip()
    # If no fenced block, return raw (model may have returned plain CSS)
    cleaned = raw.strip()
    if cleaned.startswith(":root") or cleaned.startswith("/*") or cleaned.startswith("*"):
        return cleaned
    return "/* Add your styles here */\nbody { font-family: system-ui, sans-serif; margin: 0; }\n"


def _step_readme(
    client: OllamaClient,
    question: str,
    schema_sql: str,
    flask_code: str,
) -> str:
    # Extract imports to infer pip requirements
    import_lines = [l.strip() for l in flask_code.splitlines()
                    if l.strip().startswith(("import ", "from ")) and "flask" in l.lower()]
    system_prompt = (
        f"Today: {_today()}. "
        "Write a concise README.md for this Flask + Vue 3 + SQLite app. "
        "Include: Project description, Prerequisites, Installation, Database setup, "
        "Running the app, API endpoints list, File structure. "
        "Use markdown. Be practical and specific."
    )
    user_prompt = (
        f"App: {question}\n\n"
        f"Flask imports (infer pip requirements from these):\n{chr(10).join(import_lines)}\n\n"
        f"Schema:\n```sql\n{_trim(schema_sql, 1500)}\n```\n\n"
        "Write the README.md."
    )
    return _chat(client, system_prompt, user_prompt, temperature=0.3, num_ctx=12288)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_app_pool(
    question: str,
    repo_root: Path,
    project_slug: str,
    bus: Any,
    project_context: str = "",
    research_context: str = "",
    cancel_checker: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the full app generation pipeline and write output files."""

    def _prog(stage: str, detail: dict[str, Any] | None = None) -> None:
        if callable(progress_callback):
            try:
                progress_callback(stage, detail or {})
            except Exception:
                pass

    def _cancelled() -> bool:
        if callable(cancel_checker):
            try:
                return bool(cancel_checker())
            except Exception:
                return False
        return False

    bus.emit("app_pool", "start", {"question": question, "project": project_slug})
    client = OllamaClient()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    orchestrator_cfg = lane_model_config(repo_root, "orchestrator_reasoning")
    learning = FeedbackLearningEngine(repo_root, client=client, model_cfg=orchestrator_cfg)
    learned_guidance = learning.guidance_for_lane("make_app", limit=5)

    # Detect existing build — extend if found, build from scratch if not
    existing = _find_existing_app(repo_root, project_slug)
    is_extend = bool(existing)
    if is_extend:
        _prog("app_pool_extend_mode", {
            "source": existing.get("__source_dir__", "unknown"),
            "files_found": [k for k in existing if not k.startswith("__")],
            "message": f"Extending existing build: {existing.get('__source_dir__', '')}",
        })

    # Combine all context sources for the implementers
    combined_context = "\n\n".join(filter(None, [
        learned_guidance,
        _trim(project_context, 2000),
        _trim(research_context, 3000),
    ]))

    # ------------------------------------------------------------------
    # Step 1: Database schema + helpers
    # ------------------------------------------------------------------
    if _cancelled():
        return {"ok": False, "message": "Cancelled.", "files": {}}

    _prog("app_db_architect_started", {})
    schema_sql, db_py = _step_db_architect(
        client, question, combined_context,
        existing_schema=existing.get("schema.sql", ""),
        existing_db_py=existing.get("db.py", ""),
    )
    _prog("app_db_architect_completed", {"schema_lines": schema_sql.count("\n")})

    # ------------------------------------------------------------------
    # Step 2: Flask API implementer + py_compile fix loop
    # ------------------------------------------------------------------
    if _cancelled():
        return {"ok": False, "message": "Cancelled.", "files": {}}

    _prog("app_api_implementer_started", {})
    flask_code = _step_api_implementer(
        client, question, schema_sql, db_py, combined_context,
        cancel_checker, progress_callback,
        existing_flask_code=existing.get("app.py", ""),
    )
    _prog("app_api_implementer_completed", {"lines": flask_code.count("\n")})

    # Dependency check — zero model calls, catches missing pip packages early
    _present_deps, _missing_deps = _check_dependencies(flask_code)
    if _missing_deps:
        _prog("app_dependencies_warning", {
            "missing": _missing_deps,
            "install_hint": "pip install " + " ".join(_missing_deps),
        })

    # ------------------------------------------------------------------
    # Step 3: Vue architect — plan based on actual Flask routes
    # ------------------------------------------------------------------
    if _cancelled():
        return {"ok": False, "message": "Cancelled.", "files": {}}

    _prog("app_vue_architect_started", {})
    vue_plan = _step_vue_architect(client, question, flask_code)
    _prog("app_vue_architect_completed", {"preview": vue_plan[:200]})

    # ------------------------------------------------------------------
    # Step 4: Vue implementer — index.html + app.js
    # ------------------------------------------------------------------
    if _cancelled():
        return {"ok": False, "message": "Cancelled.", "files": {}}

    _prog("app_vue_implementer_started", {})
    index_html, app_js = _step_vue_implementer(
        client, question, schema_sql, flask_code, vue_plan,
        existing_app_js=existing.get("static/app.js", ""),
        existing_index_html=existing.get("templates/index.html", ""),
    )
    _prog("app_vue_implementer_completed", {"html_lines": index_html.count("\n"), "js_lines": app_js.count("\n")})

    # HTML structure check — zero model calls, catches Vue mount issues
    _html_issues = _check_html_structure(index_html)
    if _html_issues:
        _prog("app_html_issues", {"issues": _html_issues, "count": len(_html_issues)})

    # ------------------------------------------------------------------
    # Step 5: Integration check
    # ------------------------------------------------------------------
    if _cancelled():
        return {"ok": False, "message": "Cancelled.", "files": {}}

    _prog("app_integration_check_started", {})
    integration_notes = _step_integration_check(
        client, question, flask_code, app_js, index_html,
    )
    _prog("app_integration_check_completed", {"preview": integration_notes[:200]})

    # ------------------------------------------------------------------
    # Step 6: Integration fixer (only if issues found)
    # ------------------------------------------------------------------
    if not _cancelled() and "integration looks clean" not in integration_notes.lower():
        _prog("app_integration_fixer_started", {"issues_preview": integration_notes[:200]})
        flask_code, app_js = _step_integration_fixer(
            client, question, flask_code, app_js, integration_notes, cancel_checker,
        )
        _prog("app_integration_fixer_completed", {})
    else:
        _prog("app_integration_fixer_skipped", {"reason": "no issues found"})

    # ------------------------------------------------------------------
    # Step 7: CSS writer
    # ------------------------------------------------------------------
    if _cancelled():
        return {"ok": False, "message": "Cancelled.", "files": {}}

    _prog("app_css_writer_started", {})
    styles_css = _step_css_writer(
        client, question, index_html,
        existing_css=existing.get("static/styles.css", ""),
    )
    _prog("app_css_writer_completed", {"lines": styles_css.count("\n")})

    # ------------------------------------------------------------------
    # Step 8: README
    # ------------------------------------------------------------------
    _prog("app_readme_started", {})
    readme_md = _step_readme(client, question, schema_sql, flask_code)
    _prog("app_readme_completed", {})

    # ------------------------------------------------------------------
    # Write files to disk
    # ------------------------------------------------------------------
    app_dir = (
        repo_root / "Projects" / project_slug / "implementation" / f"{ts}_app"
    )
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "static").mkdir(exist_ok=True)
    (app_dir / "templates").mkdir(exist_ok=True)

    files_written: dict[str, str] = {}

    def _write(rel: str, content: str) -> None:
        path = app_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + "\n", encoding="utf-8")
        files_written[rel] = str(path)

    _write("schema.sql", schema_sql)
    _write("db.py", db_py)
    _write("app.py", flask_code)
    _write("templates/index.html", index_html)
    _write("static/app.js", app_js)
    _write("static/styles.css", styles_css)
    _write("README.md", readme_md)

    if integration_notes and "integration looks clean" not in integration_notes.lower():
        _write("INTEGRATION_NOTES.md", f"# Integration Review\n\n{integration_notes}\n")

    mode_line = (
        f"Mode: EXTEND — source build: `{existing.get('__source_dir__', '')}`"
        if is_extend else "Mode: NEW BUILD"
    )
    summary_md = (
        f"# App Build: {question[:80]}\n\n"
        f"Generated: {ts} | {mode_line}\n\n"
        f"## Files\n"
        + "\n".join(f"- `{rel}`" for rel in files_written)
        + f"\n\n## Integration Review\n\n{integration_notes}\n"
    )
    _write("BUILD_SUMMARY.md", summary_md)

    bus.emit("app_pool", "completed", {
        "project": project_slug,
        "path": str(app_dir),
        "files": list(files_written.keys()),
    })
    _prog("app_pool_completed", {"path": str(app_dir), "files": list(files_written.keys())})

    return {
        "ok": True,
        "message": f"App built: {len(files_written)} files in {app_dir.name}/",
        "path": str(app_dir),
        "files": files_written,
        "integration_notes": integration_notes,
    }
