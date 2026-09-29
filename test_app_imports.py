"""test_app_imports.py
Smoke test: imports app.py and walks through each page-render function
without actually rendering to a Streamlit frontend.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Mock streamlit so app.py's module-level calls don't crash
import types


class _MockSt:
    """Minimal streamlit mock that records calls instead of rendering."""

    def __init__(self):
        self.session_state = {}
        self.calls = []
        self._columns = []

    def __getattr__(self, name):
        # Return a callable for any st.<name> access
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None
        return _record

    # Special-case methods that return data
    def title(self, *args, **kwargs):
        self.calls.append(("title", args, kwargs))
        return None

    def markdown(self, *args, **kwargs):
        self.calls.append(("markdown", args, kwargs))
        return None

    def columns(self, *args, **kwargs):
        self.calls.append(("columns", args, kwargs))
        # Return a list of dummy context managers (proper class)
        n = args[0] if args and isinstance(args[0], int) else 1

        class _Col:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a, **k):
                return False

            def __getattr__(self_inner, name):
                def _rec(*a, **k):
                    return None
                return _rec

        return [_Col() for _ in range(n)]

    def button(self, *args, **kwargs):
        self.calls.append(("button", args, kwargs))
        return False

    def selectbox(self, *args, **kwargs):
        self.calls.append(("selectbox", args, kwargs))
        # options is positional arg #1 (after label)
        options = kwargs.get("options", None)
        if options is None and len(args) > 1:
            options = args[1]
        if not options:
            options = [-1, 0, 1]
        return options[0]

    def file_uploader(self, *args, **kwargs):
        return None

    def cache_resource(self, fn=None, **kwargs):
        def _wrap(f):
            def _call(*a, **k):
                return f(*a, **k)
            return _call
        if fn is None:
            return _wrap
        return _wrap(fn)

    def cache_data(self, fn=None, **kwargs):
        def _wrap(f):
            def _call(*a, **k):
                return f(*a, **k)
            return _call
        if fn is None:
            return _wrap
        return _wrap(fn)

    def plotly_chart(self, *args, **kwargs):
        self.calls.append(("plotly_chart", args, kwargs))
        return None

    def dataframe(self, *args, **kwargs):
        self.calls.append(("dataframe", args, kwargs))
        return None

    def spinner(self, *args, **kwargs):
        ctx = types.SimpleNamespace()
        ctx.__enter__ = lambda *a, **k: ctx
        ctx.__exit__ = lambda *a, **k: False
        return ctx

    def error(self, *args, **kwargs):
        self.calls.append(("error", args, kwargs))

    def warning(self, *args, **kwargs):
        self.calls.append(("warning", args, kwargs))

    def info(self, *args, **kwargs):
        self.calls.append(("info", args, kwargs))

    def exception(self, *args, **kwargs):
        self.calls.append(("exception", args, kwargs))

    def caption(self, *args, **kwargs):
        pass

    def rerun(self):
        pass

    def metric(self, *args, **kwargs):
        pass

    def download_button(self, *args, **kwargs):
        pass


mock_st = _MockSt()
sys.modules["streamlit"] = mock_st

# Now import app.py
print("=" * 60)
print("Importing app.py with mocked streamlit...")
print("=" * 60)
try:
    import app  # type: ignore
    print("[OK] app.py imported successfully")
    print(f"  Pages defined: {[p[0] for p in app.PAGES]}")
    print(f"  THEME keys: {list(app.THEME.keys())[:5]} ...")
except Exception as exc:
    print(f"[FAIL] app.py import failed: {exc}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try each page renderer
print("\n" + "=" * 60)
print("Running each page renderer (with mocked st)...")
print("=" * 60)
errors = 0
for label, key, _icon in app.PAGES:
    fn_name = f"page_{key}"
    fn = getattr(app, fn_name, None)
    if fn is None:
        print(f"  [SKIP] {label}: no function {fn_name}")
        continue
    try:
        mock_st.session_state = {"page": key}
        fn()
        print(f"  [OK] {label} ({fn_name})")
    except Exception as exc:
        print(f"  [FAIL] {label} ({fn_name}): {exc}")
        errors += 1

print(f"\n{'=' * 60}")
if errors == 0:
    print(f"ALL {len(app.PAGES)} PAGE RENDERERS COMPLETED WITHOUT ERRORS")
else:
    print(f"{errors} page renderers failed")
print("=" * 60)
