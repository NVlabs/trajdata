"""
trajdata Web UI – Tornado-based Bokeh Server at root path /.

Run:
    python trajdata_webui/main.py [--port 5006] [--no-browser]
"""
import argparse
import sys
from pathlib import Path

# Make src/ and project root importable
_webui_dir    = Path(__file__).parent
_project_root = _webui_dir.parent
for _p in [str(_project_root), str(_project_root / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

from trajdata_webui.ui import build_ui


def modify_doc(doc):
    build_ui(doc)


def main():
    parser = argparse.ArgumentParser(description="trajdata Web UI")
    parser.add_argument("--port",       type=int,  default=5006)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app = Application(FunctionHandler(modify_doc))
    server = Server({"/": app}, port=args.port, num_procs=1)
    server.start()

    if not args.no_browser:
        server.io_loop.add_callback(server.show, "/")

    print(f"\n  trajdata Web UI  →  http://localhost:{args.port}/\n")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
