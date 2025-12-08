#!/usr/bin/env python3
"""
CNN-Swin Fusion API - Main Entry Point
=======================================
Professional skin cancer classification API with GradCAM visualization.

Usage:
    python main.py                    # Development mode
    python main.py --prod             # Production mode
    uvicorn app.main:app --reload     # Direct uvicorn
"""
from __future__ import annotations

import argparse
import sys
import logging
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.ERROR)
from app.main import app, settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CNN-Swin Fusion API Server")
    parser.add_argument("--host", default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--prod", action="store_true", help="Production mode (no reload)")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", default=None, help="Log level")
    return parser.parse_args()


def main():
    """Run the application server."""
    args = parse_args()
    
    # Override settings with CLI args
    host = args.host or settings.host
    port = args.port or settings.port
    reload = not args.prod and settings.reload
    log_level = (args.log_level or settings.log_level).lower()
    
    import uvicorn
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  CNN-Swin Fusion API v{settings.app_version}             ║
║  Professional Skin Cancer Classification                 ║
╚══════════════════════════════════════════════════════════╝

🚀 Starting server...
   Host: {host}
   Port: {port}
   Mode: {'Production' if args.prod else 'Development'}
   Workers: {args.workers if args.prod else 1}
   Reload: {reload}
   Log Level: {log_level.upper()}

📚 Documentation:
   Swagger UI: http://{host}:{port}/docs
   ReDoc: http://{host}:{port}/redoc
   Health: http://{host}:{port}/api/v1/health

Press CTRL+C to stop
""")
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            reload_delay=1.0,
            reload_excludes=[
            "__pycache__",
            ".venv",
            "app/static/outputs",
            "app/static/uploads",
            ".pytest_cache",
        ],
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped gracefully")
        sys.exit(0)


if __name__ == "__main__":
    main()
