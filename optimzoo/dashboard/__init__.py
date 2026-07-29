def run_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Launch the OptimZoo live dashboard (FastAPI + WebSocket server)."""
    import threading
    import webbrowser

    import uvicorn

    from optimzoo.dashboard.server import app

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()

    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["run_server"]
