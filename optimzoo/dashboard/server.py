from __future__ import annotations

import asyncio
import queue
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from optimzoo.dashboard.manager import manager

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="OptimZoo Dashboard")


class StartRunRequest(BaseModel):
    algorithm: str
    problem: str
    dimension: int = 2
    population_size: int = 40
    max_iterations: int = 200
    seed: int | None = None


@app.get("/api/algorithms")
def get_algorithms():
    return manager.list_algorithms()


@app.get("/api/problems")
def get_problems():
    return manager.list_problems()


@app.get("/api/landscape")
def get_landscape(problem: str, dimension: int = 2, resolution: int = 150):
    try:
        return manager.landscape_grid(problem, dimension, resolution)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs")
def start_run(req: StartRunRequest):
    try:
        run_id = manager.start_run(
            algorithm=req.algorithm,
            problem=req.problem,
            dimension=req.dimension,
            population_size=req.population_size,
            max_iterations=req.max_iterations,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    ok = manager.stop_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return {"stopped": True}


@app.websocket("/ws/runs/{run_id}")
async def run_events(websocket: WebSocket, run_id: str):
    await websocket.accept()
    handle = manager.get_run(run_id)
    if handle is None:
        await websocket.send_json({"type": "error", "message": "Unknown run_id"})
        await websocket.close()
        return

    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                event = await loop.run_in_executor(None, handle.queue.get, True, 30)
            except queue.Empty:
                await websocket.send_json({"type": "ping"})
                continue

            await websocket.send_json(event)
            if event.get("type") == "closed":
                break
    except WebSocketDisconnect:
        handle.stop_event.set()


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
