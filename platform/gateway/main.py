
import asyncio
import asyncpg
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import nats
import redis
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prowzi")

class ProwziState:
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.nats_client: Optional[nats.NATS] = None
        self.redis_client: Optional[redis.Redis] = None
        self.agents: Dict[str, AgentStatus] = {}
        self.missions: Dict[str, Mission] = {}
        self.active_connections: List[WebSocket] = []
        self.agent_processes: Dict[str, subprocess.Popen] = {}

class AgentStatus(BaseModel):
    id: str
    name: str
    type: str
    status: str
    mission_id: Optional[str] = None
    created_at: datetime
    metrics: Dict[str, Any] = {}

class Mission(BaseModel):
    id: str
    name: str
    status: str
    created_at: datetime
    config: Dict[str, Any] = {}

class Event(BaseModel):
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str

class Brief(BaseModel):
    id: str
    headline: str
    content: Dict[str, Any]
    impact_level: str
    confidence_score: float
    created_at: datetime
    event_ids: List[str] = []

class CreateMissionRequest(BaseModel):
    name: str
    prompt: str
    config: Dict[str, Any] = {}

# Global state
state = ProwziState()

# FastAPI app
app = FastAPI(title="Prowzi Agent Platform", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="../web"), name="static")

@app.on_event("startup")
async def startup():
    logger.info("Starting Prowzi Agent Platform...")

    # Initialize database
    try:
        state.db_pool = await asyncpg.create_pool(
            "postgresql://postgres:password@localhost:5432/prowzi",
            min_size=1,
            max_size=20
        )
        logger.info("Database pool created successfully")

        # Initialize schema
        async with state.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    type VARCHAR(100) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    mission_id VARCHAR(255),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    metrics JSONB DEFAULT '{}'
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS missions (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    prompt TEXT NOT NULL,
                    status VARCHAR(50) NOT NULL DEFAULT 'planning',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    config JSONB DEFAULT '{}'
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id VARCHAR(255) PRIMARY KEY,
                    type VARCHAR(100) NOT NULL,
                    source VARCHAR(100) NOT NULL,
                    data JSONB NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS briefs (
                    id VARCHAR(255) PRIMARY KEY,
                    headline VARCHAR(500) NOT NULL,
                    content JSONB NOT NULL,
                    impact_level VARCHAR(20) NOT NULL,
                    confidence_score FLOAT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    event_ids TEXT[] DEFAULT '{}'
                )
            """)

        logger.info("Database schema initialized")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Initialize NATS
    try:
        state.nats_client = await nats.connect("nats://localhost:4222")
        logger.info("NATS client connected")
    except Exception as e:
        logger.warning(f"NATS not available - running in standalone mode: {e}")

    # Initialize Redis
    try:
        state.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        await asyncio.get_event_loop().run_in_executor(None, state.redis_client.ping)
        logger.info("Redis client connected")
    except Exception as e:
        logger.warning(f"Redis not available - running without caching: {e}")

    logger.info("Messaging system initialized")

    # Start some demo agents
    await start_demo_agents()

    logger.info("Prowzi Agent Platform started successfully")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down Prowzi Agent Platform...")
    
    # Stop all agents
    for agent_id, process in state.agent_processes.items():
        try:
            process.terminate()
            logger.info(f"Stopped agent {agent_id}")
        except Exception as e:
            logger.error(f"Error stopping agent {agent_id}: {e}")

    # Close connections
    if state.nats_client:
        await state.nats_client.close()
    
    if state.db_pool:
        await state.db_pool.close()

    logger.info("Shutdown complete")

async def start_demo_agents():
    """Start some demo agents for testing"""
    demo_agents = [
        {"name": "sensor-default-1", "type": "sensor"},
        {"name": "evaluator-default-1", "type": "evaluator"}
    ]
    
    for agent_config in demo_agents:
        try:
            # For demo purposes, just create a simple Python process
            process = subprocess.Popen([
                "python", "-c", 
                f"import time; print('Agent {agent_config['name']} started'); time.sleep(3600)"
            ])
            
            agent_id = agent_config["name"]
            state.agent_processes[agent_id] = process
            
            # Create agent status
            agent = AgentStatus(
                id=agent_id,
                name=agent_config["name"],
                type=agent_config["type"],
                status="running",
                created_at=datetime.now(timezone.utc),
                metrics={"uptime": 0, "processed_events": 0}
            )
            
            state.agents[agent_id] = agent
            
            # Store in database
            if state.db_pool:
                async with state.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO agents (id, name, type, status, created_at, metrics)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (id) DO UPDATE SET
                        status = $4, metrics = $6
                    """, agent_id, agent.name, agent.type, agent.status, 
                    agent.created_at, json.dumps(agent.metrics))
            
            logger.info(f"Started {agent.type} agent {agent_id} (PID: {process.pid})")
            
        except Exception as e:
            logger.error(f"Failed to start agent {agent_config['name']}: {e}")

# API Routes

@app.get("/")
async def root():
    return {"message": "Prowzi Agent Platform", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agents": len(state.agents),
        "missions": len(state.missions),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/v1/agents")
async def get_agents():
    """Get all agents"""
    try:
        if state.db_pool:
            async with state.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM agents ORDER BY created_at DESC")
                agents = []
                for row in rows:
                    agents.append({
                        "id": row["id"],
                        "name": row["name"],
                        "type": row["type"],
                        "status": row["status"],
                        "mission_id": row["mission_id"],
                        "created_at": row["created_at"].isoformat(),
                        "metrics": row["metrics"] or {}
                    })
                return agents
        else:
            return [agent.dict() for agent in state.agents.values()]
    except Exception as e:
        logger.error(f"Error fetching agents: {e}")
        return []

@app.get("/api/v1/missions")
async def get_missions():
    """Get all missions"""
    try:
        if state.db_pool:
            async with state.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM missions ORDER BY created_at DESC")
                missions = []
                for row in rows:
                    missions.append({
                        "id": row["id"],
                        "name": row["name"],
                        "status": row["status"],
                        "created_at": row["created_at"].isoformat(),
                        "config": row["config"] or {}
                    })
                return missions
        else:
            return [mission.dict() for mission in state.missions.values()]
    except Exception as e:
        logger.error(f"Error fetching missions: {e}")
        return []

@app.post("/api/v1/missions")
async def create_mission(request: CreateMissionRequest):
    """Create a new mission"""
    try:
        mission_id = str(uuid.uuid4())
        mission = Mission(
            id=mission_id,
            name=request.name,
            status="planning",
            created_at=datetime.now(timezone.utc),
            config=request.config
        )
        
        state.missions[mission_id] = mission
        
        # Store in database
        if state.db_pool:
            async with state.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO missions (id, name, prompt, status, created_at, config)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, mission_id, request.name, request.prompt, mission.status,
                mission.created_at, json.dumps(mission.config))
        
        # Broadcast to connected clients
        await broadcast_to_websockets({
            "type": "mission_update",
            "mission": mission.dict()
        })
        
        return mission.dict()
        
    except Exception as e:
        logger.error(f"Error creating mission: {e}")
        raise HTTPException(status_code=500, detail="Failed to create mission")

@app.get("/api/v1/events")
async def get_events():
    """Get recent events"""
    try:
        # Generate some demo events
        demo_events = [
            {
                "id": str(uuid.uuid4()),
                "type": "sensor_data",
                "source": "solana_mempool",
                "data": {"transaction_count": 1250, "gas_price": 0.000005},
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "type": "agent_status",
                "source": "orchestrator",
                "data": {"agent_id": "sensor-default-1", "status": "running"},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        if state.db_pool:
            async with state.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM events ORDER BY timestamp DESC LIMIT 100")
                events = []
                for row in rows:
                    events.append({
                        "id": row["id"],
                        "type": row["type"],
                        "source": row["source"],
                        "data": row["data"],
                        "timestamp": row["timestamp"].isoformat()
                    })
                return events if events else demo_events
        else:
            return demo_events
            
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        return []

@app.get("/api/v1/briefs")
async def get_briefs():
    """Get recent briefs"""
    try:
        # Generate some demo briefs
        demo_briefs = [
            {
                "id": str(uuid.uuid4()),
                "headline": "High volume detected on Solana mempool",
                "content": {
                    "summary": "Unusual spike in transaction volume detected in the last 10 minutes.",
                    "evidence": [
                        {"text": "Transaction count increased by 300%", "confidence": 0.95}
                    ],
                    "suggestedActions": ["Monitor for potential network congestion"]
                },
                "impactLevel": "medium",
                "confidenceScore": 0.85,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "eventIds": [str(uuid.uuid4())]
            },
            {
                "id": str(uuid.uuid4()),
                "headline": "New GitHub repository with high star velocity",
                "content": {
                    "summary": "Repository 'awesome-ai-agents' gained 500+ stars in 2 hours.",
                    "evidence": [
                        {"text": "Star growth rate: 250 stars/hour", "confidence": 0.98}
                    ],
                    "suggestedActions": ["Analyze repository content", "Track contributor activity"]
                },
                "impactLevel": "high",
                "confidenceScore": 0.92,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "eventIds": [str(uuid.uuid4())]
            }
        ]
        
        if state.db_pool:
            async with state.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM briefs ORDER BY created_at DESC LIMIT 50")
                briefs = []
                for row in rows:
                    briefs.append({
                        "id": row["id"],
                        "headline": row["headline"],
                        "content": row["content"],
                        "impactLevel": row["impact_level"],
                        "confidenceScore": row["confidence_score"],
                        "createdAt": row["created_at"].isoformat(),
                        "eventIds": row["event_ids"] or []
                    })
                return briefs if briefs else demo_briefs
        else:
            return demo_briefs
            
    except Exception as e:
        logger.error(f"Error fetching briefs: {e}")
        return []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.active_connections.append(websocket)
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            # Echo back for now
            await websocket.send_text(f"Echo: {data}")
            
    except WebSocketDisconnect:
        state.active_connections.remove(websocket)
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in state.active_connections:
            state.active_connections.remove(websocket)

async def broadcast_to_websockets(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if not state.active_connections:
        return
        
    message_str = json.dumps(message)
    disconnected = []
    
    for websocket in state.active_connections:
        try:
            await websocket.send_text(message_str)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for websocket in disconnected:
        state.active_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
