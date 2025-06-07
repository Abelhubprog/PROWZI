python
import asyncio
import aiohttp
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
import hashlib

from prowzi_core import Actor, ActorConfig, ActorContext
from prowzi_messages import EnrichedEvent, Domain, EventPayload, ExtractedData, Entity


class GitHubEventsSensor(Actor):

    def __init__(self):
        super().__init__()
        self.session: Optional[aiohttp.ClientSession] = None
        self.etag_cache: Dict[str, str] = {}
        self.watched_repos: List[str] = []
        self.api_token: str = ""
        self.output_topic: str = "sensor.github_events"

    async def init(self, config: ActorConfig, ctx: ActorContext) -> None:
        self.api_token = config.get("github_token")
        self.watched_repos = config.get("watched_repos", [])
        self.output_topic = config.get("output_topic", "sensor.github_events")

        # Add default crypto/AI repos if none specified
        if not self.watched_repos:
            self.watched_repos = [
                "solana-labs/solana",
                "ethereum/go-ethereum",
                "openai/openai-python",
                "huggingface/transformers",
                "langchain-ai/langchain",
            ]

        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"token {self.api_token}",
                "Accept": "application/vnd.github.v3+json",
            })

        # Set polling interval
        ctx.set_tick_interval(30)  # 30 seconds

    async def tick(self, ctx: ActorContext) -> None:
        """Poll GitHub API for new events"""
        tasks = []

        for repo in self.watched_repos:
            tasks.append(self.check_repo_events(repo, ctx))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def check_repo_events(self, repo: str, ctx: ActorContext) -> None:
        """Check a specific repo for new events"""
        try:
            # Use conditional requests with ETag
            headers = {}
            if repo in self.etag_cache:
                headers["If-None-Match"] = self.etag_cache[repo]

            url = f"https://api.github.com/repos/{repo}/events"
            async with self.session.get(url, headers=headers) as response:
                # Handle rate limits
                if response.status == 403:
                    reset_time = int(
                        response.headers.get("X-RateLimit-Reset", 0))
                    await self.handle_rate_limit(reset_time)
                    return

                # No new events
                if response.status == 304:
                    return

                # Update ETag
                if "ETag" in response.headers:
                    self.etag_cache[repo] = response.headers["ETag"]

                # Process events
                events = await response.json()
                for event in events:
                    enriched = await self.process_event(repo, event)
                    if enriched:
                        await ctx.publish(self.output_topic, enriched)

        except Exception as e:
            ctx.log_error(f"Error checking repo {repo}: {e}")

    async def process_event(self, repo: str,
                            event: Dict[str, Any]) -> Optional[EnrichedEvent]:
        """Process a GitHub event into an enriched event"""
        event_type = event.get("type", "")

        # Filter for interesting events
        interesting_types = {
            "ReleaseEvent",
            "PushEvent",
            "IssuesEvent",
            "PullRequestEvent",
            "CreateEvent",
            "SecurityAdvisoryEvent",
        }

        if event_type not in interesting_types:
            return None

        # Extract entities and metrics
        extracted = await self.extract_data(repo, event)

        # Create enriched event
        enriched = EnrichedEvent(
            event_id=str(uuid.uuid4()),
            mission_id=None,
            timestamp=datetime.now(timezone.utc),
            domain=Domain.AI if self.is_ai_repo(repo) else Domain.CRYPTO,
            source="github_events",
            topic_hints=self.get_topic_hints(event_type, event),
            payload=EventPayload(
                raw=event,
                extracted=extracted,
                embeddings=[],  # Will be filled by enricher
            ),
            metadata={
                "content_hash": self.calculate_hash(event),
                "language": "en",
                "processing_time_ms": 0,  # Will be updated
            })

        return enriched

    async def extract_data(self, repo: str, event: Dict[str,
                                                        Any]) -> ExtractedData:
        """Extract structured data from event"""
        entities = []
        metrics = {}

        # Repository entity
        entities.append(
            Entity(entity_type="repository",
                   id=repo,
                   attributes={
                       "name": repo.split("/")[-1],
                       "owner": repo.split("/")[0],
                   }))

        # Actor entity
        if "actor" in event:
            entities.append(
                Entity(entity_type="developer",
                       id=event["actor"]["login"],
                       attributes={
                           "type": event["actor"]["type"],
                       }))

        # Event-specific extraction
        if event["type"] == "ReleaseEvent":
            release = event["payload"]["release"]
            entities.append(
                Entity(entity_type="release",
                       id=release["tag_name"],
                       attributes={
                           "name": release["name"],
                           "prerelease": str(release["prerelease"]),
                           "draft": str(release["draft"]),
                       }))

            # Extract metrics
            if "assets" in release:
                metrics["asset_count"] = len(release["assets"])
                metrics["total_download_size"] = sum(
                    asset["size"] for asset in release["assets"])

        elif event["type"] == "PushEvent":
            metrics["commit_count"] = len(event["payload"]["commits"])
            metrics["distinct_commit_count"] = len(
                set(c["sha"] for c in event["payload"]["commits"]))

        return ExtractedData(
            entities=entities,
            metrics=metrics,
            sentiment=None,
        )

    def is_ai_repo(self, repo: str) -> bool:
        """Determine if repo is AI-related"""
        ai_indicators = [
            "ai",
            "ml",
            "machine-learning",
            "deep-learning",
            "neural",
            "transformer",
            "llm",
            "model",
            "openai",
            "anthropic",
            "huggingface",
        ]
        repo_lower = repo.lower()
        return any(indicator in repo_lower for indicator in ai_indicators)

    def get_topic_hints(self, event_type: str, event: Dict[str,
                                                           Any]) -> List[str]:
        """Generate topic hints for the event"""
        hints = [event_type.lower().replace("event", "")]

        if event_type == "ReleaseEvent":
            if event["payload"]["release"]["prerelease"]:
                hints.append("prerelease")
            else:
                hints.append("stable_release")

        elif event_type == "SecurityAdvisoryEvent":
            hints.append("security")
            hints.append("vulnerability")

        return hints

    def calculate_hash(self, event: Dict[str, Any]) -> str:
        """Calculate content hash for deduplication"""
        # Use event ID if available
        if "id" in event:
            return hashlib.sha256(str(event["id"]).encode()).hexdigest()

        # Otherwise hash the entire event
        event_str = json.dumps(event, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()
