python
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timezone
import re
import io
import PyPDF2
from typing import Optional, Dict, List, Any

from prowzi_core import Actor, ActorConfig, ActorContext
from prowzi_messages import EnrichedEvent, Domain, EventPayload, ExtractedData, Entity

class ArxivRssSensor(Actor):
    def __init__(self):
        super().__init__()
        self.session: Optional[aiohttp.ClientSession] = None
        self.categories: List[str] = []
        self.last_check: Dict[str, datetime] = {}
        self.output_topic: str = "sensor.arxiv"

    async def init(self, config: ActorConfig, ctx: ActorContext) -> None:
        self.categories = config.get("categories", [
            "cs.AI",  # Artificial Intelligence
            "cs.LG",  # Machine Learning
            "cs.CL",  # Computation and Language
            "cs.CR",  # Cryptography and Security
            "cs.DC",  # Distributed Computing
        ])

        self.output_topic = config.get("output_topic", "sensor.arxiv")

        self.session = aiohttp.ClientSession()

        # Set polling interval (arXiv updates daily)
        ctx.set_tick_interval(3600)  # 1 hour

    async def tick(self, ctx: ActorContext) -> None:
        """Check arXiv RSS feeds"""
        tasks = []

        for category in self.categories:
            tasks.append(self.check_category(category, ctx))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def check_category(self, category: str, ctx: ActorContext) -> None:
        """Check a specific arXiv category"""
        try:
            url = f"http://export.arxiv.org/rss/{category}"

            async with self.session.get(url) as response:
                content = await response.text()

            # Parse RSS feed
            feed = feedparser.parse(content)

            # Track last check time
            current_time = datetime.now(timezone.utc)
            last_check = self.last_check.get(category, current_time)

            for entry in feed.entries:
                # Parse publication date
                pub_date = datetime.strptime(
                    entry.published, 
                    "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)

                # Only process new entries
                if pub_date > last_check:
                    enriched = await self.process_entry(category, entry)
                    if enriched:
                        await ctx.publish(self.output_topic, enriched)

            self.last_check[category] = current_time

        except Exception as e:
            ctx.log_error(f"Error checking category {category}: {e}")

    async def process_entry(self, category: str, entry: Dict[str, Any]) -> Optional[EnrichedEvent]:
        """Process an arXiv entry"""
        # Extract arXiv ID
        arxiv_id = entry.id.split("/")[-1]

        # Extract structured data
        extracted = await self.extract_data(arxiv_id, entry)

        # Determine domain
        domain = Domain.AI if category.startswith("cs.") else Domain.CRYPTO

        # Create enriched event
        enriched = EnrichedEvent(
            event_id=str(uuid.uuid4()),
            mission_id=None,
            timestamp=datetime.now(timezone.utc),
            domain=domain,
            source="arxiv",
            topic_hints=self.get_topic_hints(category, entry),
            payload=EventPayload(
                raw={
                    "arxiv_id": arxiv_id,
                    "title": entry.title,
                    "summary": entry.summary,
                    "authors": entry.authors,
                    "category": category,
                    "link": entry.link,
                },
                extracted=extracted,
                embeddings=[],  # Will be filled by enricher
            ),
            metadata={
                "content_hash": hashlib.sha256(arxiv_id.encode()).hexdigest(),
                "language": "en",
                "processing_time_ms": 0,
            }
        )

        return enriched

    async def extract_data(self, arxiv_id: str, entry: Dict[str, Any]) -> ExtractedData:
        """Extract structured data from arXiv entry"""
        entities = []
        metrics = {}

        # Paper entity
        entities.append(Entity(
            entity_type="paper",
            id=arxiv_id,
            attributes={
                "title": entry.title,
                "category": entry.arxiv_primary_category.get("term", ""),
            }
        ))

        # Author entities
        for author in entry.get("authors", []):
            entities.append(Entity(
                entity_type="author",
                id=author["name"],
                attributes={
                    "affiliation": author.get("affiliation", ""),
                }
            ))

        # Extract metrics from abstract
        abstract = entry.summary

        # Look for performance metrics
        metric_patterns = [
            (r"(\d+\.?\d*)\s*%\s*accuracy", "accuracy"),
            (r"(\d+\.?\d*)\s*%\s*improvement", "improvement"),
            (r"(\d+\.?\d*)x\s*faster", "speedup"),
            (r"(\d+\.?\d*)\s*BLEU", "bleu_score"),
            (r"perplexity\s*of\s*(\d+\.?\d*)", "perplexity"),
        ]

        for pattern, metric_name in metric_patterns:
            match = re.search(pattern, abstract, re.IGNORECASE)
            if match:
                metrics[metric_name] = float(match.group(1))

        # Try to download and extract more info from PDF
        pdf_metrics = await self.extract_pdf_metrics(arxiv_id)
        metrics.update(pdf_metrics)

        return ExtractedData(
            entities=entities,
            metrics=metrics,
            sentiment=None,
        )

    async def extract_pdf_metrics(self, arxiv_id: str) -> Dict[str, float]:
        """Extract metrics from PDF if available"""
        metrics = {}

        try:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            async with self.session.get(pdf_url) as response:
                if response.status == 200:
                    pdf_content = await response.read()

                    # Parse PDF
                    pdf_file = io.BytesIO(pdf_content)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)

                    # Extract text from first few pages
                    text = ""
                    for i in range(min(5, len(pdf_reader.pages))):
                        text += pdf_reader.pages[i].extract_text()

                    # Look for tables with results
                    # This is simplified - real implementation would be more sophisticated
                    if "Table" in text and "Results" in text:
                        metrics["has_results_table"] = 1.0

                    # Count references (rough estimate)
                    references = len(re.findall(r"\[\d+\]", text))
                    metrics["reference_count"] = float(references)

        except Exception as e:
            # PDF extraction is optional, don't fail the event
            pass

        return metrics

    def get_topic_hints(self, category: str, entry: Dict[str, Any]) -> List[str]:
        """Generate topic hints"""
        hints = [category.lower().replace(".", "_")]

        # Extract keywords from title
        title_lower = entry.title.lower()

        ai_keywords = [
            "transformer", "gpt", "bert", "llm", "language model",
            "neural", "deep learning", "reinforcement learning",
            "attention", "diffusion", "gan",
        ]

        crypto_keywords = [
            "blockchain", "consensus", "byzantine", "proof of",
            "zero knowledge", "zkp", "cryptographic", "privacy",
            "secure", "protocol",
        ]

        for keyword in ai_keywords:
            if keyword in title_lower:
                hints.append(keyword.replace(" ", "_"))

        for keyword in crypto_keywords:
            if keyword in title_lower:
                hints.append(keyword.replace(" ", "_"))

        return hints

# requirements.txt
aiohttp==3.9.0
feedparser==6.0.11
PyPDF2==3.0.1
prowzi-core==0.1.0
prowzi-messages==0.1.0
