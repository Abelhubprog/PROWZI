rust
use futures_util::StreamExt;
use solana_client::rpc_client::RpcClient;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_transaction_status::UiTransactionEncoding;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use url::Url;

use prowzi_core::{Actor, ActorConfig, ActorContext, Message as ActorMessage};
use prowzi_messages::{EnrichedEvent, Domain, EventPayload, ExtractedData, Entity};

pub struct SolanaMempoolSensor {
    rpc_client: RpcClient,
    ws_url: String,
    output_topic: String,
    filter_programs: Vec<String>,
}

#[async_trait]
impl Actor for SolanaMempoolSensor {
    async fn init(&mut self, config: ActorConfig, ctx: &mut ActorContext) -> Result<(), ActorError> {
        self.rpc_client = RpcClient::new_with_commitment(
            config.get("rpc_endpoint").unwrap(),
            CommitmentConfig::confirmed(),
        );

        self.ws_url = config.get("ws_endpoint").unwrap();
        self.output_topic = config.get("output_topic").unwrap_or("sensor.solana_mempool");

        // Subscribe to token program by default
        self.filter_programs = vec![
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".to_string(),
            "11111111111111111111111111111111".to_string(),
        ];

        ctx.set_heartbeat_interval(Duration::from_secs(10));

        Ok(())
    }

    async fn run(&mut self, ctx: &mut ActorContext) -> Result<(), ActorError> {
        let ws_url = Url::parse(&self.ws_url)?;
        let (ws_stream, _) = connect_async(ws_url).await?;
        let (_, mut read) = ws_stream.split();

        // Subscribe to program logs
        let subscribe_msg = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {
                    "mentions": self.filter_programs
                },
                {
                    "commitment": "confirmed"
                }
            ]
        });

        // Main event loop
        while let Some(msg) = read.next().await {
            let msg = msg?;

            if let Message::Text(text) = msg {
                match self.process_message(&text).await {
                    Ok(Some(event)) => {
                        ctx.publish(&self.output_topic, event).await?;
                        metrics::EVENTS_EMITTED.inc();
                    }
                    Ok(None) => {
                        // Filtered out
                    }
                    Err(e) => {
                        tracing::error!("Error processing message: {}", e);
                        metrics::PROCESSING_ERRORS.inc();
                    }
                }
            }
        }

        Ok(())
    }
}

impl SolanaMempoolSensor {
    async fn process_message(&self, text: &str) -> Result<Option<EnrichedEvent>, SensorError> {
        let value: serde_json::Value = serde_json::from_str(text)?;

        // Extract transaction details
        if value["method"] == "logsNotification" {
            let result = &value["params"]["result"];
            let signature = result["value"]["signature"].as_str().unwrap();
            let logs = result["value"]["logs"].as_array().unwrap();

            // Detect token launch
            if self.is_token_launch(logs) {
                let event = self.create_token_launch_event(signature, logs).await?;
                return Ok(Some(event));
            }

            // Detect liquidity event
            if self.is_liquidity_event(logs) {
                let event = self.create_liquidity_event(signature, logs).await?;
                return Ok(Some(event));
            }
        }

        Ok(None)
    }

    fn is_token_launch(&self, logs: &[serde_json::Value]) -> bool {
        logs.iter().any(|log| {
            let text = log.as_str().unwrap_or("");
            text.contains("InitializeMint") || text.contains("InitializeAccount")
        })
    }

    async fn create_token_launch_event(
        &self,
        signature: &str,
        logs: &[serde_json::Value],
    ) -> Result<EnrichedEvent, SensorError> {
        let start_time = Instant::now();

        // Extract token address from logs
        let token_address = self.extract_token_address(logs)?;

        // Fetch additional data
        let token_info = self.fetch_token_info(&token_address).await?;

        // Create event
        let event = EnrichedEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            mission_id: None,
            timestamp: chrono::Utc::now(),
            domain: Domain::Crypto,
            source: "solana_mempool".to_string(),
            topic_hints: vec!["token_launch".to_string()],
            payload: EventPayload {
                raw: json!({
                    "signature": signature,
                    "token_address": token_address,
                    "logs": logs,
                }),
                extracted: ExtractedData {
                    entities: vec![
                        Entity {
                            entity_type: "token".to_string(),
                            id: token_address.clone(),
                            attributes: token_info,
                        }
                    ],
                    metrics: HashMap::new(),
                    sentiment: None,
                },
                embeddings: vec![], // Will be filled by enricher
            },
            metadata: EventMetadata {
                content_hash: self.calculate_hash(signature),
                geo_location: None,
                language: "en".to_string(),
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            },
        };

        Ok(event)
    }
}
