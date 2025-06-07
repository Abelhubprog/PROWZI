use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::interval;

#[derive(Debug, Clone)]
struct HeartbeatRecord {
    agent_id: String,
    agent_type: String,
    last_beat: Instant,
    mission_id: Option<String>,
}

pub struct GuardianService {
    heartbeats: Arc<RwLock<HashMap<String, HeartbeatRecord>>>,
    config: GuardianConfig,
    pagerduty: PagerDutyClient,
}

#[derive(Clone)]
struct GuardianConfig {
    heartbeat_timeout: Duration,
    check_interval: Duration,
    quarantine_threshold: u32,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config = GuardianConfig {
        heartbeat_timeout: Duration::from_secs(20),
        check_interval: Duration::from_secs(5),
        quarantine_threshold: 3,
    };

    let guardian = Arc::new(GuardianService {
        heartbeats: Arc::new(RwLock::new(HashMap::new())),
        config,
        pagerduty: PagerDutyClient::new(std::env::var("PAGERDUTY_KEY").unwrap()),
    });

    // Start heartbeat consumer
    tokio::spawn(consume_heartbeats(guardian.clone()));

    // Start health checker
    tokio::spawn(check_agent_health(guardian.clone()));

    // HTTP server for metrics
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/agents", get(list_agents))
        .route("/metrics", get(prometheus_metrics))
        .with_state(guardian);

    let addr = "0.0.0.0:8083".parse().unwrap();
    tracing::info!("Guardian listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn consume_heartbeats(guardian: Arc<GuardianService>) {
    let consumer = create_nats_consumer("heartbeat.*").await.unwrap();
    let mut stream = consumer.messages().await.unwrap();

    while let Some(msg) = stream.next().await {
        if let Ok(heartbeat) = serde_json::from_slice::<Heartbeat>(&msg.payload) {
            let mut heartbeats = guardian.heartbeats.write().await;

            heartbeats.insert(heartbeat.agent_id.clone(), HeartbeatRecord {
                agent_id: heartbeat.agent_id,
                agent_type: heartbeat.agent_type,
                last_beat: Instant::now(),
                mission_id: heartbeat.mission_id,
            });

            msg.ack().await.unwrap();
        }
    }
}

async fn check_agent_health(guardian: Arc<GuardianService>) {
    let mut interval = interval(guardian.config.check_interval);
    let publisher = create_nats_publisher("commands").await.unwrap();

    loop {
        interval.tick().await;

        let now = Instant::now();
        let mut to_terminate = Vec::new();

        {
            let heartbeats = guardian.heartbeats.read().await;

            for (agent_id, record) in heartbeats.iter() {
                let elapsed = now.duration_since(record.last_beat);

                if elapsed > guardian.config.heartbeat_timeout {
                    to_terminate.push((
                        agent_id.clone(),
                        record.agent_type.clone(),
                        record.mission_id.clone(),
                    ));
                }
            }
        }

        // Process terminations
        for (agent_id, agent_type, mission_id) in to_terminate {
            tracing::warn!(
                "Agent {} ({}) missed heartbeat, terminating",
                agent_id,
                agent_type
            );

            // Send terminate command
            let command = TerminateCommand {
                agent_id: agent_id.clone(),
                reason: "heartbeat_timeout".to_string(),
            };

            publisher.publish(serde_json::to_vec(&command).unwrap()).await.unwrap();

            // Remove from tracking
            guardian.heartbeats.write().await.remove(&agent_id);

            // Update metrics
            metrics::HEARTBEAT_MISSED_TOTAL
                .with_label_values(&[&agent_type])
                .inc();

            // Page SRE if critical
            if agent_type == "orchestrator" || agent_type == "gateway" {
                guardian.pagerduty.trigger_incident(
                    &format!("Critical agent {} failed", agent_id),
                    &format!("Agent type: {}, Mission: {:?}", agent_type, mission_id),
                ).await.ok();
            }
        }
    }
}

// Prometheus metrics
lazy_static! {
    static ref HEARTBEAT_MISSED_TOTAL: prometheus::CounterVec = 
        prometheus::register_counter_vec!(
            "prowzi_heartbeat_missed_total",
            "Total heartbeats missed by agent type",
            &["agent_type"]
        ).unwrap();

    static ref AGENTS_TERMINATED: prometheus::CounterVec = 
        prometheus::register_counter_vec!(
            "prowzi_agents_terminated_total",
            "Total agents terminated by reason",
            &["reason"]
        ).unwrap();
}
