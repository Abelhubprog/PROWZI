 //test/integration/framework.ts
    import { TestAgent } from './agents/test-agent'
import { MockDataGenerator } from './mocks/data-generator'
import { PerformanceMonitor } from './monitoring/performance'

export class IntegrationTestFramework {
  private testAgents: Map<string, TestAgent> = new Map()
  private mockData: MockDataGenerator
  private perfMonitor: PerformanceMonitor

  async setupTestEnvironment(): Promise<TestEnvironment> {
    // Spin up test infrastructure
    const env = await this.provisionTestCluster()

    // Initialize test databases
    await this.initializeTestDatabases(env)

    // Deploy test agents
    await this.deployTestAgents(env)

    // Set up monitoring
    this.perfMonitor = new PerformanceMonitor(env)

    return env
  }

  async runEndToEndTest(scenario: TestScenario): Promise<TestResult> {
    const env = await this.setupTestEnvironment()
    const results: TestResult = {
      scenario: scenario.name,
      passed: true,
      metrics: {},
      errors: [],
    }

    try {
      // Generate test data
      const testData = await this.mockData.generateScenarioData(scenario)

      // Execute test steps
      for (const step of scenario.steps) {
        const stepResult = await this.executeStep(step, testData, env)

        if (!stepResult.success) {
          results.passed = false
          results.errors.push(stepResult.error)

          if (scenario.stopOnFailure) {
            break
          }
        }

        // Collect metrics
        results.metrics[step.name] = stepResult.metrics
      }

      // Validate outcomes
      const validations = await this.validateScenario(scenario, env)
      results.validations = validations

      // Performance analysis
      results.performance = await this.perfMonitor.analyze()

    } finally {
      // Cleanup
      await this.teardownEnvironment(env)
    }

    return results
  }

  async stressTest(config: StressTestConfig): Promise<StressTestResult> {
    const env = await this.setupTestEnvironment()

    // Ramp up load
    const loadGenerator = new LoadGenerator(config)
    const startTime = Date.now()

    const results = {
      maxThroughput: 0,
      latencyP99: 0,
      errorRate: 0,
      resourceUtilization: {},
      breakingPoint: null,
    }

    for (let load = config.startLoad; load <= config.maxLoad; load += config.increment) {
      console.log(`Testing at ${load} requests/second...`)

      // Apply load
      const metrics = await loadGenerator.applyLoad(load, config.duration)

      // Check system health
      const health = await this.checkSystemHealth(env)

      if (health.healthy) {
        results.maxThroughput = load
        results.latencyP99 = metrics.latencyP99
        results.errorRate = metrics.errorRate
        results.resourceUtilization = health.resources
      } else {
        results.breakingPoint = {
          load,
          reason: health.failureReason,
          metrics,
        }
        break
      }

      // Cool down between tests
      await new Promise(resolve => setTimeout(resolve, config.cooldownMs))
    }

    results.duration = Date.now() - startTime
    return results
  }
}

// Chaos Testing
export class ChaosTestRunner {
  private chaosMonkey: ChaosMonkey

  async runChaosTest(config: ChaosConfig): Promise<ChaosTestResult> {
    const results = {
      scenarios: [],
      systemResilience: 0,
      recoveryMetrics: {},
    }

    for (const fault of config.faults) {
      console.log(`Injecting fault: ${fault.type}`)

      // Inject fault
      await this.chaosMonkey.inject(fault)

      // Monitor system behavior
      const behavior = await this.monitorDuringChaos(fault.duration)

      // Remove fault
      await this.chaosMonkey.remove(fault)

      // Monitor recovery
      const recovery = await this.monitorRecovery()

      results.scenarios.push({
        fault,
        behavior,
        recovery,
        passed: this.evaluateResilience(behavior, recovery),
      })
    }

    results.systemResilience = this.calculateResilienceScore(results.scenarios)
    return results
  }

  private async monitorDuringChaos(duration: number): Promise<ChaosBehavior> {
    const metrics = {
      availability: [],
      latency: [],
      errors: [],
      degradation: [],
    }

    const interval = 1000 // 1 second
    const iterations = duration / interval

    for (let i = 0; i < iterations; i++) {
      const snapshot = await this.captureSystemSnapshot()

      metrics.availability.push(snapshot.availability)
      metrics.latency.push(snapshot.latency)
      metrics.errors.push(snapshot.errorRate)
      metrics.degradation.push(snapshot.degradation)

      await new Promise(resolve => setTimeout(resolve, interval))
    }

    return {
      metrics,
      impactSeverity: this.calculateImpactSeverity(metrics),
      affectedServices: await this.identifyAffectedServices(),
    }
  }
}



    import { PactV3, MatchersV3 } from '@pact-foundation/pact'
import { prowziAPIClient } from '@/clients/api'

const { like, regex, datetime, eachLike } = MatchersV3

describe('Prowzi API Contracts', () => {
  const provider = new PactV3({
    consumer: 'prowzi-web',
    provider: 'prowzi-api',
    dir: './pacts',
  })

  describe('Mission Creation', () => {
    it('should create a mission successfully', async () => {
      await provider
        .uponReceiving('a request to create a mission')
        .withRequest({
          method: 'POST',
          path: '/api/v1/missions',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': regex(/Bearer .+/, 'Bearer token123'),
          },
          body: {
            prompt: like('Track Solana token launches'),
            constraints: like({
              maxDuration: 48,
              tokenBudget: 10000,
            }),
          },
        })
        .willRespondWith({
          status: 201,
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            id: regex(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/, 'uuid'),
            status: 'planning',
            plan: like({
              objectives: eachLike({
                id: like('obj-1'),
                description: like('Monitor Solana mempool'),
                priority: regex(/^(critical|high|medium|low)$/, 'high'),
              }),
              agents: eachLike({
                type: like('solana_sensor'),
                count: like(2),
              }),
            }),
            createdAt: datetime("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),
          },
        })

      await provider.executeTest(async (mockProvider) => {
        const client = new prowziAPIClient(mockProvider.url)
        const response = await client.createMission({
          prompt: 'Track Solana token launches',
          constraints: {
            maxDuration: 48,
            tokenBudget: 10000,
          },
        })

        expect(response.status).toBe('planning')
        expect(response.id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/)
      })
    })
  })

  describe('Brief Stream', () => {
    it('should handle SSE brief stream', async () => {
      await provider
        .uponReceiving('a request for brief stream')
        .withRequest({
          method: 'GET',
          path: '/api/v1/briefs/stream',
          query: {
            domain: 'crypto',
            severity: 'high',
          },
        })
        .willRespondWith({
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
          },
          body: like(`data: ${JSON.stringify({
            briefId: 'brief-123',
            headline: 'New Solana token detected',
            impactLevel: 'high',
          })}\n\n`),
        })

      await provider.executeTest(async (mockProvider) => {
        const eventSource = new EventSource(
          `${mockProvider.url}/api/v1/briefs/stream?domain=crypto&severity=high`
        )

        return new Promise((resolve) => {
          eventSource.onmessage = (event) => {
            const brief = JSON.parse(event.data)
            expect(brief.impactLevel).toBe('high')
            eventSource.close()
            resolve(undefined)
          }
        })
      })
    })
  })
})

apiVersion: source.toolkit.fluxcd.io/v1beta2
kind: GitRepository
metadata:
  name: prowzi
  namespace: flux-system
spec:
  interval: 1m
  ref:
    branch: main
  url: https://github.com/prowzi/prowzi
  secretRef:
    name: github-credentials
---
apiVersion: kustomize.toolkit.fluxcd.io/v1beta2
kind: Kustomization
metadata:
  name: prowzi-infrastructure
  namespace: flux-system
spec:
  interval: 10m
  path: ./infrastructure/k8s/base
  prune: true
  sourceRef:
    kind: GitRepository
    name: prowzi
  validation: client
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: gateway
      namespace: prowzi
    - apiVersion: apps/v1
      kind: StatefulSet
      name: orchestrator
      namespace: prowzi
---
apiVersion: helm.toolkit.fluxcd.io/v2beta1
kind: HelmRelease
metadata:
  name: prowzi
  namespace: prowzi
spec:
  interval: 10m
  chart:
    spec:
      chart: ./charts/prowzi
      sourceRef:
        kind: GitRepository
        name: prowzi
  values:
    global:
      environment: production
      image:
        tag: ${GIT_COMMIT_SHA}

    gateway:
      replicaCount: 5
      autoscaling:
        enabled: true
        minReplicas: 5
        maxReplicas: 20

    agents:
      sensor:
        solana:
          replicas: 3
          rpcEndpoint: ${SOLANA_MAINNET_RPC}
        github:
          replicas: 2
          rateLimit: 10000

  postRenderers:
    - kustomize:
        patches:
          - target:
              kind: Deployment
              name: gateway
            patch: |
              - op: add
                path: /spec/template/spec/containers/0/env/-
                value:
                  name: OPENAI_API_KEY
                  valueFrom:
                    secretKeyRef:
                      name: prowzi-secrets
                      key: openai-api-key

                      module "primary_region" {
  source = "../../modules/prowzi-region"

  region = "us-east-1"
  environment = "production"

  vpc_config = {
    cidr = "10.0.0.0/16"
    azs = ["us-east-1a", "us-east-1b", "us-east-1c"]
    private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
    public_subnets = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  }

  eks_config = {
    cluster_version = "1.28"
    node_groups = {
      cpu_optimized = {
        instance_types = ["c6i.2xlarge"]
        scaling = {
          min_size = 3
          max_size = 50
          desired_size = 10
        }
      }
      gpu_spot = {
        instance_types = ["g4dn.xlarge"]
        capacity_type = "SPOT"
        scaling = {
          min_size = 0
          max_size = 20
          desired_size = 2
        }
        taints = [{
          key = "nvidia.com/gpu"
          value = "true"
          effect = "NO_SCHEDULE"
        }]
      }
    }
  }

  rds_config = {
    engine = "postgres"
    engine_version = "15.4"
    instance_class = "db.r6g.2xlarge"
    allocated_storage = 500
    multi_az = true
    backup_retention_period = 30
  }
}

module "failover_region" {
  source = "../../modules/prowzi-region"

  region = "eu-west-1"
  environment = "production"
  role = "failover"

  # Similar configuration with regional adjustments
}

# Global resources
resource "aws_route53_zone" "prowzi" {
  name = "prowzi.io"
}

resource "aws_route53_health_check" "primary" {
  fqdn              = module.primary_region.alb_dns_name
  port              = 443
  type              = "HTTPS"
  resource_path     = "/health"
  failure_threshold = 3
  request_interval  = 30
}

resource "aws_route53_record" "api" {
  zone_id = aws_route53_zone.prowzi.zone_id
  name    = "api.prowzi.io"
  type    = "A"

  set_identifier = "primary"

  failover_routing_policy {
    type = "PRIMARY"
  }

  alias {
    name                   = module.primary_region.alb_dns_name
    zone_id                = module.primary_region.alb_zone_id
    evaluate_target_health = true
  }

  health_check_id = aws_route53_health_check.primary.id
}

apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'prowzi-prod'
        region: 'us-east-1'

    rule_files:
      - '/etc/prometheus/rules/*.yml'

    scrape_configs:
      - job_name: 'prowzi-apps'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - prowzi
                - prowzi-tenant-*
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__

      - job_name: 'node-exporter'
        kubernetes_sd_configs:
          - role: node
        relabel_configs:
          - source_labels: [__address__]
            regex: '(.*):10250'
            replacement: '${1}:9100'
            target_label: __address__

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
                - alertmanager:9093
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  prowzi-alerts.yml: |
    groups:
      - name: prowzi_critical
        interval: 30s
        rules:
          - alert: HighBriefLatency
            expr: |
              histogram_quantile(0.99, 
                sum(rate(prowzi_brief_generation_seconds_bucket[5m])) 
                by (le, domain)
              )

              apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'prowzi-prod'
        region: 'us-east-1'

    rule_files:
      - '/etc/prometheus/rules/*.yml'

    scrape_configs:
      - job_name: 'prowzi-apps'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - prowzi
                - prowzi-tenant-*
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__

      - job_name: 'node-exporter'
        kubernetes_sd_configs:
          - role: node
        relabel_configs:
          - source_labels: [__address__]
            regex: '(.*):10250'
            replacement: '${1}:9100'
            target_label: __address__

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
                - alertmanager:9093
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  prowzi-alerts.yml: |
    groups:
      - name: prowzi_critical
        interval: 30s
        rules:
          - alert: HighBriefLatency
            expr: |
              histogram_quantile(0.99, 
                sum(rate(prowzi_brief_generation_seconds_bucket[5m])) 
                by (le, domain)
              ) > 10
            for: 5m
            labels:
              severity: critical
              team: platform
            annotations:
              summary: "Brief generation P99 latency above 10s"
              description: "Domain {{ $labels.domain }} has P99 latency of {{ $value }}s"

          - alert: AgentPoolExhausted
            expr: |
              (sum(prowzi_agent_pool_capacity) by (agent_type) - 
               sum(prowzi_active_agents) by (agent_type)) < 2
            for: 1m
            labels:
              severity: critical
              team: platform
            annotations:
              summary: "Agent pool near capacity"
              description: "{{ $labels.agent_type }} pool has only {{ $value }} agents available"

          - alert: MissionFailureRate
            expr: |
              rate(prowzi_mission_failures_total[5m]) / 
              rate(prowzi_mission_completions_total[5m]) > 0.1
            for: 10m
            labels:
              severity: warning
              team: ml
            annotations:
              summary: "High mission failure rate"
              description: "{{ $value | humanizePercentage }} of missions failing"

          - alert: TokenBudgetExceeded
            expr: |
              sum(rate(prowzi_tokens_used_total[1h])) by (tenant_id) > 
              sum(prowzi_token_budget_limit) by (tenant_id)
            for: 5m
            labels:
              severity: warning
              team: billing
            annotations:
              summary: "Tenant exceeding token budget"
              description: "Tenant {{ $labels.tenant_id }} using tokens at {{ $value }}/hour"

              {
  "dashboard": {
    "title": "Prowzi Executive Dashboard",
    "panels": [
      {
        "title": "Real-Time KPIs",
        "gridPos": { "h": 4, "w": 24, "x": 0, "y": 0 },
        "type": "stat",
        "targets": [
          {
            "expr": "sum(increase(prowzi_revenue_total[24h]))",
            "legendFormat": "24h Revenue"
          },
          {
            "expr": "sum(prowzi_active_users)",
            "legendFormat": "Active Users"
          },
          {
            "expr": "sum(rate(prowzi_briefs_generated_total[5m])) * 60",
            "legendFormat": "Briefs/min"
          },
          {
            "expr": "avg(prowzi_brief_quality_score)",
            "legendFormat": "Avg Quality"
          }
        ]
      },
      {
        "title": "System Performance",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 4 },
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, sum(rate(prowzi_brief_generation_seconds_bucket[5m])) by (le))",
            "legendFormat": "P99 Latency"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(prowzi_brief_generation_seconds_bucket[5m])) by (le))",
            "legendFormat": "P95 Latency"
          },
          {
            "expr": "histogram_quantile(0.50, sum(rate(prowzi_brief_generation_seconds_bucket[5m])) by (le))",
            "legendFormat": "P50 Latency"
          }
        ]
      },
      {
        "title": "Cost Analysis",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 4 },
        "type": "piechart",
        "targets": [
          {
            "expr": "sum(prowzi_infrastructure_cost_dollars) by (component)",
            "legendFormat": "{{ component }}"
          }
        ]
      }
    ]
  }
}

apiVersion: v1
kind: ConfigMap
metadata:
  name: pulsar-broker-config
  namespace: prowzi
data:
  broker.conf: |
    # Cluster configuration
    clusterName=prowzi-prod
    zookeeperServers=zk-0.zk-svc:2181,zk-1.zk-svc:2181,zk-2.zk-svc:2181
    configurationStoreServers=zk-0.zk-svc:2181,zk-1.zk-svc:2181,zk-2.zk-svc:2181

    # Storage
    managedLedgerDefaultEnsembleSize=3
    managedLedgerDefaultWriteQuorum=3
    managedLedgerDefaultAckQuorum=2

    # Retention
    defaultRetentionTimeInMinutes=10080  # 7 days
    defaultRetentionSizeInMB=10240       # 10GB

    # Performance
    maxConcurrentLookupRequest=50000
    maxConcurrentTopicLoadRequest=5000

    # Tiered storage
    managedLedgerOffloadDriver=aws-s3
    s3ManagedLedgerOffloadBucket=prowzi-events-archive
    s3ManagedLedgerOffloadRegion=us-east-1
    managedLedgerOffloadThresholdInMB=1024

    # Functions
    functionsWorkerEnabled=true

  functions-worker.yml: |
    workerId: prowzi-functions
    workerHostname: pulsar-function
    workerPort: 6750

    functionRuntimeFactoryClassName: org.apache.pulsar.functions.runtime.kubernetes.KubernetesRuntimeFactory

    kubernetesContainerFactory:
      k8Uri: kubernetes.default.svc.cluster.local
      jobNamespace: prowzi-functions
      pulsarDockerImageName: apachepulsar/pulsar:3.1.0
      imagePullPolicy: IfNotPresent

      # Resource limits
      cpuRequest: 0.1
      memoryRequest: 256Mi
      cpuLimit: 1.0
      memoryLimit: 1Gi
      use pulsar::{Consumer, Producer, Pulsar, SerializeMessage, DeserializeMessage};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct RawEvent {
    pub event_id: String,
    pub source: String,
    pub payload: serde_json::Value,
    pub timestamp: i64,
}

#[derive(Serialize, Deserialize)]
pub struct EnrichedEvent {
    pub event_id: String,
    pub source: String,
    pub payload: serde_json::Value,
    pub enrichments: HashMap<String, serde_json::Value>,
    pub embeddings: Vec<f32>,
    pub timestamp: i64,
    pub processing_time_ms: u64,
}

impl SerializeMessage for EnrichedEvent {
    fn serialize_message(input: Self) -> Result<producer::Message, pulsar::Error> {
        let payload = serde_json::to_vec(&input).map_err(|e| {
            pulsar::Error::Custom(format!("Serialization error: {}", e))
        })?;
        Ok(producer::Message {
            payload,
            ..Default::default()
        })
    }
}

pub struct EnrichmentFunction {
    consumer: Consumer<RawEvent>,
    producer: Producer<EnrichedEvent>,
    enrichment_cache: HashMap<String, serde_json::Value>,
    embedding_model: EmbeddingModel,
}

impl EnrichmentFunction {
    pub async fn new(pulsar: &Pulsar) -> Result<Self, pulsar::Error> {
        let consumer = pulsar
            .consumer()
            .with_topic("persistent://prowzi/events/raw")
            .with_subscription("enrichment-function")
            .build()
            .await?;

        let producer = pulsar
            .producer()
            .with_topic("persistent://prowzi/events/enriched")
            .build()
            .await?;

        Ok(Self {
            consumer,
            producer,
            enrichment_cache: HashMap::new(),
            embedding_model: EmbeddingModel::load().await?,
        })
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let msg = self.consumer.recv().await?;
            let start_time = std::time::Instant::now();

            let raw_event = msg.deserialize()?;

            // Enrich event
            let enriched = self.enrich_event(raw_event).await?;

            // Send to next topic
            self.producer.send(enriched).await?;

            // Acknowledge processing
            self.consumer.ack(&msg).await?;

            // Update metrics
            EVENTS_PROCESSED.inc();
            PROCESSING_TIME.observe(start_time.elapsed().as_secs_f64());
        }
    }

    async fn enrich_event(&mut self, event: RawEvent) -> Result<EnrichedEvent, Box<dyn std::error::Error>> {
        let mut enrichments = HashMap::new();

        // Source-specific enrichment
        match event.source.as_str() {
            "solana_mempool" => {
                let token_info = self.enrich_token_info(&event.payload).await?;
                enrichments.insert("token_info".to_string(), token_info);

                let wallet_history = self.get_wallet_history(&event.payload).await?;
                enrichments.insert("wallet_history".to_string(), wallet_history);
            }
            "github_events" => {
                let repo_stats = self.get_repo_statistics(&event.payload).await?;
                enrichments.insert("repo_stats".to_string(), repo_stats);

                let contributor_info = self.get_contributor_info(&event.payload).await?;
                enrichments.insert("contributors".to_string(), contributor_info);
            }
            "arxiv" => {
                let citations = self.get_citation_count(&event.payload).await?;
                enrichments.insert("citations".to_string(), citations);

                let related_papers = self.find_related_papers(&event.payload).await?;
                enrichments.insert("related".to_string(), related_papers);
            }
            _ => {}
        }

        // Generate embeddings
        let text = self.extract_text(&event.payload);
        let embeddings = self.embedding_model.encode(&text).await?;

        Ok(EnrichedEvent {
            event_id: event.event_id,
            source: event.source,
            payload: event.payload,
            enrichments,
            embeddings,
            timestamp: event.timestamp,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }
}

import { EventEmitter } from 'events'
import WebSocket from 'ws'

export interface ProwziConfig {
  apiKey: string
  baseUrl?: string
  timeout?: number
  maxRetries?: number
}

export interface Mission {
  id: string
  status: 'planning' | 'active' | 'paused' | 'completed' | 'failed'
  plan: MissionPlan
  resourceUsage: ResourceUsage
  createdAt: Date
  completedAt?: Date
}

export interface Brief {
  id: string
  headline: string
  content: BriefContent
  impactLevel: 'critical' | 'high' | 'medium' | 'low'
  confidence: number
  eventIds: string[]
  createdAt: Date
}

export class ProwziClient extends EventEmitter {
  private config: Required<ProwziConfig>
  private ws?: WebSocket

  constructor(config: ProwziConfig) {
    super()
    this.config = {
      baseUrl: 'https://api.prowzi.io',
      timeout: 30000,
      maxRetries: 3,
      ...config,
    }
  }

  // Mission Management
  async createMission(options: CreateMissionOptions): Promise<Mission> {
    const response = await this.request('/missions', {
      method: 'POST',
      body: JSON.stringify(options),
    })

    return this.parseMission(response)
  }

  async getMission(missionId: string): Promise<Mission> {
    const response = await this.request(`/missions/${missionId}`)
    return this.parseMission(response)
  }

  async pauseMission(missionId: string): Promise<Mission> {
    const response = await this.request(`/missions/${missionId}/pause`, {
      method: 'POST',
    })
    return this.parseMission(response)
  }

  async resumeMission(missionId: string): Promise<Mission> {
    const response = await this.request(`/missions/${missionId}/resume`, {
      method: 'POST',
    })
    return this.parseMission(response)
  }

  // Brief Streaming
  subscribeToBriefs(options?:

