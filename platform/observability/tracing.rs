use opentelemetry::{
  trace::{Tracer, Span, StatusCode},
  Context as OtelContext,
};
use std::time::Instant;

pub struct DistributedTracer {
  tracer: Box<dyn Tracer>,
  sampling_strategy: Arc<SamplingStrategy>,
}

impl DistributedTracer {
  pub fn trace_mission_execution<F, R>(
      &self,
      mission_id: &str,
      operation: &str,
      f: F,
  ) -> Result<R, TraceError>
  where
      F: FnOnce(&mut dyn Span) -> Result<R, Box<dyn std::error::Error>>,
  {
      let mut span = self.tracer
          .span_builder(operation)
          .with_kind(SpanKind::Internal)
          .with_attributes(vec![
              KeyValue::new("mission.id", mission_id.to_string()),
              KeyValue::new("service.name", "prowzi-core"),
          ])
          .start(&self.tracer);

      // Add custom sampling decision
      if !self.sampling_strategy.should_sample(mission_id, operation) {
          span.set_attribute(KeyValue::new("sampling.decision", "dropped"));
          return f(&mut span);
      }

      let start = Instant::now();

      // Execute operation
      let result = f(&mut span);

      // Add performance metrics
      span.set_attribute(KeyValue::new("duration.ms", start.elapsed().as_millis() as i64));

      // Set status based on result
      match &result {
          Ok(_) => span.set_status(StatusCode::Ok, "".to_string()),
          Err(e) => {
              span.set_status(StatusCode::Error, e.to_string());
              span.record_error(e.as_ref());
          }
      }

      result
  }

  pub fn create_agent_span(&self, agent_id: &str, operation: &str) -> AgentSpan {
      let span = self.tracer
          .span_builder(format!("agent.{}", operation))
          .with_attributes(vec![
              KeyValue::new("agent.id", agent_id.to_string()),
              KeyValue::new("agent.operation", operation.to_string()),
          ])
          .start(&self.tracer);

      AgentSpan::new(span, agent_id.to_string())
  }
}

pub struct AgentSpan {
  span: Box<dyn Span>,
  agent_id: String,
  checkpoints: Vec<Checkpoint>,
}

impl AgentSpan {
  pub fn checkpoint(&mut self, name: &str, metadata: serde_json::Value) {
      let checkpoint = Checkpoint {
          name: name.to_string(),
          timestamp: chrono::Utc::now(),
          metadata,
      };

      self.span.add_event(
          name.to_string(),
          vec![
              KeyValue::new("checkpoint.data", checkpoint.metadata.to_string()),
          ],
      );

      self.checkpoints.push(checkpoint);
  }

  pub fn record_resource_usage(&mut self, usage: ResourceUsage) {
      self.span.set_attributes(vec![
          KeyValue::new("resource.cpu_percent", usage.cpu_percent),
          KeyValue::new("resource.memory_mb", usage.memory_mb as i64),
          KeyValue::new("resource.tokens_used", usage.tokens_used as i64),
      ]);
  }
}

// Performance profiling
pub struct PerformanceProfiler {
  metrics_collector: Arc<MetricsCollector>,
  profile_store: Arc<ProfileStore>,
}

impl PerformanceProfiler {
  pub async fn profile_critical_path(
      &self,
      event_id: &str,
  ) -> Result<CriticalPathAnalysis, ProfileError> {
      // Collect all spans for the event
      let spans = self.collect_event_spans(event_id).await?;

      // Build execution graph
      let graph = self.build_execution_graph(&spans);

      // Find critical path
      let critical_path = self.find_critical_path(&graph);

      // Analyze bottlenecks
      let bottlenecks = self.identify_bottlenecks(&critical_path);

      // Generate optimization suggestions
      let suggestions = self.generate_optimization_suggestions(&bottlenecks);

      Ok(CriticalPathAnalysis {
          event_id: event_id.to_string(),
          total_duration: critical_path.total_duration(),
          critical_operations: critical_path.operations,
          bottlenecks,
          suggestions,
          flame_graph: self.generate_flame_graph(&spans),
      })
  }

  pub async fn continuous_profiling(&self) {
      let mut interval = tokio::time::interval(Duration::from_secs(60));

      loop {
          interval.tick().await;

          // Sample current operations
          let samples = self.collect_performance_samples().await;

          // Detect performance regressions
          for sample in samples {
              if let Some(regression) = self.detect_regression(&sample).await {
                  // Store profile for analysis
                  self.profile_store.store_regression(regression).await.unwrap();

                  // Alert if severe
                  if regression.severity > 0.8 {
                      self.alert_performance_regression(regression).await;
                  }
              }
          }

          // Update baseline metrics
          self.update_performance_baseline().await;
      }
  }
}
