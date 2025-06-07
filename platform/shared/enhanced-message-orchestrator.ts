/**
 * Enhanced Message Flow Orchestrator for Prowzi Platform
 * 
 * Provides intelligent message routing, circuit breaker integration, event sourcing,
 * message replay capabilities, and comprehensive observability for distributed communications
 */

import { EventEmitter } from 'events';
import { MessageBus, Message, MessagePriority, MessageBusConfig } from './message-bus';
import { ConnectionManager, CircuitBreaker } from './connection-manager';
import { 
  BaseMessage, 
  MessageType, 
  validateMessage, 
  serializeMessage, 
  deserializeMessage 
} from './types/messages';
import { v4 as uuidv4 } from 'uuid';
import { performance } from 'perf_hooks';
import Redis from 'ioredis';

export interface OrchestratorConfig {
  nodeId: string;
  messageBus: MessageBusConfig;
  eventSourcing: {
    enabled: boolean;
    retentionDays: number;
    snapshotInterval: number;
    compressionEnabled: boolean;
  };
  circuitBreaker: {
    failureThreshold: number;
    resetTimeout: number;
    monitoringPeriod: number;
  };
  messageReplay: {
    enabled: boolean;
    maxReplayBatchSize: number;
    replayTimeout: number;
  };
  priorityQueues: {
    enabled: boolean;
    maxQueueSize: number;
    processingIntervals: {
      [key in MessagePriority]: number;
    };
  };
  observability: {
    enabled: boolean;
    metricsInterval: number;
    tracingEnabled: boolean;
    detailedLogging: boolean;
  };
}

export interface MessageEnvelope {
  id: string;
  message: BaseMessage;
  metadata: {
    receivedAt: number;
    processedAt?: number;
    routingHistory: string[];
    retryCount: number;
    originalPriority: MessagePriority;
    processingTimeMs?: number;
    circuitBreakerState?: string;
  };
  eventSourcing: {
    sequenceNumber: number;
    partitionKey: string;
    timestamp: number;
    eventType: string;
  };
}

export interface RoutingRule {
  id: string;
  pattern: string;
  destination: string;
  priority: number;
  condition?: (message: BaseMessage) => boolean;
  transformation?: (message: BaseMessage) => BaseMessage;
  circuitBreakerConfig?: {
    enabled: boolean;
    serviceName: string;
  };
}

export interface MessageFlowMetrics {
  totalMessages: number;
  messagesPerSecond: number;
  averageProcessingTime: number;
  messagesByPriority: Record<MessagePriority, number>;
  messagesByType: Record<string, number>;
  circuitBreakerStates: Record<string, string>;
  queueSizes: Record<MessagePriority, number>;
  errorRate: number;
  throughputTrend: number[];
}

export interface ReplayRequest {
  id: string;
  fromTimestamp: number;
  toTimestamp: number;
  messageTypes?: MessageType[];
  sourceFilter?: string;
  destinationFilter?: string;
  batchSize: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: {
    processed: number;
    total: number;
    currentBatch: number;
  };
}

export class EnhancedMessageOrchestrator extends EventEmitter {
  private config: OrchestratorConfig;
  private messageBus: MessageBus;
  private connectionManager: ConnectionManager;
  private redis: Redis;

  // Message routing and processing
  private routingRules: Map<string, RoutingRule> = new Map();
  private priorityQueues: Map<MessagePriority, MessageEnvelope[]> = new Map();
  private processingIntervals: Map<MessagePriority, NodeJS.Timeout> = new Map();
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();

  // Event sourcing
  private eventStore: EventStore;
  private sequenceNumber = 0;
  private snapshotManager: SnapshotManager;

  // Message replay
  private activeReplays: Map<string, ReplayRequest> = new Map();
  private replayWorkers: Map<string, NodeJS.Timeout> = new Map();

  // Observability
  private metrics: MessageFlowMetrics;
  private metricsCollectionInterval: NodeJS.Timeout | null = null;
  private tracingData: Map<string, TraceContext> = new Map();
  private messageHistory: Map<string, MessageEnvelope[]> = new Map();

  // State management
  private isInitialized = false;
  private isShuttingDown = false;
  private startTime = Date.now();

  constructor(
    config: OrchestratorConfig,
    connectionManager: ConnectionManager
  ) {
    super();
    this.config = config;
    this.connectionManager = connectionManager;
    this.messageBus = new MessageBus(config.messageBus);
    this.redis = connectionManager.getRedisConnection();

    this.initializeMetrics();
    this.eventStore = new EventStore(this.redis, config.eventSourcing);
    this.snapshotManager = new SnapshotManager(this.redis, config.eventSourcing);
    
    this.setupEventHandlers();
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      throw new Error('Orchestrator already initialized');
    }

    try {
      // Initialize message bus
      await this.messageBus.start();

      // Initialize priority queues
      if (this.config.priorityQueues.enabled) {
        this.initializePriorityQueues();
      }

      // Load routing rules from persistence
      await this.loadRoutingRules();

      // Initialize circuit breakers for known services
      await this.initializeCircuitBreakers();

      // Start event sourcing if enabled
      if (this.config.eventSourcing.enabled) {
        await this.eventStore.initialize();
        await this.snapshotManager.initialize();
      }

      // Start observability collection
      if (this.config.observability.enabled) {
        this.startMetricsCollection();
      }

      // Set up message handlers
      this.setupMessageHandlers();

      this.isInitialized = true;
      this.emit('initialized');

      console.log(`Enhanced Message Orchestrator initialized for node: ${this.config.nodeId}`);
    } catch (error) {
      this.emit('error', { type: 'initialization', error });
      throw error;
    }
  }

  private initializeMetrics(): void {
    this.metrics = {
      totalMessages: 0,
      messagesPerSecond: 0,
      averageProcessingTime: 0,
      messagesByPriority: {
        [MessagePriority.CRITICAL]: 0,
        [MessagePriority.HIGH]: 0,
        [MessagePriority.MEDIUM]: 0,
        [MessagePriority.LOW]: 0,
        [MessagePriority.BACKGROUND]: 0,
      },
      messagesByType: {},
      circuitBreakerStates: {},
      queueSizes: {
        [MessagePriority.CRITICAL]: 0,
        [MessagePriority.HIGH]: 0,
        [MessagePriority.MEDIUM]: 0,
        [MessagePriority.LOW]: 0,
        [MessagePriority.BACKGROUND]: 0,
      },
      errorRate: 0,
      throughputTrend: [],
    };
  }

  private setupEventHandlers(): void {
    // Message bus events
    this.messageBus.on('connected', () => {
      this.emit('message-bus:connected');
    });

    this.messageBus.on('error', (error) => {
      this.emit('message-bus:error', error);
    });

    // Connection manager events
    this.connectionManager.on('connection:unhealthy', (event) => {
      this.handleUnhealthyConnection(event);
    });

    this.connectionManager.on('connection:recovered', (event) => {
      this.handleConnectionRecovery(event);
    });
  }

  private initializePriorityQueues(): void {
    for (const priority of Object.values(MessagePriority)) {
      if (typeof priority === 'number') {
        this.priorityQueues.set(priority, []);
        
        // Start processing interval for each priority level
        const interval = this.config.priorityQueues.processingIntervals[priority];
        if (interval > 0) {
          const timer = setInterval(() => {
            this.processPriorityQueue(priority);
          }, interval);
          
          this.processingIntervals.set(priority, timer);
        }
      }
    }
  }

  private async loadRoutingRules(): Promise<void> {
    try {
      const rulesData = await this.redis.get(`orchestrator:${this.config.nodeId}:routing-rules`);
      if (rulesData) {
        const rules: RoutingRule[] = JSON.parse(rulesData);
        for (const rule of rules) {
          this.routingRules.set(rule.id, rule);
        }
        console.log(`Loaded ${rules.length} routing rules`);
      }
    } catch (error) {
      console.warn('Failed to load routing rules:', error);
    }
  }

  private async initializeCircuitBreakers(): Promise<void> {
    // Get unique service names from routing rules
    const serviceNames = new Set<string>();
    for (const rule of this.routingRules.values()) {
      if (rule.circuitBreakerConfig?.enabled && rule.circuitBreakerConfig.serviceName) {
        serviceNames.add(rule.circuitBreakerConfig.serviceName);
      }
    }

    // Initialize circuit breakers for each service
    for (const serviceName of serviceNames) {
      const circuitBreaker = new CircuitBreaker(
        this.config.circuitBreaker.failureThreshold,
        this.config.circuitBreaker.resetTimeout,
        this.config.circuitBreaker.monitoringPeriod
      );

      circuitBreaker.on('open', () => {
        this.emit('circuit-breaker:open', { service: serviceName });
        this.metrics.circuitBreakerStates[serviceName] = 'OPEN';
      });

      circuitBreaker.on('half-open', () => {
        this.emit('circuit-breaker:half-open', { service: serviceName });
        this.metrics.circuitBreakerStates[serviceName] = 'HALF_OPEN';
      });

      circuitBreaker.on('closed', () => {
        this.emit('circuit-breaker:closed', { service: serviceName });
        this.metrics.circuitBreakerStates[serviceName] = 'CLOSED';
      });

      this.circuitBreakers.set(serviceName, circuitBreaker);
    }
  }

  private setupMessageHandlers(): void {
    // Subscribe to all messages for orchestration
    this.messageBus.subscribe('*', async (message) => {
      await this.handleIncomingMessage(message);
    }, {
      priority: MessagePriority.HIGH,
      timeout: 30000,
    });

    // Subscribe to system messages for special handling
    this.messageBus.subscribe('system:*', async (message) => {
      await this.handleSystemMessage(message);
    }, {
      priority: MessagePriority.CRITICAL,
      timeout: 10000,
    });

    // Subscribe to replay requests
    this.messageBus.subscribe('orchestrator:replay:*', async (message) => {
      await this.handleReplayRequest(message);
    }, {
      priority: MessagePriority.MEDIUM,
      timeout: 60000,
    });
  }

  private async handleIncomingMessage(message: Message): Promise<void> {
    const startTime = performance.now();
    const envelope = await this.createMessageEnvelope(message);

    try {
      // Validate message
      const validation = validateMessage(envelope.message);
      if (!validation.valid) {
        throw new Error(`Invalid message: ${validation.errors.join(', ')}`);
      }

      // Store in event store if enabled
      if (this.config.eventSourcing.enabled) {
        await this.eventStore.append(envelope);
      }

      // Apply routing rules
      const matchingRules = this.findMatchingRules(envelope.message);
      
      // Process through priority queue or direct routing
      if (this.config.priorityQueues.enabled) {
        await this.enqueueMessage(envelope);
      } else {
        await this.routeMessage(envelope, matchingRules);
      }

      // Update metrics
      const processingTime = performance.now() - startTime;
      envelope.metadata.processingTimeMs = processingTime;
      this.updateMetrics(envelope, processingTime, true);

      // Add to tracing if enabled
      if (this.config.observability.tracingEnabled) {
        this.addToTrace(envelope);
      }

    } catch (error) {
      this.updateMetrics(envelope, performance.now() - startTime, false);
      this.emit('message:processing-error', { envelope, error });
      
      // Handle retry logic if needed
      if (envelope.metadata.retryCount < 3) {
        envelope.metadata.retryCount++;
        setTimeout(() => {
          this.handleIncomingMessage(message);
        }, Math.pow(2, envelope.metadata.retryCount) * 1000);
      }
    }
  }

  private async createMessageEnvelope(message: Message): Promise<MessageEnvelope> {
    const sequenceNumber = ++this.sequenceNumber;
    
    return {
      id: uuidv4(),
      message: message as BaseMessage,
      metadata: {
        receivedAt: Date.now(),
        routingHistory: [],
        retryCount: 0,
        originalPriority: message.priority,
      },
      eventSourcing: {
        sequenceNumber,
        partitionKey: this.generatePartitionKey(message),
        timestamp: Date.now(),
        eventType: `message:${message.type}`,
      },
    };
  }

  private generatePartitionKey(message: Message): string {
    // Use source for partitioning to maintain ordering per source
    return `${message.source}:${message.type}`;
  }

  private findMatchingRules(message: BaseMessage): RoutingRule[] {
    const matchingRules: RoutingRule[] = [];

    for (const rule of this.routingRules.values()) {
      if (this.messageMatchesRule(message, rule)) {
        matchingRules.push(rule);
      }
    }

    // Sort by priority (higher priority first)
    return matchingRules.sort((a, b) => b.priority - a.priority);
  }

  private messageMatchesRule(message: BaseMessage, rule: RoutingRule): boolean {
    // Check pattern match
    const regex = new RegExp(rule.pattern.replace(/\*/g, '.*'));
    if (!regex.test(`${message.source}:${message.type}`)) {
      return false;
    }

    // Check custom condition if present
    if (rule.condition && !rule.condition(message)) {
      return false;
    }

    return true;
  }

  private async enqueueMessage(envelope: MessageEnvelope): Promise<void> {
    const priority = envelope.message.priority;
    const queue = this.priorityQueues.get(priority);
    
    if (!queue) {
      throw new Error(`Priority queue for level ${priority} not found`);
    }

    // Check queue size limits
    if (queue.length >= this.config.priorityQueues.maxQueueSize) {
      // Remove oldest message if queue is full
      const removed = queue.shift();
      if (removed) {
        this.emit('message:queue-overflow', { removed, new: envelope });
      }
    }

    queue.push(envelope);
    this.metrics.queueSizes[priority] = queue.length;
  }

  private async processPriorityQueue(priority: MessagePriority): Promise<void> {
    if (this.isShuttingDown) return;

    const queue = this.priorityQueues.get(priority);
    if (!queue || queue.length === 0) return;

    const envelope = queue.shift();
    if (!envelope) return;

    this.metrics.queueSizes[priority] = queue.length;

    try {
      const matchingRules = this.findMatchingRules(envelope.message);
      await this.routeMessage(envelope, matchingRules);
    } catch (error) {
      this.emit('message:queue-processing-error', { envelope, error });
      
      // Re-queue with increased retry count
      if (envelope.metadata.retryCount < 3) {
        envelope.metadata.retryCount++;
        queue.push(envelope);
      }
    }
  }

  private async routeMessage(envelope: MessageEnvelope, rules: RoutingRule[]): Promise<void> {
    if (rules.length === 0) {
      // Default routing - broadcast
      await this.messageBus.publish(
        envelope.message.type,
        envelope.message,
        undefined,
        envelope.message.priority
      );
      return;
    }

    for (const rule of rules) {
      try {
        // Apply transformation if present
        let message = envelope.message;
        if (rule.transformation) {
          message = rule.transformation(message);
        }

        // Add routing history
        envelope.metadata.routingHistory.push(rule.id);

        // Check circuit breaker if enabled
        if (rule.circuitBreakerConfig?.enabled) {
          const circuitBreaker = this.circuitBreakers.get(rule.circuitBreakerConfig.serviceName);
          if (circuitBreaker) {
            envelope.metadata.circuitBreakerState = circuitBreaker.getState();
            
            await circuitBreaker.execute(async () => {
              await this.messageBus.publish(
                message.type,
                message,
                rule.destination,
                message.priority,
                { routingRule: rule.id, ...envelope.metadata }
              );
            });
          } else {
            await this.messageBus.publish(
              message.type,
              message,
              rule.destination,
              message.priority,
              { routingRule: rule.id, ...envelope.metadata }
            );
          }
        } else {
          await this.messageBus.publish(
            message.type,
            message,
            rule.destination,
            message.priority,
            { routingRule: rule.id, ...envelope.metadata }
          );
        }

        this.emit('message:routed', { envelope, rule });

      } catch (error) {
        this.emit('message:routing-error', { envelope, rule, error });
        
        // Continue with next rule if available
        if (rules.indexOf(rule) === rules.length - 1) {
          throw error; // Last rule failed, propagate error
        }
      }
    }
  }

  private async handleSystemMessage(message: Message): Promise<void> {
    // Handle system-specific messages with special priority
    switch (message.type) {
      case 'system:shutdown':
        await this.handleShutdownMessage(message);
        break;
      case 'system:health-check':
        await this.handleHealthCheckMessage(message);
        break;
      case 'system:metrics-request':
        await this.handleMetricsRequest(message);
        break;
      default:
        // Default system message handling
        break;
    }
  }

  private async handleReplayRequest(message: Message): Promise<void> {
    if (!this.config.messageReplay.enabled) {
      throw new Error('Message replay is not enabled');
    }

    const replayRequest: ReplayRequest = message.payload as ReplayRequest;
    this.activeReplays.set(replayRequest.id, replayRequest);

    // Start replay worker
    const worker = setInterval(async () => {
      await this.processReplayBatch(replayRequest.id);
    }, 1000);

    this.replayWorkers.set(replayRequest.id, worker);
    this.emit('replay:started', { requestId: replayRequest.id });
  }

  private async processReplayBatch(replayId: string): Promise<void> {
    const request = this.activeReplays.get(replayId);
    if (!request || request.status === 'completed' || request.status === 'failed') {
      const worker = this.replayWorkers.get(replayId);
      if (worker) {
        clearInterval(worker);
        this.replayWorkers.delete(replayId);
      }
      return;
    }

    try {
      request.status = 'processing';
      
      const events = await this.eventStore.getEventsBatch(
        request.fromTimestamp,
        request.toTimestamp,
        request.batchSize,
        request.progress.currentBatch,
        {
          messageTypes: request.messageTypes,
          sourceFilter: request.sourceFilter,
          destinationFilter: request.destinationFilter,
        }
      );

      for (const event of events) {
        // Replay the message through the system
        await this.handleIncomingMessage(event.message);
        request.progress.processed++;
      }

      request.progress.currentBatch++;

      if (events.length < request.batchSize) {
        // Replay completed
        request.status = 'completed';
        this.emit('replay:completed', { requestId: replayId });
      }

    } catch (error) {
      request.status = 'failed';
      this.emit('replay:error', { requestId: replayId, error });
    }
  }

  private updateMetrics(envelope: MessageEnvelope, processingTime: number, success: boolean): void {
    this.metrics.totalMessages++;
    this.metrics.messagesByPriority[envelope.message.priority]++;
    
    const messageType = envelope.message.type;
    this.metrics.messagesByType[messageType] = (this.metrics.messagesByType[messageType] || 0) + 1;

    // Update average processing time
    this.metrics.averageProcessingTime = 
      (this.metrics.averageProcessingTime * (this.metrics.totalMessages - 1) + processingTime) / 
      this.metrics.totalMessages;

    if (!success) {
      // Update error rate (simple moving average over last 1000 messages)
      const errorWindow = Math.min(this.metrics.totalMessages, 1000);
      this.metrics.errorRate = (this.metrics.errorRate * (errorWindow - 1) + 1) / errorWindow;
    }
  }

  private startMetricsCollection(): void {
    this.metricsCollectionInterval = setInterval(() => {
      this.collectAndEmitMetrics();
    }, this.config.observability.metricsInterval);
  }

  private collectAndEmitMetrics(): void {
    // Calculate messages per second
    const uptimeSeconds = (Date.now() - this.startTime) / 1000;
    this.metrics.messagesPerSecond = this.metrics.totalMessages / uptimeSeconds;

    // Update throughput trend (last 10 intervals)
    this.metrics.throughputTrend.push(this.metrics.messagesPerSecond);
    if (this.metrics.throughputTrend.length > 10) {
      this.metrics.throughputTrend.shift();
    }

    // Update circuit breaker states
    for (const [serviceName, breaker] of this.circuitBreakers) {
      this.metrics.circuitBreakerStates[serviceName] = breaker.getState();
    }

    this.emit('metrics:collected', this.metrics);
  }

  private addToTrace(envelope: MessageEnvelope): void {
    const traceId = envelope.message.correlationId || envelope.id;
    const traces = this.tracingData.get(traceId) || new TraceContext();
    
    traces.addSpan({
      operationName: `process-${envelope.message.type}`,
      startTime: envelope.metadata.receivedAt,
      endTime: envelope.metadata.processedAt || Date.now(),
      tags: {
        messageType: envelope.message.type,
        source: envelope.message.source,
        priority: envelope.message.priority,
        routingHistory: envelope.metadata.routingHistory,
      },
    });

    this.tracingData.set(traceId, traces);
  }

  private async handleUnhealthyConnection(event: any): void {
    // Adjust routing to avoid unhealthy connections
    const connectionKey = event.connection;
    console.warn(`Connection ${connectionKey} is unhealthy, adjusting routing`);
    
    // TODO: Implement routing adjustments based on connection health
  }

  private async handleConnectionRecovery(event: any): void {
    // Resume normal routing for recovered connections
    const connectionKey = event.connection;
    console.info(`Connection ${connectionKey} recovered, resuming normal routing`);
    
    // TODO: Implement routing recovery logic
  }

  private async handleShutdownMessage(message: Message): Promise<void> {
    console.log('Received shutdown message, initiating graceful shutdown');
    await this.shutdown();
  }

  private async handleHealthCheckMessage(message: Message): Promise<void> {
    const healthStatus = {
      orchestrator: {
        status: 'healthy',
        uptime: Date.now() - this.startTime,
        metrics: this.metrics,
        queueSizes: Object.fromEntries(this.metrics.queueSizes),
        circuitBreakerStates: this.metrics.circuitBreakerStates,
      },
      messageBus: this.messageBus.getStats(),
      connections: Object.fromEntries(this.connectionManager.getConnectionHealth() as Map<string, any>),
    };

    await this.messageBus.respond(message, healthStatus);
  }

  private async handleMetricsRequest(message: Message): Promise<void> {
    await this.messageBus.respond(message, this.metrics);
  }

  // Public API methods

  /**
   * Add a routing rule
   */
  addRoutingRule(rule: RoutingRule): void {
    this.routingRules.set(rule.id, rule);
    
    // Persist to Redis
    this.persistRoutingRules();
    
    // Initialize circuit breaker if needed
    if (rule.circuitBreakerConfig?.enabled && rule.circuitBreakerConfig.serviceName) {
      if (!this.circuitBreakers.has(rule.circuitBreakerConfig.serviceName)) {
        const circuitBreaker = new CircuitBreaker(
          this.config.circuitBreaker.failureThreshold,
          this.config.circuitBreaker.resetTimeout,
          this.config.circuitBreaker.monitoringPeriod
        );
        this.circuitBreakers.set(rule.circuitBreakerConfig.serviceName, circuitBreaker);
      }
    }

    this.emit('routing-rule:added', rule);
  }

  /**
   * Remove a routing rule
   */
  removeRoutingRule(ruleId: string): boolean {
    const removed = this.routingRules.delete(ruleId);
    if (removed) {
      this.persistRoutingRules();
      this.emit('routing-rule:removed', { ruleId });
    }
    return removed;
  }

  /**
   * Start message replay
   */
  async startMessageReplay(request: Omit<ReplayRequest, 'id' | 'status' | 'progress'>): Promise<string> {
    if (!this.config.messageReplay.enabled) {
      throw new Error('Message replay is not enabled');
    }

    const replayRequest: ReplayRequest = {
      id: uuidv4(),
      status: 'pending',
      progress: {
        processed: 0,
        total: 0,
        currentBatch: 0,
      },
      ...request,
    };

    await this.messageBus.publish(
      'orchestrator:replay:start',
      replayRequest,
      this.config.nodeId,
      MessagePriority.MEDIUM
    );

    return replayRequest.id;
  }

  /**
   * Get replay status
   */
  getReplayStatus(replayId: string): ReplayRequest | null {
    return this.activeReplays.get(replayId) || null;
  }

  /**
   * Get current metrics
   */
  getMetrics(): MessageFlowMetrics {
    return { ...this.metrics };
  }

  /**
   * Get routing rules
   */
  getRoutingRules(): RoutingRule[] {
    return Array.from(this.routingRules.values());
  }

  /**
   * Get message trace
   */
  getMessageTrace(traceId: string): TraceContext | null {
    return this.tracingData.get(traceId) || null;
  }

  private async persistRoutingRules(): Promise<void> {
    try {
      const rules = Array.from(this.routingRules.values());
      await this.redis.set(
        `orchestrator:${this.config.nodeId}:routing-rules`,
        JSON.stringify(rules)
      );
    } catch (error) {
      console.error('Failed to persist routing rules:', error);
    }
  }

  /**
   * Graceful shutdown
   */
  async shutdown(): Promise<void> {
    if (this.isShuttingDown) return;
    
    this.isShuttingDown = true;
    this.emit('shutdown:started');

    // Stop metrics collection
    if (this.metricsCollectionInterval) {
      clearInterval(this.metricsCollectionInterval);
    }

    // Stop priority queue processing
    for (const timer of this.processingIntervals.values()) {
      clearInterval(timer);
    }

    // Stop replay workers
    for (const worker of this.replayWorkers.values()) {
      clearInterval(worker);
    }

    // Flush remaining messages in priority queues
    for (const [priority, queue] of this.priorityQueues) {
      while (queue.length > 0) {
        const envelope = queue.shift();
        if (envelope) {
          try {
            const rules = this.findMatchingRules(envelope.message);
            await this.routeMessage(envelope, rules);
          } catch (error) {
            console.error('Failed to flush message during shutdown:', error);
          }
        }
      }
    }

    // Stop message bus
    await this.messageBus.stop();

    // Final snapshot if event sourcing is enabled
    if (this.config.eventSourcing.enabled) {
      await this.snapshotManager.createSnapshot(this.sequenceNumber);
    }

    this.emit('shutdown:complete');
  }
}

// Supporting classes

class EventStore {
  constructor(
    private redis: Redis,
    private config: OrchestratorConfig['eventSourcing']
  ) {}

  async initialize(): Promise<void> {
    // Set up Redis streams for event sourcing
    // Implementation would use Redis Streams for persistent, ordered storage
  }

  async append(envelope: MessageEnvelope): Promise<void> {
    if (!this.config.enabled) return;

    const streamKey = `events:${envelope.eventSourcing.partitionKey}`;
    const eventData = {
      sequenceNumber: envelope.eventSourcing.sequenceNumber,
      timestamp: envelope.eventSourcing.timestamp,
      eventType: envelope.eventSourcing.eventType,
      messageData: serializeMessage(envelope.message),
      metadata: JSON.stringify(envelope.metadata),
    };

    await this.redis.xadd(streamKey, '*', ...Object.entries(eventData).flat());
  }

  async getEventsBatch(
    fromTimestamp: number,
    toTimestamp: number,
    batchSize: number,
    batchNumber: number,
    filters?: {
      messageTypes?: MessageType[];
      sourceFilter?: string;
      destinationFilter?: string;
    }
  ): Promise<MessageEnvelope[]> {
    // Implementation would query Redis streams with filtering
    // This is a placeholder - actual implementation would be more complex
    return [];
  }
}

class SnapshotManager {
  constructor(
    private redis: Redis,
    private config: OrchestratorConfig['eventSourcing']
  ) {}

  async initialize(): Promise<void> {
    // Set up snapshot storage in Redis
  }

  async createSnapshot(sequenceNumber: number): Promise<void> {
    if (!this.config.enabled) return;

    const snapshot = {
      sequenceNumber,
      timestamp: Date.now(),
      // Add snapshot data as needed
    };

    await this.redis.set(
      `snapshots:latest`,
      JSON.stringify(snapshot)
    );
  }
}

class TraceContext {
  private spans: TraceSpan[] = [];

  addSpan(span: TraceSpan): void {
    this.spans.push(span);
  }

  getSpans(): TraceSpan[] {
    return [...this.spans];
  }
}

interface TraceSpan {
  operationName: string;
  startTime: number;
  endTime: number;
  tags: Record<string, any>;
}

export { EnhancedMessageOrchestrator, OrchestratorConfig, MessageEnvelope, RoutingRule };