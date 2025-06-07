/**
 * Prowzi Platform Shared Message Types
 * 
 * This module defines the core message types used throughout the Prowzi platform
 * for communication between AI agents, services, and external systems.
 */

export interface BaseMessage {
  id: string;
  timestamp: Date;
  source: string;
  type: MessageType;
  version?: string;
  correlationId?: string;
  replyTo?: string;
  expiresAt?: Date;
  priority: MessagePriority;
  metadata?: Record<string, any>;
  
  // Event sourcing extensions
  eventSourcing?: {
    streamId: string;
    eventId: string;
    eventNumber: number;
    causationId?: string;
    eventVersion: number;
    aggregateId?: string;
    aggregateType?: string;
    isSnapshot?: boolean;
  };
  
  // Message flow tracking
  routing?: {
    hops: MessageHop[];
    totalHops: number;
    routingStartTime: number;
    routingStrategy: 'broadcast' | 'direct' | 'load-balance' | 'priority';
  };
  
  // Delivery guarantees
  delivery?: {
    guaranteeLevel: 'at-most-once' | 'at-least-once' | 'exactly-once';
    maxRetries: number;
    retryBackoffMs: number;
    deadLetterQueue?: string;
    acknowledgmentRequired: boolean;
  };
}

export enum MessageType {
  // Agent Communication
  AGENT_STATUS = 'agent_status',
  AGENT_COMMAND = 'agent_command',
  AGENT_RESPONSE = 'agent_response',
  AGENT_HEARTBEAT = 'agent_heartbeat',
  AGENT_SHUTDOWN = 'agent_shutdown',
  
  // Market Data
  MARKET_DATA = 'market_data',
  PRICE_UPDATE = 'price_update',
  VOLUME_UPDATE = 'volume_update',
  ORDERBOOK_UPDATE = 'orderbook_update',
  TRADE_EXECUTION = 'trade_execution',
  
  // Solana Specific
  SOLANA_TRANSACTION = 'solana_transaction',
  SOLANA_BLOCK = 'solana_block',
  SOLANA_ACCOUNT_UPDATE = 'solana_account_update',
  SOLANA_PROGRAM_LOG = 'solana_program_log',
  SOLANA_MEMPOOL = 'solana_mempool',
  
  // AI Analysis
  AI_ANALYSIS = 'ai_analysis',
  SENTIMENT_ANALYSIS = 'sentiment_analysis',
  ANOMALY_DETECTION = 'anomaly_detection',
  PREDICTION = 'prediction',
  PATTERN_RECOGNITION = 'pattern_recognition',
  
  // System Events
  SYSTEM_ALERT = 'system_alert',
  SYSTEM_ERROR = 'system_error',
  SYSTEM_WARNING = 'system_warning',
  SYSTEM_INFO = 'system_info',
  
  // Notifications
  NOTIFICATION = 'notification',
  WEBHOOK = 'webhook',
  EMAIL = 'email',
  DISCORD = 'discord',
  SLACK = 'slack',
  
  // Data Pipeline
  DATA_INGESTION = 'data_ingestion',
  DATA_PROCESSING = 'data_processing',
  DATA_VALIDATION = 'data_validation',
  DATA_ENRICHMENT = 'data_enrichment',
  
  // Custom
  CUSTOM = 'custom',
  
  // Event sourcing specific
  EVENT_SOURCING_SNAPSHOT = 'event_sourcing_snapshot',
  EVENT_SOURCING_REPLAY = 'event_sourcing_replay',
  EVENT_SOURCING_PROJECTION = 'event_sourcing_projection',
  
  // Message flow control
  MESSAGE_FLOW_BACKPRESSURE = 'message_flow_backpressure',
  MESSAGE_FLOW_CIRCUIT_BREAKER = 'message_flow_circuit_breaker',
  MESSAGE_FLOW_RATE_LIMIT = 'message_flow_rate_limit'
}

export enum MessagePriority {
  LOW = 1,
  NORMAL = 2,
  HIGH = 3,
  CRITICAL = 4,
  URGENT = 5
}

export interface AgentMessage extends BaseMessage {
  type: MessageType.AGENT_STATUS | MessageType.AGENT_COMMAND | MessageType.AGENT_RESPONSE | MessageType.AGENT_HEARTBEAT | MessageType.AGENT_SHUTDOWN;
  agentId: string;
  agentType: string;
  agentVersion: string;
  status?: AgentStatus;
  command?: AgentCommand;
  response?: AgentResponse;
  performance?: AgentPerformance;
}

export enum AgentStatus {
  INITIALIZING = 'initializing',
  IDLE = 'idle',
  RUNNING = 'running',
  BUSY = 'busy',
  ERROR = 'error',
  SHUTTING_DOWN = 'shutting_down',
  OFFLINE = 'offline'
}

export interface AgentCommand {
  command: string;
  parameters?: Record<string, any>;
  timeout?: number;
  retries?: number;
}

export interface AgentResponse {
  success: boolean;
  data?: any;
  error?: string;
  executionTime?: number;
  resultMetadata?: Record<string, any>;
}

export interface AgentPerformance {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage?: number;
  networkUsage?: number;
  tasksCompleted: number;
  tasksInProgress: number;
  averageResponseTime: number;
  errorRate: number;
  uptime: number;
}

export interface MarketDataMessage extends BaseMessage {
  type: MessageType.MARKET_DATA | MessageType.PRICE_UPDATE | MessageType.VOLUME_UPDATE | MessageType.ORDERBOOK_UPDATE | MessageType.TRADE_EXECUTION;
  symbol: string;
  exchange: string;
  data: MarketData;
}

export interface MarketData {
  price?: number;
  volume?: number;
  change24h?: number;
  changePercent24h?: number;
  high24h?: number;
  low24h?: number;
  marketCap?: number;
  supply?: number;
  orderbook?: OrderBook;
  trades?: Trade[];
  timestamp: Date;
}

export interface OrderBook {
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  spread: number;
  timestamp: Date;
}

export interface OrderBookEntry {
  price: number;
  quantity: number;
  orders?: number;
}

export interface Trade {
  id: string;
  price: number;
  quantity: number;
  side: 'buy' | 'sell';
  timestamp: Date;
  maker?: boolean;
}

export interface SolanaMessage extends BaseMessage {
  type: MessageType.SOLANA_TRANSACTION | MessageType.SOLANA_BLOCK | MessageType.SOLANA_ACCOUNT_UPDATE | MessageType.SOLANA_PROGRAM_LOG | MessageType.SOLANA_MEMPOOL;
  slot?: number;
  blockHeight?: number;
  signature?: string;
  programId?: string;
  accountKey?: string;
  data: SolanaData;
}

export interface SolanaData {
  transaction?: SolanaTransaction;
  block?: SolanaBlock;
  accountUpdate?: SolanaAccountUpdate;
  programLog?: SolanaProgramLog;
  mempoolData?: SolanaMempoolData;
}

export interface SolanaTransaction {
  signature: string;
  slot: number;
  blockTime?: number;
  fee: number;
  status: 'success' | 'failed';
  instructions: SolanaInstruction[];
  logs?: string[];
  computeUnitsConsumed?: number;
}

export interface SolanaInstruction {
  programId: string;
  accounts: string[];
  data: string;
  innerInstructions?: SolanaInstruction[];
}

export interface SolanaBlock {
  slot: number;
  parent_slot: number;
  blockhash: string;
  previous_blockhash: string;
  blockTime: number;
  transactions: SolanaTransaction[];
  rewards?: SolanaReward[];
}

export interface SolanaReward {
  pubkey: string;
  lamports: number;
  postBalance: number;
  rewardType: string;
  commission?: number;
}

export interface SolanaAccountUpdate {
  pubkey: string;
  lamports: number;
  owner: string;
  executable: boolean;
  rentEpoch: number;
  data?: string;
  slot: number;
}

export interface SolanaProgramLog {
  signature: string;
  programId: string;
  logs: string[];
  slot: number;
  timestamp: Date;
}

export interface SolanaMempoolData {
  signature: string;
  fee: number;
  priorityFee?: number;
  computeUnits?: number;
  accounts: string[];
  programIds: string[];
  timestamp: Date;
}

export interface AIAnalysisMessage extends BaseMessage {
  type: MessageType.AI_ANALYSIS | MessageType.SENTIMENT_ANALYSIS | MessageType.ANOMALY_DETECTION | MessageType.PREDICTION | MessageType.PATTERN_RECOGNITION;
  analysisType: string;
  modelId: string;
  modelVersion: string;
  confidence: number;
  data: AIAnalysisData;
}

export interface AIAnalysisData {
  input: any;
  output: any;
  confidence: number;
  reasoning?: string;
  alternatives?: AlternativeResult[];
  metadata?: Record<string, any>;
}

export interface AlternativeResult {
  output: any;
  confidence: number;
  reasoning?: string;
}

export interface SystemMessage extends BaseMessage {
  type: MessageType.SYSTEM_ALERT | MessageType.SYSTEM_ERROR | MessageType.SYSTEM_WARNING | MessageType.SYSTEM_INFO;
  level: 'info' | 'warning' | 'error' | 'critical';
  component: string;
  message: string;
  details?: Record<string, any>;
  stackTrace?: string;
  resolvedAt?: Date;
}

export interface NotificationMessage extends BaseMessage {
  type: MessageType.NOTIFICATION | MessageType.WEBHOOK | MessageType.EMAIL | MessageType.DISCORD | MessageType.SLACK;
  title?: string;
  content: string;
  channels: string[];
  template?: string;
  templateData?: Record<string, any>;
  recipients?: string[];
  urgent?: boolean;
}

export interface DataPipelineMessage extends BaseMessage {
  type: MessageType.DATA_INGESTION | MessageType.DATA_PROCESSING | MessageType.DATA_VALIDATION | MessageType.DATA_ENRICHMENT;
  pipelineId: string;
  stageId: string;
  status: 'started' | 'completed' | 'failed' | 'skipped';
  recordsProcessed?: number;
  recordsSucceeded?: number;
  recordsFailed?: number;
  duration?: number;
  error?: string;
  data?: any;
}

// Message Validation
export interface MessageValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export function validateMessage(message: BaseMessage): MessageValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Required fields validation
  if (!message.id) errors.push('Message ID is required');
  if (!message.timestamp) errors.push('Timestamp is required');
  if (!message.source) errors.push('Source is required');
  if (!message.type) errors.push('Message type is required');
  if (message.priority === undefined) errors.push('Priority is required');

  // Type validation
  if (!Object.values(MessageType).includes(message.type)) {
    errors.push(`Invalid message type: ${message.type}`);
  }

  // Priority validation
  if (![1, 2, 3, 4, 5].includes(message.priority)) {
    errors.push(`Invalid priority: ${message.priority}`);
  }

  // Expiration validation
  if (message.expiresAt && message.expiresAt <= new Date()) {
    warnings.push('Message has already expired');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

// Message Serialization/Deserialization
export function serializeMessage(message: BaseMessage): string {
  return JSON.stringify(message, (key, value) => {
    if (value instanceof Date) {
      return value.toISOString();
    }
    return value;
  });
}

export function deserializeMessage(data: string): BaseMessage {
  return JSON.parse(data, (key, value) => {
    if (key === 'timestamp' || key === 'expiresAt' || key.endsWith('At') || key.endsWith('Time')) {
      return new Date(value);
    }
    return value;
  });
}

// Message Factory
export class MessageFactory {
  static createAgentMessage(
    agentId: string,
    agentType: string,
    type: MessageType,
    data: Partial<AgentMessage>
  ): AgentMessage {
    return {
      id: this.generateId(),
      timestamp: new Date(),
      source: `agent:${agentId}`,
      type,
      priority: MessagePriority.NORMAL,
      agentId,
      agentType,
      agentVersion: '1.0.0',
      ...data
    };
  }

  static createSystemMessage(
    component: string,
    level: 'info' | 'warning' | 'error' | 'critical',
    message: string,
    details?: Record<string, any>
  ): SystemMessage {
    return {
      id: this.generateId(),
      timestamp: new Date(),
      source: `system:${component}`,
      type: this.getSystemMessageType(level),
      priority: this.getSystemMessagePriority(level),
      level,
      component,
      message,
      details
    };
  }

  static createNotificationMessage(
    title: string,
    content: string,
    channels: string[],
    priority: MessagePriority = MessagePriority.NORMAL
  ): NotificationMessage {
    return {
      id: this.generateId(),
      timestamp: new Date(),
      source: 'notification-service',
      type: MessageType.NOTIFICATION,
      priority,
      title,
      content,
      channels
    };
  }

  private static generateId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private static getSystemMessageType(level: string): MessageType {
    switch (level) {
      case 'info': return MessageType.SYSTEM_INFO;
      case 'warning': return MessageType.SYSTEM_WARNING;
      case 'error': return MessageType.SYSTEM_ERROR;
      case 'critical': return MessageType.SYSTEM_ALERT;
      default: return MessageType.SYSTEM_INFO;
    }
  }

  private static getSystemMessagePriority(level: string): MessagePriority {
    switch (level) {
      case 'info': return MessagePriority.LOW;
      case 'warning': return MessagePriority.NORMAL;
      case 'error': return MessagePriority.HIGH;
      case 'critical': return MessagePriority.CRITICAL;
      default: return MessagePriority.NORMAL;
    }
  }
}

// Extended interfaces for enhanced message flow

export interface MessageHop {
  nodeId: string;
  timestamp: number;
  processingTimeMs: number;
  routingDecision: string;
  circuitBreakerState?: string;
  queueDepth?: number;
}

export interface EventSourcingCommand extends BaseMessage {
  type: MessageType.EVENT_SOURCING_SNAPSHOT | MessageType.EVENT_SOURCING_REPLAY | MessageType.EVENT_SOURCING_PROJECTION;
  command: {
    action: 'create' | 'restore' | 'replay' | 'project';
    streamId?: string;
    fromEventNumber?: number;
    toEventNumber?: number;
    projectionName?: string;
    snapshotFrequency?: number;
  };
}

export interface MessageFlowControl extends BaseMessage {
  type: MessageType.MESSAGE_FLOW_BACKPRESSURE | MessageType.MESSAGE_FLOW_CIRCUIT_BREAKER | MessageType.MESSAGE_FLOW_RATE_LIMIT;
  control: {
    action: 'enable' | 'disable' | 'configure' | 'status';
    targetService?: string;
    configuration?: {
      backpressure?: {
        threshold: number;
        recoveryThreshold: number;
      };
      circuitBreaker?: {
        failureThreshold: number;
        resetTimeoutMs: number;
      };
      rateLimit?: {
        requestsPerSecond: number;
        burstSize: number;
      };
    };
  };
}

export interface MessageAcknowledgment extends BaseMessage {
  type: MessageType.SYSTEM_INFO;
  acknowledgment: {
    originalMessageId: string;
    status: 'received' | 'processed' | 'failed' | 'retrying';
    processingNode: string;
    error?: string;
    retryAttempt?: number;
  };
}

// Enhanced message processing context
export interface MessageProcessingContext {
  message: BaseMessage;
  processingStartTime: number;
  routingHistory: MessageHop[];
  attemptNumber: number;
  maxAttempts: number;
  backoffStrategy: 'linear' | 'exponential' | 'fixed';
  deadline?: number;
  circuitBreakerStatus?: 'closed' | 'open' | 'half-open';
  queueingMetrics: {
    timeInQueue: number;
    queuePosition: number;
    queueDepth: number;
  };
}

// Event sourcing aggregate
export interface AggregateEvent {
  aggregateId: string;
  aggregateType: string;
  eventType: string;
  eventData: any;
  eventNumber: number;
  timestamp: Date;
  causationId?: string;
  correlationId?: string;
  metadata?: Record<string, any>;
}

export interface AggregateSnapshot {
  aggregateId: string;
  aggregateType: string;
  snapshotData: any;
  snapshotVersion: number;
  lastEventNumber: number;
  timestamp: Date;
}

// Message replay functionality
export interface ReplayConfiguration {
  streamId: string;
  fromEventNumber: number;
  toEventNumber?: number;
  replaySpeed: 'realtime' | 'fast' | 'slow';
  filterCriteria?: {
    eventTypes?: string[];
    aggregateIds?: string[];
    timeRange?: {
      start: Date;
      end: Date;
    };
  };
  destination: {
    type: 'broadcast' | 'specific';
    targets?: string[];
  };
}

// Message observability
export interface MessageMetrics {
  messageId: string;
  processingMetrics: {
    totalProcessingTime: number;
    queueingTime: number;
    routingTime: number;
    serializationTime: number;
    networkTime: number;
  };
  flowMetrics: {
    hopCount: number;
    firstHopTime: number;
    lastHopTime: number;
    avgHopTime: number;
  };
  reliabilityMetrics: {
    deliveryAttempts: number;
    successfulDelivery: boolean;
    circuitBreakerTriggered: boolean;
    rateLimitEncountered: boolean;
  };
}

// Enhanced validation with flow-aware rules
export function validateMessageFlow(context: MessageProcessingContext): MessageValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Basic message validation
  const baseValidation = validateMessage(context.message);
  errors.push(...baseValidation.errors);
  warnings.push(...baseValidation.warnings);

  // Flow-specific validation
  if (context.routingHistory.length > 10) {
    warnings.push('Message has been through many routing hops, potential loop detected');
  }

  if (context.deadline && Date.now() > context.deadline) {
    errors.push('Message has exceeded processing deadline');
  }

  if (context.attemptNumber > context.maxAttempts) {
    errors.push('Message has exceeded maximum retry attempts');
  }

  if (context.queueingMetrics.timeInQueue > 30000) {
    warnings.push('Message has been queued for an extended period');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

// Factory for creating flow-aware messages
export class EnhancedMessageFactory extends MessageFactory {
  static createEventSourcingMessage(
    streamId: string,
    eventType: string,
    eventData: any,
    aggregateInfo?: { id: string; type: string }
  ): BaseMessage {
    return {
      id: this.generateId(),
      timestamp: new Date(),
      source: 'event-store',
      type: MessageType.EVENT_SOURCING_PROJECTION,
      priority: MessagePriority.NORMAL,
      eventSourcing: {
        streamId,
        eventId: this.generateId(),
        eventNumber: 0, // Will be set by event store
        eventVersion: 1,
        aggregateId: aggregateInfo?.id,
        aggregateType: aggregateInfo?.type,
        isSnapshot: false,
      },
      metadata: {
        eventType,
        eventData,
      }
    };
  }

  static createFlowControlMessage(
    action: 'enable' | 'disable' | 'configure',
    targetService: string,
    configuration: any
  ): MessageFlowControl {
    return {
      id: this.generateId(),
      timestamp: new Date(),
      source: 'flow-controller',
      type: MessageType.MESSAGE_FLOW_CIRCUIT_BREAKER,
      priority: MessagePriority.HIGH,
      control: {
        action,
        targetService,
        configuration,
      }
    };
  }

  static createAcknowledgmentMessage(
    originalMessageId: string,
    status: 'received' | 'processed' | 'failed',
    processingNode: string,
    error?: string
  ): MessageAcknowledgment {
    return {
      id: this.generateId(),
      timestamp: new Date(),
      source: processingNode,
      type: MessageType.SYSTEM_INFO,
      priority: MessagePriority.LOW,
      acknowledgment: {
        originalMessageId,
        status,
        processingNode,
        error,
      }
    };
  }
}