/**
 * Enhanced message bus system for inter-component communication
 * Provides reliable, scalable messaging between Prowzi platform services
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import WebSocket from 'ws';
import Redis from 'ioredis';

export interface Message {
  id: string;
  type: string;
  source: string;
  destination?: string;
  payload: any;
  timestamp: number;
  priority: MessagePriority;
  replyTo?: string;
  correlationId?: string;
  metadata?: Record<string, any>;
}

export enum MessagePriority {
  CRITICAL = 0,
  HIGH = 1,
  MEDIUM = 2,
  LOW = 3,
  BACKGROUND = 4,
}

export interface MessageHandler {
  (message: Message): Promise<any>;
}

export interface MessageBusConfig {
  redisUrl: string;
  nodeId: string;
  maxRetries: number;
  retryDelay: number;
  deadLetterQueue: boolean;
  compression: boolean;
}

export interface Subscription {
  id: string;
  pattern: string;
  handler: MessageHandler;
  options: SubscriptionOptions;
}

export interface SubscriptionOptions {
  priority?: MessagePriority;
  timeout?: number;
  retries?: number;
  durableQueue?: boolean;
  exclusive?: boolean;
}

export interface MessageResponse {
  success: boolean;
  data?: any;
  error?: string;
  responseTime: number;
}

/**
 * Enhanced message bus for distributed communication
 */
export class MessageBus extends EventEmitter {
  private redis: Redis;
  private config: MessageBusConfig;
  private subscriptions: Map<string, Subscription> = new Map();
  private pendingRequests: Map<string, {
    resolve: (value: any) => void;
    reject: (error: any) => void;
    timeout: NodeJS.Timeout;
  }> = new Map();
  private connectionState: 'connecting' | 'connected' | 'disconnected' = 'disconnected';
  private messageQueue: Message[] = [];
  private wsConnections: Map<string, WebSocket> = new Map();

  constructor(config: MessageBusConfig) {
    super();
    this.config = config;
    this.redis = new Redis(config.redisUrl, {
      retryDelayOnFailover: 100,
      enableReadyCheck: false,
      maxRetriesPerRequest: 3,
    });
    
    this.setupRedisEventHandlers();
  }

  private setupRedisEventHandlers(): void {
    this.redis.on('connect', () => {
      console.log('MessageBus connected to Redis');
      this.connectionState = 'connected';
      this.emit('connected');
      this.processQueuedMessages();
    });

    this.redis.on('error', (error) => {
      console.error('MessageBus Redis error:', error);
      this.connectionState = 'disconnected';
      this.emit('error', error);
    });

    this.redis.on('message', this.handleIncomingMessage.bind(this));
    this.redis.on('pmessage', this.handlePatternMessage.bind(this));
  }

  /**
   * Start the message bus
   */
  async start(): Promise<void> {
    this.connectionState = 'connecting';
    
    // Subscribe to node-specific channel
    await this.redis.subscribe(`prowzi:node:${this.config.nodeId}`);
    
    // Subscribe to broadcast channel
    await this.redis.subscribe('prowzi:broadcast');
    
    // Start health check
    this.startHealthCheck();
    
    console.log(`MessageBus started for node: ${this.config.nodeId}`);
  }

  /**
   * Stop the message bus
   */
  async stop(): Promise<void> {
    this.connectionState = 'disconnected';
    
    // Close all WebSocket connections
    for (const [id, ws] of this.wsConnections) {
      ws.close();
    }
    this.wsConnections.clear();
    
    // Clear pending requests
    for (const [id, request] of this.pendingRequests) {
      clearTimeout(request.timeout);
      request.reject(new Error('MessageBus stopped'));
    }
    this.pendingRequests.clear();
    
    await this.redis.disconnect();
    console.log('MessageBus stopped');
  }

  /**
   * Publish a message
   */
  async publish(
    type: string,
    payload: any,
    destination?: string,
    priority: MessagePriority = MessagePriority.MEDIUM,
    metadata?: Record<string, any>
  ): Promise<void> {
    const message: Message = {
      id: uuidv4(),
      type,
      source: this.config.nodeId,
      destination,
      payload,
      timestamp: Date.now(),
      priority,
      metadata,
    };

    await this.sendMessage(message);
  }

  /**
   * Send a request and wait for response
   */
  async request(
    type: string,
    payload: any,
    destination?: string,
    timeout: number = 30000
  ): Promise<any> {
    const message: Message = {
      id: uuidv4(),
      type,
      source: this.config.nodeId,
      destination,
      payload,
      timestamp: Date.now(),
      priority: MessagePriority.HIGH,
      replyTo: `prowzi:node:${this.config.nodeId}`,
      correlationId: uuidv4(),
    };

    return new Promise((resolve, reject) => {
      const timeoutHandle = setTimeout(() => {
        this.pendingRequests.delete(message.correlationId!);
        reject(new Error(`Request timeout after ${timeout}ms`));
      }, timeout);

      this.pendingRequests.set(message.correlationId!, {
        resolve,
        reject,
        timeout: timeoutHandle,
      });

      this.sendMessage(message).catch(reject);
    });
  }

  /**
   * Subscribe to message patterns
   */
  subscribe(
    pattern: string,
    handler: MessageHandler,
    options: SubscriptionOptions = {}
  ): string {
    const subscription: Subscription = {
      id: uuidv4(),
      pattern,
      handler,
      options,
    };

    this.subscriptions.set(subscription.id, subscription);
    
    // Subscribe to Redis pattern
    this.redis.psubscribe(`prowzi:${pattern}`);
    
    return subscription.id;
  }

  /**
   * Unsubscribe from messages
   */
  unsubscribe(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId);
    if (subscription) {
      this.subscriptions.delete(subscriptionId);
      this.redis.punsubscribe(`prowzi:${subscription.pattern}`);
    }
  }

  /**
   * Send a response to a request
   */
  async respond(
    originalMessage: Message,
    responsePayload: any,
    success: boolean = true
  ): Promise<void> {
    if (!originalMessage.replyTo || !originalMessage.correlationId) {
      throw new Error('Cannot respond to message without replyTo or correlationId');
    }

    const response: Message = {
      id: uuidv4(),
      type: `${originalMessage.type}:response`,
      source: this.config.nodeId,
      destination: originalMessage.source,
      payload: {
        success,
        data: responsePayload,
        correlationId: originalMessage.correlationId,
        responseTime: Date.now() - originalMessage.timestamp,
      },
      timestamp: Date.now(),
      priority: originalMessage.priority,
      correlationId: originalMessage.correlationId,
    };

    await this.sendDirectMessage(originalMessage.replyTo, response);
  }

  /**
   * Create a WebSocket connection for real-time communication
   */
  createWebSocketConnection(id: string, url: string): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      
      ws.on('open', () => {
        this.wsConnections.set(id, ws);
        console.log(`WebSocket connection established: ${id}`);
        
        // Setup message forwarding
        ws.on('message', (data) => {
          try {
            const message = JSON.parse(data.toString()) as Message;
            this.handleWebSocketMessage(id, message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        });
        
        resolve(ws);
      });
      
      ws.on('error', (error) => {
        console.error(`WebSocket error for ${id}:`, error);
        this.wsConnections.delete(id);
        reject(error);
      });
      
      ws.on('close', () => {
        console.log(`WebSocket connection closed: ${id}`);
        this.wsConnections.delete(id);
      });
    });
  }

  /**
   * Broadcast message to all WebSocket connections
   */
  broadcastToWebSockets(message: Message): void {
    const data = JSON.stringify(message);
    
    for (const [id, ws] of this.wsConnections) {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(data);
        } catch (error) {
          console.error(`Failed to send to WebSocket ${id}:`, error);
        }
      }
    }
  }

  private async sendMessage(message: Message): Promise<void> {
    if (this.connectionState !== 'connected') {
      this.messageQueue.push(message);
      return;
    }

    try {
      const channel = message.destination 
        ? `prowzi:node:${message.destination}`
        : 'prowzi:broadcast';
        
      const serialized = this.serializeMessage(message);
      await this.redis.publish(channel, serialized);
      
      // Also broadcast to WebSocket connections if appropriate
      if (!message.destination || message.type.startsWith('realtime:')) {
        this.broadcastToWebSockets(message);
      }
      
    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Retry logic
      if (this.config.maxRetries > 0) {
        setTimeout(() => {
          this.sendMessage(message);
        }, this.config.retryDelay);
      }
    }
  }

  private async sendDirectMessage(channel: string, message: Message): Promise<void> {
    const serialized = this.serializeMessage(message);
    await this.redis.publish(channel, serialized);
  }

  private async handleIncomingMessage(channel: string, data: string): Promise<void> {
    try {
      const message = this.deserializeMessage(data);
      
      // Handle responses to pending requests
      if (message.correlationId && this.pendingRequests.has(message.correlationId)) {
        const request = this.pendingRequests.get(message.correlationId)!;
        this.pendingRequests.delete(message.correlationId);
        clearTimeout(request.timeout);
        
        if (message.payload.success) {
          request.resolve(message.payload.data);
        } else {
          request.reject(new Error(message.payload.error || 'Request failed'));
        }
        return;
      }
      
      // Route to subscribers
      await this.routeMessage(message);
      
    } catch (error) {
      console.error('Failed to handle incoming message:', error);
    }
  }

  private async handlePatternMessage(
    pattern: string,
    channel: string,
    data: string
  ): Promise<void> {
    try {
      const message = this.deserializeMessage(data);
      await this.routeMessage(message);
    } catch (error) {
      console.error('Failed to handle pattern message:', error);
    }
  }

  private async handleWebSocketMessage(connectionId: string, message: Message): Promise<void> {
    // Add connection context to message
    message.metadata = {
      ...message.metadata,
      webSocketConnection: connectionId,
    };
    
    await this.routeMessage(message);
  }

  private async routeMessage(message: Message): Promise<void> {
    const matchingSubscriptions = Array.from(this.subscriptions.values())
      .filter(sub => this.matchesPattern(message.type, sub.pattern))
      .sort((a, b) => (a.options.priority || MessagePriority.MEDIUM) - (b.options.priority || MessagePriority.MEDIUM));

    for (const subscription of matchingSubscriptions) {
      try {
        const result = await this.executeHandler(subscription, message);
        
        // Send response if this was a request
        if (message.replyTo && message.correlationId) {
          await this.respond(message, result, true);
        }
        
      } catch (error) {
        console.error(`Handler failed for subscription ${subscription.id}:`, error);
        
        // Send error response if this was a request
        if (message.replyTo && message.correlationId) {
          await this.respond(message, error.message, false);
        }
      }
    }
  }

  private async executeHandler(
    subscription: Subscription,
    message: Message
  ): Promise<any> {
    const timeout = subscription.options.timeout || 30000;
    
    return new Promise((resolve, reject) => {
      const timeoutHandle = setTimeout(() => {
        reject(new Error(`Handler timeout after ${timeout}ms`));
      }, timeout);
      
      subscription.handler(message)
        .then(result => {
          clearTimeout(timeoutHandle);
          resolve(result);
        })
        .catch(error => {
          clearTimeout(timeoutHandle);
          reject(error);
        });
    });
  }

  private matchesPattern(messageType: string, pattern: string): boolean {
    // Convert glob pattern to regex
    const regexPattern = pattern
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.');
    
    const regex = new RegExp(`^${regexPattern}$`);
    return regex.test(messageType);
  }

  private serializeMessage(message: Message): string {
    return JSON.stringify(message);
  }

  private deserializeMessage(data: string): Message {
    return JSON.parse(data);
  }

  private async processQueuedMessages(): Promise<void> {
    const queue = [...this.messageQueue];
    this.messageQueue = [];
    
    for (const message of queue) {
      await this.sendMessage(message);
    }
  }

  private startHealthCheck(): void {
    setInterval(async () => {
      try {
        await this.redis.ping();
      } catch (error) {
        console.error('MessageBus health check failed:', error);
        this.connectionState = 'disconnected';
        this.emit('error', error);
      }
    }, 30000); // 30 second health check
  }

  /**
   * Get message bus statistics
   */
  getStats(): {
    connectionState: string;
    subscriptions: number;
    pendingRequests: number;
    queuedMessages: number;
    webSocketConnections: number;
  } {
    return {
      connectionState: this.connectionState,
      subscriptions: this.subscriptions.size,
      pendingRequests: this.pendingRequests.size,
      queuedMessages: this.messageQueue.length,
      webSocketConnections: this.wsConnections.size,
    };
  }
}

/**
 * Message bus factory with default configuration
 */
export class MessageBusFactory {
  static create(nodeId: string, redisUrl?: string): MessageBus {
    const config: MessageBusConfig = {
      redisUrl: redisUrl || process.env.REDIS_URL || 'redis://localhost:6379',
      nodeId,
      maxRetries: 3,
      retryDelay: 1000,
      deadLetterQueue: true,
      compression: false,
    };
    
    return new MessageBus(config);
  }
}

/**
 * Typed message helpers for specific message types
 */
export namespace Messages {
  export interface AgentCoordination {
    agentId: string;
    action: 'spawn' | 'stop' | 'pause' | 'resume';
    parameters?: any;
  }

  export interface TradingSignal {
    symbol: string;
    signal: 'buy' | 'sell' | 'hold';
    confidence: number;
    price: number;
    timestamp: number;
    metadata: any;
  }

  export interface RiskAlert {
    level: 'low' | 'medium' | 'high' | 'critical';
    type: string;
    description: string;
    affectedAssets: string[];
    recommendedActions: string[];
  }

  export interface SystemMetrics {
    nodeId: string;
    cpu: number;
    memory: number;
    activeAgents: number;
    timestamp: number;
  }
}