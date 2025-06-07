/**
 * Comprehensive Connection Pool Manager for Prowzi Platform
 * Optimizes database and external service connections with intelligent pooling,
 * health monitoring, failover mechanisms, and resource optimization
 */

import { EventEmitter } from 'events';
import { createHash } from 'crypto';
import Redis, { Cluster as RedisCluster } from 'ioredis';
import { Pool as PgPool, PoolClient, PoolConfig } from 'pg';
import WebSocket from 'ws';
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { promisify } from 'util';
import { performance } from 'perf_hooks';

export interface ConnectionConfig {
  // Database configurations
  postgres?: {
    host: string;
    port: number;
    database: string;
    user: string;
    password: string;
    ssl?: boolean;
    poolSize: number;
    maxIdleTime: number;
    connectionTimeout: number;
    readReplicas?: string[];
  };
  
  // Redis configurations
  redis?: {
    url: string;
    cluster?: boolean;
    nodes?: Array<{ host: string; port: number }>;
    poolSize: number;
    retryDelayOnFailover: number;
    enableReadyCheck: boolean;
    keyPrefix?: string;
    db?: number;
  };
  
  // HTTP client configurations
  httpClients?: {
    [serviceName: string]: {
      baseURL: string;
      timeout: number;
      retries: number;
      keepAlive: boolean;
      poolSize: number;
      headers?: Record<string, string>;
      auth?: {
        username: string;
        password: string;
      } | {
        token: string;
      };
    };
  };
  
  // WebSocket configurations
  websockets?: {
    [connectionName: string]: {
      url: string;
      protocols?: string[];
      reconnect: boolean;
      maxReconnectAttempts: number;
      reconnectInterval: number;
      pingInterval: number;
      compression: boolean;
    };
  };
  
  // External service configurations
  externalServices?: {
    [serviceName: string]: {
      type: 'rest' | 'graphql' | 'grpc' | 'websocket';
      endpoint: string;
      authentication: any;
      rateLimit?: {
        requestsPerSecond: number;
        burstSize: number;
      };
      circuitBreaker?: {
        failureThreshold: number;
        resetTimeout: number;
        monitoringPeriod: number;
      };
    };
  };
}

export interface ConnectionHealth {
  isHealthy: boolean;
  latency: number;
  lastCheck: number;
  errorCount: number;
  consecutiveFailures: number;
  metrics: {
    totalRequests: number;
    successfulRequests: number;
    averageResponseTime: number;
    throughput: number;
  };
}

export interface PoolStats {
  active: number;
  idle: number;
  waiting: number;
  total: number;
  acquired: number;
  released: number;
  errors: number;
  utilization: number;
}

export class CircuitBreaker extends EventEmitter {
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';
  private failureCount = 0;
  private lastFailureTime = 0;
  private successCount = 0;

  constructor(
    private failureThreshold: number,
    private resetTimeout: number,
    private monitoringPeriod: number = 60000
  ) {
    super();
    this.startMonitoring();
  }

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime >= this.resetTimeout) {
        this.state = 'HALF_OPEN';
        this.emit('half-open');
      } else {
        throw new Error('Circuit breaker is OPEN - operation not allowed');
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failureCount = 0;
    if (this.state === 'HALF_OPEN') {
      this.state = 'CLOSED';
      this.emit('closed');
    }
    this.successCount++;
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    
    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
      this.emit('open');
    }
  }

  private startMonitoring(): void {
    setInterval(() => {
      this.emit('metrics', {
        state: this.state,
        failureCount: this.failureCount,
        successCount: this.successCount,
        lastFailureTime: this.lastFailureTime,
      });
      
      // Reset counters for next period
      this.successCount = 0;
    }, this.monitoringPeriod);
  }

  getState(): string {
    return this.state;
  }
}

export class RateLimiter {
  private tokens: number;
  private lastRefill: number;
  private requestQueue: Array<{
    resolve: () => void;
    reject: (error: Error) => void;
    timestamp: number;
  }> = [];

  constructor(
    private requestsPerSecond: number,
    private burstSize: number
  ) {
    this.tokens = burstSize;
    this.lastRefill = Date.now();
    this.startTokenRefill();
  }

  async acquire(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.tokens > 0) {
        this.tokens--;
        resolve();
      } else {
        this.requestQueue.push({
          resolve,
          reject,
          timestamp: Date.now(),
        });
        
        // Clean up old requests (older than 30 seconds)
        this.requestQueue = this.requestQueue.filter(req => {
          if (Date.now() - req.timestamp > 30000) {
            req.reject(new Error('Rate limit timeout'));
            return false;
          }
          return true;
        });
      }
    });
  }

  private startTokenRefill(): void {
    setInterval(() => {
      const now = Date.now();
      const elapsed = now - this.lastRefill;
      const tokensToAdd = Math.floor((elapsed / 1000) * this.requestsPerSecond);
      
      if (tokensToAdd > 0) {
        this.tokens = Math.min(this.burstSize, this.tokens + tokensToAdd);
        this.lastRefill = now;
        
        // Process queued requests
        while (this.tokens > 0 && this.requestQueue.length > 0) {
          const request = this.requestQueue.shift()!;
          this.tokens--;
          request.resolve();
        }
      }
    }, 100); // Check every 100ms
  }
}

export class ConnectionManager extends EventEmitter {
  private config: ConnectionConfig;
  private connections: Map<string, any> = new Map();
  private healthChecks: Map<string, ConnectionHealth> = new Map();
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private rateLimiters: Map<string, RateLimiter> = new Map();
  private pools: Map<string, PoolStats> = new Map();
  private isShuttingDown = false;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private metricsCollectionInterval: NodeJS.Timeout | null = null;

  // Connection pools
  private pgPools: Map<string, PgPool> = new Map();
  private redisConnections: Map<string, Redis | RedisCluster> = new Map();
  private httpClients: Map<string, AxiosInstance> = new Map();
  private websocketConnections: Map<string, WebSocket> = new Map();

  constructor(config: ConnectionConfig) {
    super();
    this.config = config;
    this.initializeConnections();
    this.startHealthChecking();
    this.startMetricsCollection();
  }

  private async initializeConnections(): Promise<void> {
    try {
      // Initialize PostgreSQL connections
      if (this.config.postgres) {
        await this.initializePostgreSQL();
      }

      // Initialize Redis connections
      if (this.config.redis) {
        await this.initializeRedis();
      }

      // Initialize HTTP clients
      if (this.config.httpClients) {
        await this.initializeHttpClients();
      }

      // Initialize WebSocket connections
      if (this.config.websockets) {
        await this.initializeWebSockets();
      }

      // Initialize external services
      if (this.config.externalServices) {
        await this.initializeExternalServices();
      }

      this.emit('initialized');
    } catch (error) {
      this.emit('error', { type: 'initialization', error });
      throw error;
    }
  }

  private async initializePostgreSQL(): Promise<void> {
    const pgConfig = this.config.postgres!;
    
    // Primary database pool
    const primaryPool = new PgPool({
      host: pgConfig.host,
      port: pgConfig.port,
      database: pgConfig.database,
      user: pgConfig.user,
      password: pgConfig.password,
      ssl: pgConfig.ssl,
      max: pgConfig.poolSize,
      idleTimeoutMillis: pgConfig.maxIdleTime,
      connectionTimeoutMillis: pgConfig.connectionTimeout,
      keepAlive: true,
      keepAliveInitialDelayMillis: 10000,
    });

    // Setup pool event handlers
    primaryPool.on('connect', (client) => {
      this.emit('connection:established', { type: 'postgresql', pool: 'primary' });
    });

    primaryPool.on('error', (error, client) => {
      this.emit('connection:error', { type: 'postgresql', pool: 'primary', error });
    });

    primaryPool.on('remove', (client) => {
      this.emit('connection:removed', { type: 'postgresql', pool: 'primary' });
    });

    this.pgPools.set('primary', primaryPool);
    this.connections.set('postgresql:primary', primaryPool);

    // Read replica pools
    if (pgConfig.readReplicas) {
      for (let i = 0; i < pgConfig.readReplicas.length; i++) {
        const replicaUrl = pgConfig.readReplicas[i];
        const [host, port] = replicaUrl.split(':');
        
        const replicaPool = new PgPool({
          host,
          port: parseInt(port),
          database: pgConfig.database,
          user: pgConfig.user,
          password: pgConfig.password,
          ssl: pgConfig.ssl,
          max: Math.floor(pgConfig.poolSize / 2), // Smaller pool for replicas
          idleTimeoutMillis: pgConfig.maxIdleTime,
          connectionTimeoutMillis: pgConfig.connectionTimeout,
        });

        const replicaKey = `replica-${i}`;
        this.pgPools.set(replicaKey, replicaPool);
        this.connections.set(`postgresql:${replicaKey}`, replicaPool);
      }
    }

    this.initializeHealthCheck('postgresql:primary', () => this.checkPostgreSQLHealth('primary'));
  }

  private async initializeRedis(): Promise<void> {
    const redisConfig = this.config.redis!;

    if (redisConfig.cluster && redisConfig.nodes) {
      // Redis Cluster
      const cluster = new RedisCluster(redisConfig.nodes, {
        enableReadyCheck: redisConfig.enableReadyCheck,
        retryDelayOnFailover: redisConfig.retryDelayOnFailover,
        keyPrefix: redisConfig.keyPrefix,
        scaleReads: 'slave',
        maxRetriesPerRequest: 3,
      });

      cluster.on('connect', () => {
        this.emit('connection:established', { type: 'redis', mode: 'cluster' });
      });

      cluster.on('error', (error) => {
        this.emit('connection:error', { type: 'redis', mode: 'cluster', error });
      });

      this.redisConnections.set('cluster', cluster);
      this.connections.set('redis:cluster', cluster);
    } else {
      // Single Redis instance
      const redis = new Redis(redisConfig.url, {
        retryDelayOnFailover: redisConfig.retryDelayOnFailover,
        enableReadyCheck: redisConfig.enableReadyCheck,
        maxRetriesPerRequest: 3,
        keyPrefix: redisConfig.keyPrefix,
        db: redisConfig.db || 0,
      });

      redis.on('connect', () => {
        this.emit('connection:established', { type: 'redis', mode: 'single' });
      });

      redis.on('error', (error) => {
        this.emit('connection:error', { type: 'redis', mode: 'single', error });
      });

      this.redisConnections.set('primary', redis);
      this.connections.set('redis:primary', redis);
    }

    this.initializeHealthCheck('redis:primary', () => this.checkRedisHealth('primary'));
  }

  private async initializeHttpClients(): Promise<void> {
    const httpConfigs = this.config.httpClients!;

    for (const [serviceName, config] of Object.entries(httpConfigs)) {
      const axiosConfig: AxiosRequestConfig = {
        baseURL: config.baseURL,
        timeout: config.timeout,
        headers: config.headers || {},
      };

      // Add authentication
      if (config.auth) {
        if ('username' in config.auth) {
          axiosConfig.auth = {
            username: config.auth.username,
            password: config.auth.password,
          };
        } else if ('token' in config.auth) {
          axiosConfig.headers!['Authorization'] = `Bearer ${config.auth.token}`;
        }
      }

      // Configure connection pooling
      if (config.keepAlive) {
        const https = require('https');
        const http = require('http');
        
        axiosConfig.httpsAgent = new https.Agent({
          keepAlive: true,
          maxSockets: config.poolSize,
          maxFreeSockets: Math.floor(config.poolSize / 2),
        });
        
        axiosConfig.httpAgent = new http.Agent({
          keepAlive: true,
          maxSockets: config.poolSize,
          maxFreeSockets: Math.floor(config.poolSize / 2),
        });
      }

      const client = axios.create(axiosConfig);

      // Add retry interceptor
      client.interceptors.response.use(
        (response) => response,
        async (error) => {
          const retryCount = error.config.__retryCount || 0;
          
          if (retryCount < config.retries && this.shouldRetry(error)) {
            error.config.__retryCount = retryCount + 1;
            await this.delay(Math.pow(2, retryCount) * 1000); // Exponential backoff
            return client.request(error.config);
          }
          
          return Promise.reject(error);
        }
      );

      this.httpClients.set(serviceName, client);
      this.connections.set(`http:${serviceName}`, client);
      
      this.initializeHealthCheck(`http:${serviceName}`, () => this.checkHttpHealth(serviceName));
    }
  }

  private async initializeWebSockets(): Promise<void> {
    const wsConfigs = this.config.websockets!;

    for (const [connectionName, config] of Object.entries(wsConfigs)) {
      await this.createWebSocketConnection(connectionName, config);
    }
  }

  private async createWebSocketConnection(
    connectionName: string,
    config: ConnectionConfig['websockets'][string]
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(config.url, config.protocols);
      let reconnectAttempts = 0;

      const setupConnection = () => {
        ws.on('open', () => {
          reconnectAttempts = 0;
          this.emit('connection:established', { type: 'websocket', name: connectionName });
          
          // Setup ping-pong for connection health
          if (config.pingInterval) {
            const pingInterval = setInterval(() => {
              if (ws.readyState === WebSocket.OPEN) {
                ws.ping();
              } else {
                clearInterval(pingInterval);
              }
            }, config.pingInterval);
          }
          
          resolve();
        });

        ws.on('error', (error) => {
          this.emit('connection:error', { type: 'websocket', name: connectionName, error });
          reject(error);
        });

        ws.on('close', () => {
          this.emit('connection:closed', { type: 'websocket', name: connectionName });
          
          // Reconnect logic
          if (config.reconnect && reconnectAttempts < config.maxReconnectAttempts && !this.isShuttingDown) {
            reconnectAttempts++;
            setTimeout(() => {
              this.createWebSocketConnection(connectionName, config);
            }, config.reconnectInterval);
          }
        });

        ws.on('pong', () => {
          // Update health metrics
          const health = this.healthChecks.get(`websocket:${connectionName}`);
          if (health) {
            health.lastCheck = Date.now();
            health.latency = Date.now() - health.lastCheck;
          }
        });
      };

      setupConnection();
      this.websocketConnections.set(connectionName, ws);
      this.connections.set(`websocket:${connectionName}`, ws);
      
      this.initializeHealthCheck(`websocket:${connectionName}`, () => this.checkWebSocketHealth(connectionName));
    });
  }

  private async initializeExternalServices(): Promise<void> {
    const serviceConfigs = this.config.externalServices!;

    for (const [serviceName, config] of Object.entries(serviceConfigs)) {
      // Setup circuit breaker
      if (config.circuitBreaker) {
        const circuitBreaker = new CircuitBreaker(
          config.circuitBreaker.failureThreshold,
          config.circuitBreaker.resetTimeout,
          config.circuitBreaker.monitoringPeriod
        );
        
        this.circuitBreakers.set(serviceName, circuitBreaker);
      }

      // Setup rate limiter
      if (config.rateLimit) {
        const rateLimiter = new RateLimiter(
          config.rateLimit.requestsPerSecond,
          config.rateLimit.burstSize
        );
        
        this.rateLimiters.set(serviceName, rateLimiter);
      }

      this.initializeHealthCheck(`service:${serviceName}`, () => this.checkExternalServiceHealth(serviceName));
    }
  }

  private initializeHealthCheck(connectionKey: string, healthCheckFn: () => Promise<boolean>): void {
    this.healthChecks.set(connectionKey, {
      isHealthy: true,
      latency: 0,
      lastCheck: Date.now(),
      errorCount: 0,
      consecutiveFailures: 0,
      metrics: {
        totalRequests: 0,
        successfulRequests: 0,
        averageResponseTime: 0,
        throughput: 0,
      },
    });
  }

  private startHealthChecking(): void {
    this.healthCheckInterval = setInterval(async () => {
      if (this.isShuttingDown) return;

      for (const [connectionKey, health] of this.healthChecks) {
        try {
          const startTime = performance.now();
          const isHealthy = await this.performHealthCheck(connectionKey);
          const latency = performance.now() - startTime;

          health.latency = latency;
          health.lastCheck = Date.now();
          health.metrics.totalRequests++;

          if (isHealthy) {
            health.isHealthy = true;
            health.consecutiveFailures = 0;
            health.metrics.successfulRequests++;
          } else {
            health.isHealthy = false;
            health.consecutiveFailures++;
            health.errorCount++;
          }

          // Update average response time
          health.metrics.averageResponseTime = 
            (health.metrics.averageResponseTime * (health.metrics.totalRequests - 1) + latency) / 
            health.metrics.totalRequests;

          // Emit health status changes
          if (health.consecutiveFailures >= 3) {
            this.emit('connection:unhealthy', { connection: connectionKey, health });
          } else if (health.consecutiveFailures === 0 && health.errorCount > 0) {
            this.emit('connection:recovered', { connection: connectionKey, health });
          }

        } catch (error) {
          health.isHealthy = false;
          health.consecutiveFailures++;
          health.errorCount++;
          this.emit('connection:health-check-failed', { connection: connectionKey, error });
        }
      }
    }, 10000); // Health check every 10 seconds
  }

  private async performHealthCheck(connectionKey: string): Promise<boolean> {
    const [type, identifier] = connectionKey.split(':');

    switch (type) {
      case 'postgresql':
        return this.checkPostgreSQLHealth(identifier);
      case 'redis':
        return this.checkRedisHealth(identifier);
      case 'http':
        return this.checkHttpHealth(identifier);
      case 'websocket':
        return this.checkWebSocketHealth(identifier);
      case 'service':
        return this.checkExternalServiceHealth(identifier);
      default:
        return false;
    }
  }

  private async checkPostgreSQLHealth(poolName: string): Promise<boolean> {
    try {
      const pool = this.pgPools.get(poolName);
      if (!pool) return false;

      const client = await pool.connect();
      const result = await client.query('SELECT 1');
      client.release();
      
      return result.rows.length === 1;
    } catch (error) {
      return false;
    }
  }

  private async checkRedisHealth(connectionName: string): Promise<boolean> {
    try {
      const redis = this.redisConnections.get(connectionName);
      if (!redis) return false;

      const result = await redis.ping();
      return result === 'PONG';
    } catch (error) {
      return false;
    }
  }

  private async checkHttpHealth(serviceName: string): Promise<boolean> {
    try {
      const client = this.httpClients.get(serviceName);
      if (!client) return false;

      const response = await client.get('/health', { timeout: 5000 });
      return response.status >= 200 && response.status < 300;
    } catch (error) {
      return false;
    }
  }

  private async checkWebSocketHealth(connectionName: string): Promise<boolean> {
    try {
      const ws = this.websocketConnections.get(connectionName);
      if (!ws) return false;

      return ws.readyState === WebSocket.OPEN;
    } catch (error) {
      return false;
    }
  }

  private async checkExternalServiceHealth(serviceName: string): Promise<boolean> {
    // Implement service-specific health checks
    const serviceConfig = this.config.externalServices![serviceName];
    
    try {
      // Simple HTTP health check for REST services
      if (serviceConfig.type === 'rest') {
        const response = await axios.get(`${serviceConfig.endpoint}/health`, { timeout: 5000 });
        return response.status >= 200 && response.status < 300;
      }
      
      // Add other service type health checks as needed
      return true;
    } catch (error) {
      return false;
    }
  }

  private startMetricsCollection(): void {
    this.metricsCollectionInterval = setInterval(() => {
      this.collectPoolStats();
      this.emit('metrics:collected', this.getMetrics());
    }, 30000); // Collect metrics every 30 seconds
  }

  private collectPoolStats(): void {
    // PostgreSQL pool stats
    for (const [poolName, pool] of this.pgPools) {
      const stats: PoolStats = {
        active: pool.totalCount - pool.idleCount,
        idle: pool.idleCount,
        waiting: pool.waitingCount,
        total: pool.totalCount,
        acquired: 0, // Not directly available in pg
        released: 0, // Not directly available in pg
        errors: 0, // Would need custom tracking
        utilization: pool.totalCount > 0 ? (pool.totalCount - pool.idleCount) / pool.totalCount : 0,
      };
      
      this.pools.set(`postgresql:${poolName}`, stats);
    }

    // Add Redis and other connection pool stats as needed
  }

  // Public API methods

  /**
   * Get a PostgreSQL connection from the pool
   */
  async getPostgreSQLConnection(poolName: string = 'primary'): Promise<PoolClient> {
    const pool = this.pgPools.get(poolName);
    if (!pool) {
      throw new Error(`PostgreSQL pool '${poolName}' not found`);
    }

    const health = this.healthChecks.get(`postgresql:${poolName}`);
    if (health && !health.isHealthy) {
      throw new Error(`PostgreSQL pool '${poolName}' is unhealthy`);
    }

    return pool.connect();
  }

  /**
   * Get a Redis connection
   */
  getRedisConnection(connectionName: string = 'primary'): Redis | RedisCluster {
    const redis = this.redisConnections.get(connectionName);
    if (!redis) {
      throw new Error(`Redis connection '${connectionName}' not found`);
    }

    const health = this.healthChecks.get(`redis:${connectionName}`);
    if (health && !health.isHealthy) {
      throw new Error(`Redis connection '${connectionName}' is unhealthy`);
    }

    return redis;
  }

  /**
   * Get an HTTP client
   */
  getHttpClient(serviceName: string): AxiosInstance {
    const client = this.httpClients.get(serviceName);
    if (!client) {
      throw new Error(`HTTP client '${serviceName}' not found`);
    }

    const health = this.healthChecks.get(`http:${serviceName}`);
    if (health && !health.isHealthy) {
      throw new Error(`HTTP service '${serviceName}' is unhealthy`);
    }

    return client;
  }

  /**
   * Get a WebSocket connection
   */
  getWebSocketConnection(connectionName: string): WebSocket {
    const ws = this.websocketConnections.get(connectionName);
    if (!ws) {
      throw new Error(`WebSocket connection '${connectionName}' not found`);
    }

    const health = this.healthChecks.get(`websocket:${connectionName}`);
    if (health && !health.isHealthy) {
      throw new Error(`WebSocket connection '${connectionName}' is unhealthy`);
    }

    return ws;
  }

  /**
   * Execute operation with circuit breaker protection
   */
  async executeWithCircuitBreaker<T>(
    serviceName: string,
    operation: () => Promise<T>
  ): Promise<T> {
    const circuitBreaker = this.circuitBreakers.get(serviceName);
    if (!circuitBreaker) {
      return operation();
    }

    return circuitBreaker.execute(operation);
  }

  /**
   * Execute operation with rate limiting
   */
  async executeWithRateLimit<T>(
    serviceName: string,
    operation: () => Promise<T>
  ): Promise<T> {
    const rateLimiter = this.rateLimiters.get(serviceName);
    if (rateLimiter) {
      await rateLimiter.acquire();
    }

    return operation();
  }

  /**
   * Get connection health status
   */
  getConnectionHealth(connectionKey?: string): ConnectionHealth | Map<string, ConnectionHealth> {
    if (connectionKey) {
      const health = this.healthChecks.get(connectionKey);
      if (!health) {
        throw new Error(`Connection '${connectionKey}' not found`);
      }
      return health;
    }
    
    return new Map(this.healthChecks);
  }

  /**
   * Get pool statistics
   */
  getPoolStats(poolKey?: string): PoolStats | Map<string, PoolStats> {
    if (poolKey) {
      const stats = this.pools.get(poolKey);
      if (!stats) {
        throw new Error(`Pool '${poolKey}' not found`);
      }
      return stats;
    }
    
    return new Map(this.pools);
  }

  /**
   * Get comprehensive metrics
   */
  getMetrics(): {
    connections: Map<string, ConnectionHealth>;
    pools: Map<string, PoolStats>;
    circuitBreakers: Map<string, string>;
    uptime: number;
  } {
    const circuitBreakerStates = new Map<string, string>();
    for (const [serviceName, breaker] of this.circuitBreakers) {
      circuitBreakerStates.set(serviceName, breaker.getState());
    }

    return {
      connections: new Map(this.healthChecks),
      pools: new Map(this.pools),
      circuitBreakers: circuitBreakerStates,
      uptime: process.uptime(),
    };
  }

  /**
   * Graceful shutdown
   */
  async shutdown(): Promise<void> {
    this.isShuttingDown = true;
    
    // Clear intervals
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    if (this.metricsCollectionInterval) {
      clearInterval(this.metricsCollectionInterval);
    }

    const shutdownPromises: Promise<void>[] = [];

    // Close PostgreSQL pools
    for (const [poolName, pool] of this.pgPools) {
      shutdownPromises.push(pool.end());
    }

    // Close Redis connections
    for (const [connectionName, redis] of this.redisConnections) {
      shutdownPromises.push(redis.disconnect());
    }

    // Close WebSocket connections
    for (const [connectionName, ws] of this.websocketConnections) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }

    await Promise.all(shutdownPromises);
    this.emit('shutdown:complete');
  }

  // Helper methods

  private shouldRetry(error: any): boolean {
    if (!error.response) return true; // Network error
    if (error.response.status >= 500) return true; // Server error
    if (error.response.status === 429) return true; // Rate limited
    return false;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Connection manager factory with environment-based configuration
 */
export class ConnectionManagerFactory {
  static create(overrides?: Partial<ConnectionConfig>): ConnectionManager {
    const config: ConnectionConfig = {
      postgres: {
        host: process.env.POSTGRES_HOST || 'localhost',
        port: parseInt(process.env.POSTGRES_PORT || '5432'),
        database: process.env.POSTGRES_DB || 'prowzi',
        user: process.env.POSTGRES_USER || 'postgres',
        password: process.env.POSTGRES_PASSWORD || '',
        ssl: process.env.POSTGRES_SSL === 'true',
        poolSize: parseInt(process.env.POSTGRES_POOL_SIZE || '20'),
        maxIdleTime: parseInt(process.env.POSTGRES_MAX_IDLE_TIME || '30000'),
        connectionTimeout: parseInt(process.env.POSTGRES_CONNECTION_TIMEOUT || '10000'),
        readReplicas: process.env.POSTGRES_READ_REPLICAS?.split(','),
      },
      
      redis: {
        url: process.env.REDIS_URL || 'redis://localhost:6379',
        cluster: process.env.REDIS_CLUSTER === 'true',
        poolSize: parseInt(process.env.REDIS_POOL_SIZE || '10'),
        retryDelayOnFailover: parseInt(process.env.REDIS_RETRY_DELAY || '100'),
        enableReadyCheck: process.env.REDIS_READY_CHECK !== 'false',
        keyPrefix: process.env.REDIS_KEY_PREFIX || 'prowzi:',
        db: parseInt(process.env.REDIS_DB || '0'),
      },
      
      ...overrides,
    };

    return new ConnectionManager(config);
  }
}

/**
 * Singleton connection manager instance
 */
let connectionManagerInstance: ConnectionManager | null = null;

export function getConnectionManager(config?: ConnectionConfig): ConnectionManager {
  if (!connectionManagerInstance) {
    connectionManagerInstance = config 
      ? new ConnectionManager(config)
      : ConnectionManagerFactory.create();
  }
  
  return connectionManagerInstance;
}

export function resetConnectionManager(): void {
  if (connectionManagerInstance) {
    connectionManagerInstance.shutdown();
    connectionManagerInstance = null;
  }
}