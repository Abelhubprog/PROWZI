import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { Agent } from '../../../agent-runtime/core/src/agent';
import { AgentConfig, AgentStatus, TaskResult } from '../../../agent-runtime/core/src/types';

describe('Test Agent for Integration Testing', () => {
  let testAgent: TestAgent;
  let mockConfig: AgentConfig;

  beforeEach(() => {
    mockConfig = {
      id: 'test-agent-001',
      name: 'Test Agent',
      type: 'test',
      version: '1.0.0',
      capabilities: ['test-execution', 'mock-data-generation'],
      config: {
        testDuration: 5000,
        mockDataSize: 100,
        failureRate: 0.1,
      },
      resources: {
        maxMemory: '512Mi',
        maxCpu: '0.5',
      },
    };

    testAgent = new TestAgent(mockConfig);
  });

  afterEach(async () => {
    if (testAgent) {
      await testAgent.stop();
    }
  });

  describe('Agent Lifecycle', () => {
    it('should initialize properly', async () => {
      await testAgent.initialize();
      
      expect(testAgent.getStatus()).toBe(AgentStatus.READY);
      expect(testAgent.getId()).toBe('test-agent-001');
      expect(testAgent.getName()).toBe('Test Agent');
    });

    it('should start and stop correctly', async () => {
      await testAgent.initialize();
      await testAgent.start();
      
      expect(testAgent.getStatus()).toBe(AgentStatus.RUNNING);
      
      await testAgent.stop();
      expect(testAgent.getStatus()).toBe(AgentStatus.STOPPED);
    });

    it('should handle pause and resume', async () => {
      await testAgent.initialize();
      await testAgent.start();
      
      await testAgent.pause();
      expect(testAgent.getStatus()).toBe(AgentStatus.PAUSED);
      
      await testAgent.resume();
      expect(testAgent.getStatus()).toBe(AgentStatus.RUNNING);
    });
  });

  describe('Task Execution', () => {
    beforeEach(async () => {
      await testAgent.initialize();
      await testAgent.start();
    });

    it('should execute simple tasks', async () => {
      const task = {
        id: 'task-001',
        type: 'simple-test',
        payload: { testValue: 42 },
        priority: 1,
      };

      const result = await testAgent.executeTask(task);
      
      expect(result.success).toBe(true);
      expect(result.taskId).toBe('task-001');
      expect(result.executionTime).toBeGreaterThan(0);
    });

    it('should handle task failures gracefully', async () => {
      const task = {
        id: 'task-002',
        type: 'failing-test',
        payload: { shouldFail: true },
        priority: 1,
      };

      const result = await testAgent.executeTask(task);
      
      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.taskId).toBe('task-002');
    });

    it('should handle concurrent tasks', async () => {
      const tasks = Array.from({ length: 5 }, (_, i) => ({
        id: `task-${i + 3}`,
        type: 'concurrent-test',
        payload: { index: i },
        priority: 1,
      }));

      const results = await Promise.all(
        tasks.map(task => testAgent.executeTask(task))
      );

      expect(results).toHaveLength(5);
      results.forEach((result, index) => {
        expect(result.taskId).toBe(`task-${index + 3}`);
        expect(result.success).toBe(true);
      });
    });
  });

  describe('Performance Monitoring', () => {
    beforeEach(async () => {
      await testAgent.initialize();
      await testAgent.start();
    });

    it('should track execution metrics', async () => {
      const task = {
        id: 'perf-task-001',
        type: 'performance-test',
        payload: { iterations: 10 },
        priority: 1,
      };

      await testAgent.executeTask(task);
      
      const metrics = testAgent.getMetrics();
      expect(metrics.tasksExecuted).toBeGreaterThan(0);
      expect(metrics.averageExecutionTime).toBeGreaterThan(0);
      expect(metrics.memoryUsage).toBeGreaterThan(0);
    });

    it('should report system resource usage', async () => {
      const metrics = testAgent.getSystemMetrics();
      
      expect(metrics.cpuUsage).toBeGreaterThanOrEqual(0);
      expect(metrics.memoryUsage).toBeGreaterThan(0);
      expect(metrics.uptime).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle initialization errors', async () => {
      const invalidConfig = { ...mockConfig, id: '' };
      const invalidAgent = new TestAgent(invalidConfig);
      
      await expect(invalidAgent.initialize()).rejects.toThrow('Invalid agent ID');
    });

    it('should handle resource exhaustion', async () => {
      await testAgent.initialize();
      await testAgent.start();
      
      // Simulate resource exhaustion
      const heavyTask = {
        id: 'heavy-task',
        type: 'resource-intensive',
        payload: { memorySize: '2Gi' },
        priority: 1,
      };

      const result = await testAgent.executeTask(heavyTask);
      expect(result.success).toBe(false);
      expect(result.error).toContain('Resource limit exceeded');
    });
  });
});

class TestAgent extends Agent {
  private taskCounter = 0;
  private executionMetrics = {
    tasksExecuted: 0,
    totalExecutionTime: 0,
    successfulTasks: 0,
    failedTasks: 0,
  };

  constructor(config: AgentConfig) {
    super(config);
  }

  async initialize(): Promise<void> {
    if (!this.config.id) {
      throw new Error('Invalid agent ID');
    }
    
    this.logger.info('Initializing test agent', { agentId: this.config.id });
    this.status = AgentStatus.READY;
  }

  async start(): Promise<void> {
    this.logger.info('Starting test agent');
    this.status = AgentStatus.RUNNING;
    
    // Start background health monitoring
    this.startHealthMonitoring();
  }

  async stop(): Promise<void> {
    this.logger.info('Stopping test agent');
    this.status = AgentStatus.STOPPED;
    
    // Clean up resources
    this.stopHealthMonitoring();
  }

  async pause(): Promise<void> {
    this.logger.info('Pausing test agent');
    this.status = AgentStatus.PAUSED;
  }

  async resume(): Promise<void> {
    this.logger.info('Resuming test agent');
    this.status = AgentStatus.RUNNING;
  }

  async executeTask(task: any): Promise<TaskResult> {
    const startTime = Date.now();
    this.taskCounter++;
    
    try {
      this.logger.debug('Executing task', { taskId: task.id, type: task.type });
      
      // Simulate task execution based on type
      const result = await this.simulateTaskExecution(task);
      
      const executionTime = Date.now() - startTime;
      this.executionMetrics.tasksExecuted++;
      this.executionMetrics.totalExecutionTime += executionTime;
      this.executionMetrics.successfulTasks++;
      
      return {
        success: true,
        taskId: task.id,
        result: result,
        executionTime,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const executionTime = Date.now() - startTime;
      this.executionMetrics.tasksExecuted++;
      this.executionMetrics.totalExecutionTime += executionTime;
      this.executionMetrics.failedTasks++;
      
      return {
        success: false,
        taskId: task.id,
        error: error.message,
        executionTime,
        timestamp: new Date().toISOString(),
      };
    }
  }

  private async simulateTaskExecution(task: any): Promise<any> {
    // Simulate different task types
    switch (task.type) {
      case 'simple-test':
        await this.delay(100);
        return { processed: task.payload.testValue * 2 };
      
      case 'failing-test':
        if (task.payload.shouldFail) {
          throw new Error('Simulated task failure');
        }
        return { success: true };
      
      case 'concurrent-test':
        await this.delay(50 + Math.random() * 100);
        return { index: task.payload.index, processed: true };
      
      case 'performance-test':
        const iterations = task.payload.iterations || 1;
        for (let i = 0; i < iterations; i++) {
          await this.delay(10);
        }
        return { iterations, completed: true };
      
      case 'resource-intensive':
        // Simulate resource check
        if (task.payload.memorySize === '2Gi') {
          throw new Error('Resource limit exceeded: insufficient memory');
        }
        return { resourcesAllocated: task.payload.memorySize };
      
      default:
        throw new Error(`Unknown task type: ${task.type}`);
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  getMetrics() {
    return {
      ...this.executionMetrics,
      averageExecutionTime: this.executionMetrics.totalExecutionTime / this.executionMetrics.tasksExecuted || 0,
      successRate: this.executionMetrics.successfulTasks / this.executionMetrics.tasksExecuted || 0,
      memoryUsage: process.memoryUsage().heapUsed,
    };
  }

  getSystemMetrics() {
    const memUsage = process.memoryUsage();
    return {
      cpuUsage: process.cpuUsage().user / 1000000, // Convert to seconds
      memoryUsage: memUsage.heapUsed,
      memoryTotal: memUsage.heapTotal,
      uptime: process.uptime(),
      pid: process.pid,
    };
  }

  private startHealthMonitoring() {
    // Implement health monitoring logic
    this.logger.debug('Health monitoring started');
  }

  private stopHealthMonitoring() {
    // Implement cleanup logic
    this.logger.debug('Health monitoring stopped');
  }
}