import { EventEmitter } from 'events'
import { performance } from 'perf_hooks'
import { promisify } from 'util'
import { writeFile } from 'fs'

const writeFileAsync = promisify(writeFile)

export interface PerformanceMetrics {
  timestamp: number
  testName: string
  duration: number
  memory: {
    used: number
    total: number
    external: number
    heapUsed: number
    heapTotal: number
  }
  cpu: {
    userTime: number
    systemTime: number
  }
  network?: {
    bytesReceived: number
    bytesSent: number
    requests: number
    errors: number
  }
  database?: {
    queries: number
    duration: number
    connections: number
  }
}

export interface PerformanceThresholds {
  maxDuration: number
  maxMemoryUsage: number
  maxCpuUsage: number
  maxResponseTime: number
  minThroughput: number
}

export class PerformanceMonitor extends EventEmitter {
  private metrics: PerformanceMetrics[] = []
  private startTimes: Map<string, number> = new Map()
  private thresholds: PerformanceThresholds
  private isRecording = false

  constructor(thresholds: Partial<PerformanceThresholds> = {}) {
    super()
    this.thresholds = {
      maxDuration: 30000, // 30 seconds
      maxMemoryUsage: 500 * 1024 * 1024, // 500MB
      maxCpuUsage: 80, // 80%
      maxResponseTime: 2000, // 2 seconds
      minThroughput: 100, // 100 req/sec
      ...thresholds
    }
  }

  startRecording(): void {
    this.isRecording = true
    this.emit('recording-started')
  }

  stopRecording(): void {
    this.isRecording = false
    this.emit('recording-stopped')
  }

  startTest(testName: string): void {
    if (!this.isRecording) return
    
    this.startTimes.set(testName, performance.now())
    this.emit('test-started', testName)
  }

  endTest(testName: string, additionalData?: Partial<PerformanceMetrics>): PerformanceMetrics {
    if (!this.isRecording) {
      throw new Error('Performance monitoring is not active')
    }

    const startTime = this.startTimes.get(testName)
    if (!startTime) {
      throw new Error(`No start time found for test: ${testName}`)
    }

    const duration = performance.now() - startTime
    const memoryUsage = process.memoryUsage()
    const cpuUsage = process.cpuUsage()

    const metrics: PerformanceMetrics = {
      timestamp: Date.now(),
      testName,
      duration,
      memory: {
        used: memoryUsage.rss,
        total: memoryUsage.rss + memoryUsage.external,
        external: memoryUsage.external,
        heapUsed: memoryUsage.heapUsed,
        heapTotal: memoryUsage.heapTotal
      },
      cpu: {
        userTime: cpuUsage.user / 1000, // Convert to milliseconds
        systemTime: cpuUsage.system / 1000
      },
      ...additionalData
    }

    this.metrics.push(metrics)
    this.startTimes.delete(testName)

    // Check thresholds and emit warnings
    this.checkThresholds(metrics)

    this.emit('test-completed', metrics)
    return metrics
  }

  private checkThresholds(metrics: PerformanceMetrics): void {
    const warnings: string[] = []

    if (metrics.duration > this.thresholds.maxDuration) {
      warnings.push(`Test duration (${metrics.duration}ms) exceeded threshold (${this.thresholds.maxDuration}ms)`)
    }

    if (metrics.memory.used > this.thresholds.maxMemoryUsage) {
      warnings.push(`Memory usage (${this.formatBytes(metrics.memory.used)}) exceeded threshold (${this.formatBytes(this.thresholds.maxMemoryUsage)})`)
    }

    if (warnings.length > 0) {
      this.emit('threshold-exceeded', {
        testName: metrics.testName,
        warnings,
        metrics
      })
    }
  }

  getMetrics(): PerformanceMetrics[] {
    return [...this.metrics]
  }

  getMetricsForTest(testName: string): PerformanceMetrics[] {
    return this.metrics.filter(m => m.testName === testName)
  }

  getAverageMetrics(testName?: string): Partial<PerformanceMetrics> {
    const relevantMetrics = testName 
      ? this.getMetricsForTest(testName)
      : this.metrics

    if (relevantMetrics.length === 0) {
      return {}
    }

    const avg = {
      duration: 0,
      memory: { used: 0, total: 0, external: 0, heapUsed: 0, heapTotal: 0 },
      cpu: { userTime: 0, systemTime: 0 }
    }

    relevantMetrics.forEach(metric => {
      avg.duration += metric.duration
      avg.memory.used += metric.memory.used
      avg.memory.total += metric.memory.total
      avg.memory.external += metric.memory.external
      avg.memory.heapUsed += metric.memory.heapUsed
      avg.memory.heapTotal += metric.memory.heapTotal
      avg.cpu.userTime += metric.cpu.userTime
      avg.cpu.systemTime += metric.cpu.systemTime
    })

    const count = relevantMetrics.length
    return {
      duration: avg.duration / count,
      memory: {
        used: avg.memory.used / count,
        total: avg.memory.total / count,
        external: avg.memory.external / count,
        heapUsed: avg.memory.heapUsed / count,
        heapTotal: avg.memory.heapTotal / count
      },
      cpu: {
        userTime: avg.cpu.userTime / count,
        systemTime: avg.cpu.systemTime / count
      }
    }
  }

  generateReport(): string {
    const report = {
      summary: {
        totalTests: this.metrics.length,
        averageMetrics: this.getAverageMetrics(),
        thresholds: this.thresholds
      },
      tests: this.metrics.map(metric => ({
        testName: metric.testName,
        timestamp: new Date(metric.timestamp).toISOString(),
        duration: `${metric.duration.toFixed(2)}ms`,
        memory: {
          used: this.formatBytes(metric.memory.used),
          heap: this.formatBytes(metric.memory.heapUsed)
        },
        cpu: {
          total: `${(metric.cpu.userTime + metric.cpu.systemTime).toFixed(2)}ms`
        }
      })),
      violations: this.getThresholdViolations()
    }

    return JSON.stringify(report, null, 2)
  }

  private getThresholdViolations(): any[] {
    const violations: any[] = []

    this.metrics.forEach(metric => {
      if (metric.duration > this.thresholds.maxDuration) {
        violations.push({
          testName: metric.testName,
          type: 'duration',
          value: metric.duration,
          threshold: this.thresholds.maxDuration,
          severity: 'high'
        })
      }

      if (metric.memory.used > this.thresholds.maxMemoryUsage) {
        violations.push({
          testName: metric.testName,
          type: 'memory',
          value: metric.memory.used,
          threshold: this.thresholds.maxMemoryUsage,
          severity: 'medium'
        })
      }
    })

    return violations
  }

  async exportToFile(filePath: string): Promise<void> {
    const report = this.generateReport()
    await writeFileAsync(filePath, report, 'utf8')
  }

  clearMetrics(): void {
    this.metrics = []
    this.startTimes.clear()
    this.emit('metrics-cleared')
  }

  private formatBytes(bytes: number): string {
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    if (bytes === 0) return '0 Bytes'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
  }

  // Utility method for measuring async operations
  async measureAsync<T>(testName: string, operation: () => Promise<T>): Promise<T> {
    this.startTest(testName)
    try {
      const result = await operation()
      this.endTest(testName)
      return result
    } catch (error) {
      this.endTest(testName)
      throw error
    }
  }

  // Utility method for measuring synchronous operations
  measureSync<T>(testName: string, operation: () => T): T {
    this.startTest(testName)
    try {
      const result = operation()
      this.endTest(testName)
      return result
    } catch (error) {
      this.endTest(testName)
      throw error
    }
  }
}

// Singleton instance for global usage
export const performanceMonitor = new PerformanceMonitor()

// Utility functions
export function withPerformanceTracking<T extends any[], R>(
  testName: string,
  fn: (...args: T) => R
): (...args: T) => R {
  return (...args: T): R => {
    return performanceMonitor.measureSync(testName, () => fn(...args))
  }
}

export function withAsyncPerformanceTracking<T extends any[], R>(
  testName: string,
  fn: (...args: T) => Promise<R>
): (...args: T) => Promise<R> {
  return async (...args: T): Promise<R> => {
    return performanceMonitor.measureAsync(testName, () => fn(...args))
  }
}

// Example usage decorator
export function PerformanceTest(testName?: string) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value
    const name = testName || `${target.constructor.name}.${propertyKey}`

    descriptor.value = function (...args: any[]) {
      return performanceMonitor.measureSync(name, () => originalMethod.apply(this, args))
    }

    return descriptor
  }
}

export function AsyncPerformanceTest(testName?: string) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const originalMethod = descriptor.value
    const name = testName || `${target.constructor.name}.${propertyKey}`

    descriptor.value = async function (...args: any[]) {
      return performanceMonitor.measureAsync(name, () => originalMethod.apply(this, args))
    }

    return descriptor
  }
}