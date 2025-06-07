import express, { Request, Response } from 'express'
import { promisify } from 'util'
import { exec } from 'child_process'
import { Redis } from 'ioredis'
import { Pool } from 'pg'
import axios from 'axios'

const execAsync = promisify(exec)

export interface ServiceStatus {
  name: string
  status: 'operational' | 'degraded_performance' | 'partial_outage' | 'major_outage'
  lastChecked: Date
  responseTime?: number
  uptime?: number
  message?: string
}

export interface SystemMetrics {
  cpu: {
    usage: number
    cores: number
  }
  memory: {
    used: number
    total: number
    percentage: number
  }
  disk: {
    used: number
    total: number
    percentage: number
  }
  network: {
    inbound: number
    outbound: number
  }
}

export interface IncidentReport {
  id: string
  title: string
  description: string
  status: 'investigating' | 'identified' | 'monitoring' | 'resolved'
  severity: 'minor' | 'major' | 'critical'
  affectedServices: string[]
  createdAt: Date
  updatedAt: Date
  resolvedAt?: Date
  updates: {
    timestamp: Date
    message: string
    status: string
  }[]
}

export class StatusPageService {
  private redis: Redis
  private db: Pool
  private services: Map<string, ServiceStatus> = new Map()
  private incidents: Map<string, IncidentReport> = new Map()
  private metrics: SystemMetrics | null = null

  constructor(redisUrl: string, dbConfig: any) {
    this.redis = new Redis(redisUrl)
    this.db = new Pool(dbConfig)
    this.initializeServices()
  }

  private initializeServices(): void {
    const serviceConfigs = [
      { name: 'Gateway', url: process.env.GATEWAY_URL || 'http://localhost:8080/health' },
      { name: 'Auth Service', url: process.env.AUTH_URL || 'http://localhost:8081/health' },
      { name: 'Agent Runtime', url: process.env.AGENT_RUNTIME_URL || 'http://localhost:8082/health' },
      { name: 'Notifier', url: process.env.NOTIFIER_URL || 'http://localhost:8083/health' },
      { name: 'Mission Control', url: process.env.MISSION_CONTROL_URL || 'http://localhost:8084/health' },
      { name: 'Database', type: 'database' },
      { name: 'Redis', type: 'redis' },
      { name: 'Message Queue', url: process.env.MQ_URL || 'http://localhost:15672/api/healthchecks/node' }
    ]

    serviceConfigs.forEach(config => {
      this.services.set(config.name, {
        name: config.name,
        status: 'operational',
        lastChecked: new Date()
      })
    })
  }

  async checkServiceHealth(serviceName: string): Promise<ServiceStatus> {
    const service = this.services.get(serviceName)
    if (!service) {
      throw new Error(`Service ${serviceName} not found`)
    }

    const startTime = Date.now()
    let status: ServiceStatus['status'] = 'operational'
    let message: string | undefined

    try {
      switch (serviceName) {
        case 'Database':
          await this.checkDatabaseHealth()
          break
        case 'Redis':
          await this.checkRedisHealth()
          break
        default:
          const url = this.getServiceUrl(serviceName)
          if (url) {
            await this.checkHttpHealth(url)
          }
      }
    } catch (error) {
      status = 'major_outage'
      message = error instanceof Error ? error.message : 'Unknown error'
    }

    const responseTime = Date.now() - startTime
    const updatedService: ServiceStatus = {
      ...service,
      status,
      lastChecked: new Date(),
      responseTime,
      message
    }

    this.services.set(serviceName, updatedService)
    return updatedService
  }

  private async checkHttpHealth(url: string): Promise<void> {
    const response = await axios.get(url, {
      timeout: 5000,
      validateStatus: (status) => status < 500
    })

    if (response.status >= 400) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
  }

  private async checkDatabaseHealth(): Promise<void> {
    const client = await this.db.connect()
    try {
      await client.query('SELECT 1')
    } finally {
      client.release()
    }
  }

  private async checkRedisHealth(): Promise<void> {
    await this.redis.ping()
  }

  private getServiceUrl(serviceName: string): string | undefined {
    const urls: Record<string, string> = {
      'Gateway': process.env.GATEWAY_URL || 'http://localhost:8080/health',
      'Auth Service': process.env.AUTH_URL || 'http://localhost:8081/health',
      'Agent Runtime': process.env.AGENT_RUNTIME_URL || 'http://localhost:8082/health',
      'Notifier': process.env.NOTIFIER_URL || 'http://localhost:8083/health',
      'Mission Control': process.env.MISSION_CONTROL_URL || 'http://localhost:8084/health',
      'Message Queue': process.env.MQ_URL || 'http://localhost:15672/api/healthchecks/node'
    }
    return urls[serviceName]
  }

  async checkAllServices(): Promise<ServiceStatus[]> {
    const promises = Array.from(this.services.keys()).map(name => 
      this.checkServiceHealth(name).catch(error => ({
        name,
        status: 'major_outage' as const,
        lastChecked: new Date(),
        message: error.message
      }))
    )

    const results = await Promise.all(promises)
    return results
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    try {
      // Get CPU usage
      const { stdout: cpuInfo } = await execAsync("wmic cpu get loadpercentage /value | findstr LoadPercentage")
      const cpuUsage = parseInt(cpuInfo.split('=')[1]?.trim() || '0')

      // Get memory info
      const { stdout: memInfo } = await execAsync('wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value')
      const memLines = memInfo.split('\n').filter(line => line.includes('='))
      const totalMem = parseInt(memLines.find(line => line.includes('TotalVisibleMemorySize'))?.split('=')[1]?.trim() || '0') * 1024
      const freeMem = parseInt(memLines.find(line => line.includes('FreePhysicalMemory'))?.split('=')[1]?.trim() || '0') * 1024
      const usedMem = totalMem - freeMem

      // Get disk info
      const { stdout: diskInfo } = await execAsync('wmic logicaldisk get Size,FreeSpace /value | findstr "="')
      const diskLines = diskInfo.split('\n').filter(line => line.includes('='))
      const totalDisk = diskLines.filter(line => line.includes('Size')).reduce((sum, line) => {
        return sum + parseInt(line.split('=')[1]?.trim() || '0')
      }, 0)
      const freeDisk = diskLines.filter(line => line.includes('FreeSpace')).reduce((sum, line) => {
        return sum + parseInt(line.split('=')[1]?.trim() || '0')
      }, 0)
      const usedDisk = totalDisk - freeDisk

      this.metrics = {
        cpu: {
          usage: cpuUsage,
          cores: require('os').cpus().length
        },
        memory: {
          used: usedMem,
          total: totalMem,
          percentage: (usedMem / totalMem) * 100
        },
        disk: {
          used: usedDisk,
          total: totalDisk,
          percentage: (usedDisk / totalDisk) * 100
        },
        network: {
          inbound: 0, // Would need network monitoring tool
          outbound: 0
        }
      }

      return this.metrics
    } catch (error) {
      // Fallback to Node.js process metrics
      const process = require('process')
      const os = require('os')

      return {
        cpu: {
          usage: process.cpuUsage().user / 1000000,
          cores: os.cpus().length
        },
        memory: {
          used: process.memoryUsage().rss,
          total: os.totalmem(),
          percentage: (process.memoryUsage().rss / os.totalmem()) * 100
        },
        disk: {
          used: 0,
          total: 0,
          percentage: 0
        },
        network: {
          inbound: 0,
          outbound: 0
        }
      }
    }
  }

  async createIncident(incident: Omit<IncidentReport, 'id' | 'createdAt' | 'updatedAt' | 'updates'>): Promise<IncidentReport> {
    const id = `incident-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const newIncident: IncidentReport = {
      ...incident,
      id,
      createdAt: new Date(),
      updatedAt: new Date(),
      updates: []
    }

    this.incidents.set(id, newIncident)
    await this.redis.hset('incidents', id, JSON.stringify(newIncident))
    
    return newIncident
  }

  async updateIncident(id: string, update: {
    message: string
    status?: IncidentReport['status']
  }): Promise<IncidentReport> {
    const incident = this.incidents.get(id)
    if (!incident) {
      throw new Error(`Incident ${id} not found`)
    }

    const updatedIncident: IncidentReport = {
      ...incident,
      status: update.status || incident.status,
      updatedAt: new Date(),
      resolvedAt: update.status === 'resolved' ? new Date() : incident.resolvedAt,
      updates: [
        ...incident.updates,
        {
          timestamp: new Date(),
          message: update.message,
          status: update.status || incident.status
        }
      ]
    }

    this.incidents.set(id, updatedIncident)
    await this.redis.hset('incidents', id, JSON.stringify(updatedIncident))
    
    return updatedIncident
  }

  getIncidents(): IncidentReport[] {
    return Array.from(this.incidents.values()).sort((a, b) => 
      b.createdAt.getTime() - a.createdAt.getTime()
    )
  }

  getActiveIncidents(): IncidentReport[] {
    return this.getIncidents().filter(incident => incident.status !== 'resolved')
  }

  getOverallStatus(): 'operational' | 'degraded_performance' | 'partial_outage' | 'major_outage' {
    const statuses = Array.from(this.services.values()).map(s => s.status)
    
    if (statuses.some(s => s === 'major_outage')) return 'major_outage'
    if (statuses.some(s => s === 'partial_outage')) return 'partial_outage'
    if (statuses.some(s => s === 'degraded_performance')) return 'degraded_performance'
    
    return 'operational'
  }

  async getStatusPage(): Promise<{
    overall: string
    services: ServiceStatus[]
    incidents: IncidentReport[]
    metrics: SystemMetrics
    lastUpdated: Date
  }> {
    const services = await this.checkAllServices()
    const metrics = await this.getSystemMetrics()
    const incidents = this.getActiveIncidents()
    
    return {
      overall: this.getOverallStatus(),
      services,
      incidents,
      metrics,
      lastUpdated: new Date()
    }
  }
}

// Express router setup
export function createStatusPageRouter(statusService: StatusPageService): express.Router {
  const router = express.Router()

  // Main status page endpoint
  router.get('/', async (req: Request, res: Response) => {
    try {
      const statusPage = await statusService.getStatusPage()
      res.json(statusPage)
    } catch (error) {
      res.status(500).json({ 
        error: 'Failed to get status page',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  })

  // Individual service health check
  router.get('/services/:serviceName', async (req: Request, res: Response) => {
    try {
      const { serviceName } = req.params
      const service = await statusService.checkServiceHealth(serviceName)
      res.json(service)
    } catch (error) {
      res.status(404).json({ 
        error: 'Service not found',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  })

  // System metrics endpoint
  router.get('/metrics', async (req: Request, res: Response) => {
    try {
      const metrics = await statusService.getSystemMetrics()
      res.json(metrics)
    } catch (error) {
      res.status(500).json({ 
        error: 'Failed to get metrics',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  })

  // Incidents endpoints
  router.get('/incidents', (req: Request, res: Response) => {
    const incidents = statusService.getIncidents()
    res.json(incidents)
  })

  router.get('/incidents/active', (req: Request, res: Response) => {
    const incidents = statusService.getActiveIncidents()
    res.json(incidents)
  })

  router.post('/incidents', async (req: Request, res: Response) => {
    try {
      const incident = await statusService.createIncident(req.body)
      res.status(201).json(incident)
    } catch (error) {
      res.status(400).json({ 
        error: 'Failed to create incident',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  })

  router.patch('/incidents/:id', async (req: Request, res: Response) => {
    try {
      const { id } = req.params
      const incident = await statusService.updateIncident(id, req.body)
      res.json(incident)
    } catch (error) {
      res.status(404).json({ 
        error: 'Failed to update incident',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  })

  // Health check endpoint for the status service itself
  router.get('/health', (req: Request, res: Response) => {
    res.json({ 
      status: 'ok', 
      timestamp: new Date().toISOString(),
      uptime: process.uptime()
    })
  })

  return router
}

// HTML status page renderer (optional)
export function renderStatusPageHTML(statusData: any): string {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational': return '#10B981'
      case 'degraded_performance': return '#F59E0B'
      case 'partial_outage': return '#EF4444'
      case 'major_outage': return '#DC2626'
      default: return '#6B7280'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'operational': return '✅'
      case 'degraded_performance': return '⚠️'
      case 'partial_outage': return '🔶'
      case 'major_outage': return '🔴'
      default: return '⚪'
    }
  }

  return `
<!DOCTYPE html>
<html>
<head>
    <title>Prowzi System Status</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f8fafc; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .header { padding: 30px; border-bottom: 1px solid #e5e7eb; text-align: center; }
        .status-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; color: white; font-weight: 500; margin-top: 10px; }
        .services { padding: 30px; }
        .service { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #f3f4f6; }
        .service:last-child { border-bottom: none; }
        .service-name { font-weight: 500; }
        .service-status { display: flex; align-items: center; gap: 8px; }
        .incidents { padding: 30px; background: #fef2f2; margin: 20px; border-radius: 8px; }
        .incident { margin-bottom: 20px; padding: 16px; background: white; border-radius: 6px; border-left: 4px solid #ef4444; }
        .metrics { padding: 30px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .metric { text-align: center; padding: 20px; background: #f8fafc; border-radius: 6px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #1f2937; }
        .metric-label { color: #6b7280; margin-top: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Prowzi System Status</h1>
            <div class="status-badge" style="background-color: ${getStatusColor(statusData.overall)}">
                ${getStatusIcon(statusData.overall)} ${statusData.overall.replace('_', ' ').toUpperCase()}
            </div>
            <p style="color: #6b7280; margin-top: 10px;">Last updated: ${statusData.lastUpdated}</p>
        </div>
        
        <div class="services">
            <h2>Services</h2>
            ${statusData.services.map((service: ServiceStatus) => `
                <div class="service">
                    <span class="service-name">${service.name}</span>
                    <div class="service-status">
                        <span>${getStatusIcon(service.status)}</span>
                        <span style="color: ${getStatusColor(service.status)}">${service.status.replace('_', ' ')}</span>
                        ${service.responseTime ? `<span style="color: #6b7280">(${service.responseTime}ms)</span>` : ''}
                    </div>
                </div>
            `).join('')}
        </div>
        
        ${statusData.incidents.length > 0 ? `
        <div class="incidents">
            <h2>Active Incidents</h2>
            ${statusData.incidents.map((incident: IncidentReport) => `
                <div class="incident">
                    <h3>${incident.title}</h3>
                    <p>${incident.description}</p>
                    <p><strong>Status:</strong> ${incident.status} | <strong>Severity:</strong> ${incident.severity}</p>
                    <p><strong>Affected:</strong> ${incident.affectedServices.join(', ')}</p>
                </div>
            `).join('')}
        </div>
        ` : ''}
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">${statusData.metrics.cpu.usage.toFixed(1)}%</div>
                <div class="metric-label">CPU Usage</div>
            </div>
            <div class="metric">
                <div class="metric-value">${statusData.metrics.memory.percentage.toFixed(1)}%</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            <div class="metric">
                <div class="metric-value">${statusData.metrics.disk.percentage.toFixed(1)}%</div>
                <div class="metric-label">Disk Usage</div>
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
  `
}