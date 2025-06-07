import { faker } from '@faker-js/faker'
import { randomBytes } from 'crypto'

export interface MockUser {
  id: string
  email: string
  username: string
  firstName: string
  lastName: string
  avatar: string
  createdAt: Date
  lastLogin: Date
  preferences: {
    notifications: boolean
    theme: 'light' | 'dark'
    language: string
  }
  subscription: {
    plan: 'free' | 'pro' | 'enterprise'
    startDate: Date
    endDate?: Date
    features: string[]
  }
}

export interface MockAgent {
  id: string
  name: string
  type: 'sensor' | 'analyzer' | 'executor' | 'guardian'
  status: 'active' | 'idle' | 'error' | 'maintenance'
  version: string
  capabilities: string[]
  performance: {
    score: number
    uptime: number
    successRate: number
    averageResponseTime: number
  }
  resources: {
    cpu: number
    memory: number
    network: number
  }
  configuration: Record<string, any>
  createdAt: Date
  lastSeen: Date
}

export interface MockMission {
  id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused'
  priority: 'low' | 'medium' | 'high' | 'critical'
  progress: number
  estimatedDuration: number
  actualDuration?: number
  assignedAgents: string[]
  requirements: {
    minAgents: number
    requiredCapabilities: string[]
    resourceLimits: {
      cpu: number
      memory: number
      storage: number
    }
  }
  results?: {
    data: any
    artifacts: string[]
    metrics: Record<string, number>
  }
  createdAt: Date
  startedAt?: Date
  completedAt?: Date
}

export interface MockTransaction {
  id: string
  hash: string
  from: string
  to: string
  value: string
  gasPrice: string
  gasLimit: number
  gasUsed?: number
  blockNumber?: number
  blockHash?: string
  timestamp: Date
  status: 'pending' | 'confirmed' | 'failed'
  type: 'transfer' | 'contract_call' | 'contract_creation'
  network: 'ethereum' | 'solana' | 'polygon' | 'arbitrum'
}

export interface MockMarketData {
  symbol: string
  price: number
  volume24h: number
  marketCap: number
  change24h: number
  high24h: number
  low24h: number
  timestamp: Date
  exchange: string
}

export interface MockLog {
  id: string
  level: 'debug' | 'info' | 'warn' | 'error' | 'fatal'
  message: string
  timestamp: Date
  source: string
  metadata: Record<string, any>
  traceId?: string
  spanId?: string
}

export class MockDataGenerator {
  private userCounter = 0
  private agentCounter = 0
  private missionCounter = 0
  private transactionCounter = 0

  constructor(private seed?: string) {
    if (seed) {
      faker.seed(parseInt(seed, 16))
    }
  }

  generateUser(overrides: Partial<MockUser> = {}): MockUser {
    const userId = `user-${(++this.userCounter).toString().padStart(6, '0')}`
    const firstName = faker.person.firstName()
    const lastName = faker.person.lastName()
    
    return {
      id: userId,
      email: faker.internet.email({ firstName, lastName }),
      username: faker.internet.username({ firstName, lastName }),
      firstName,
      lastName,
      avatar: faker.image.avatar(),
      createdAt: faker.date.past({ years: 2 }),
      lastLogin: faker.date.recent({ days: 7 }),
      preferences: {
        notifications: faker.datatype.boolean(),
        theme: faker.helpers.arrayElement(['light', 'dark']),
        language: faker.helpers.arrayElement(['en', 'es', 'fr', 'de', 'ja'])
      },
      subscription: {
        plan: faker.helpers.arrayElement(['free', 'pro', 'enterprise']),
        startDate: faker.date.past({ years: 1 }),
        endDate: faker.datatype.boolean() ? faker.date.future({ years: 1 }) : undefined,
        features: faker.helpers.arrayElements([
          'advanced_analytics', 'priority_support', 'custom_agents',
          'unlimited_missions', 'api_access', 'white_label'
        ], { min: 1, max: 4 })
      },
      ...overrides
    }
  }

  generateUsers(count: number): MockUser[] {
    return Array.from({ length: count }, () => this.generateUser())
  }

  generateAgent(overrides: Partial<MockAgent> = {}): MockAgent {
    const agentId = `agent-${(++this.agentCounter).toString().padStart(6, '0')}`
    const agentType = faker.helpers.arrayElement(['sensor', 'analyzer', 'executor', 'guardian'])
    
    const capabilities = {
      sensor: ['data_collection', 'real_time_monitoring', 'event_detection'],
      analyzer: ['pattern_recognition', 'risk_assessment', 'data_analysis'],
      executor: ['transaction_execution', 'automated_trading', 'smart_contracts'],
      guardian: ['security_monitoring', 'threat_detection', 'compliance_checking']
    }

    return {
      id: agentId,
      name: `${faker.hacker.adjective()} ${faker.hacker.noun()} ${agentType}`,
      type: agentType,
      status: faker.helpers.arrayElement(['active', 'idle', 'error', 'maintenance']),
      version: `${faker.datatype.number({ min: 1, max: 3 })}.${faker.datatype.number({ min: 0, max: 9 })}.${faker.datatype.number({ min: 0, max: 9 })}`,
      capabilities: capabilities[agentType] || [],
      performance: {
        score: faker.datatype.float({ min: 0, max: 1, precision: 0.01 }),
        uptime: faker.datatype.float({ min: 0.8, max: 1, precision: 0.001 }),
        successRate: faker.datatype.float({ min: 0.7, max: 1, precision: 0.001 }),
        averageResponseTime: faker.datatype.number({ min: 50, max: 2000 })
      },
      resources: {
        cpu: faker.datatype.float({ min: 0, max: 100, precision: 0.1 }),
        memory: faker.datatype.float({ min: 0, max: 100, precision: 0.1 }),
        network: faker.datatype.float({ min: 0, max: 100, precision: 0.1 })
      },
      configuration: {
        maxConcurrentTasks: faker.datatype.number({ min: 1, max: 10 }),
        timeout: faker.datatype.number({ min: 5000, max: 30000 }),
        retryCount: faker.datatype.number({ min: 3, max: 10 }),
        logLevel: faker.helpers.arrayElement(['debug', 'info', 'warn', 'error'])
      },
      createdAt: faker.date.past({ years: 1 }),
      lastSeen: faker.date.recent({ days: 1 }),
      ...overrides
    }
  }

  generateAgents(count: number): MockAgent[] {
    return Array.from({ length: count }, () => this.generateAgent())
  }

  generateMission(overrides: Partial<MockMission> = {}): MockMission {
    const missionId = `mission-${(++this.missionCounter).toString().padStart(6, '0')}`
    const status = faker.helpers.arrayElement(['pending', 'running', 'completed', 'failed', 'paused'])
    
    const createdAt = faker.date.past({ years: 1 })
    const startedAt = status !== 'pending' ? faker.date.between({ from: createdAt, to: new Date() }) : undefined
    const completedAt = ['completed', 'failed'].includes(status) && startedAt 
      ? faker.date.between({ from: startedAt, to: new Date() }) : undefined

    return {
      id: missionId,
      name: `${faker.hacker.verb()} ${faker.hacker.noun()} Analysis`,
      description: faker.lorem.sentences(2),
      status,
      priority: faker.helpers.arrayElement(['low', 'medium', 'high', 'critical']),
      progress: status === 'completed' ? 1 : faker.datatype.float({ min: 0, max: 0.9, precision: 0.01 }),
      estimatedDuration: faker.datatype.number({ min: 300, max: 7200 }), // 5 minutes to 2 hours
      actualDuration: completedAt && startedAt ? completedAt.getTime() - startedAt.getTime() : undefined,
      assignedAgents: faker.helpers.arrayElements([
        'agent-001', 'agent-002', 'agent-003', 'agent-004', 'agent-005'
      ], { min: 1, max: 3 }),
      requirements: {
        minAgents: faker.datatype.number({ min: 1, max: 5 }),
        requiredCapabilities: faker.helpers.arrayElements([
          'data_collection', 'pattern_recognition', 'risk_assessment', 'real_time_monitoring'
        ], { min: 1, max: 3 }),
        resourceLimits: {
          cpu: faker.datatype.number({ min: 50, max: 100 }),
          memory: faker.datatype.number({ min: 512, max: 4096 }),
          storage: faker.datatype.number({ min: 100, max: 1000 })
        }
      },
      results: status === 'completed' ? {
        data: {
          findings: faker.datatype.number({ min: 0, max: 50 }),
          anomalies: faker.datatype.number({ min: 0, max: 10 }),
          riskScore: faker.datatype.float({ min: 0, max: 1, precision: 0.01 })
        },
        artifacts: faker.helpers.arrayElements([
          'report.pdf', 'data.json', 'chart.png', 'log.txt'
        ], { min: 1, max: 4 }),
        metrics: {
          executionTime: faker.datatype.number({ min: 100, max: 5000 }),
          dataProcessed: faker.datatype.number({ min: 1000, max: 1000000 }),
          accuracy: faker.datatype.float({ min: 0.8, max: 1, precision: 0.001 })
        }
      } : undefined,
      createdAt,
      startedAt,
      completedAt,
      ...overrides
    }
  }

  generateMissions(count: number): MockMission[] {
    return Array.from({ length: count }, () => this.generateMission())
  }

  generateTransaction(overrides: Partial<MockTransaction> = {}): MockTransaction {
    const networks = ['ethereum', 'solana', 'polygon', 'arbitrum'] as const
    const network = faker.helpers.arrayElement(networks)
    
    return {
      id: `tx-${(++this.transactionCounter).toString().padStart(8, '0')}`,
      hash: `0x${randomBytes(32).toString('hex')}`,
      from: network === 'solana' ? faker.string.alphanumeric(44) : `0x${randomBytes(20).toString('hex')}`,
      to: network === 'solana' ? faker.string.alphanumeric(44) : `0x${randomBytes(20).toString('hex')}`,
      value: faker.datatype.bigInt({ min: 1000, max: 1000000000000000000 }).toString(),
      gasPrice: faker.datatype.number({ min: 1000000000, max: 100000000000 }).toString(),
      gasLimit: faker.datatype.number({ min: 21000, max: 500000 }),
      gasUsed: faker.datatype.number({ min: 21000, max: 500000 }),
      blockNumber: faker.datatype.number({ min: 15000000, max: 20000000 }),
      blockHash: `0x${randomBytes(32).toString('hex')}`,
      timestamp: faker.date.recent({ days: 7 }),
      status: faker.helpers.arrayElement(['pending', 'confirmed', 'failed']),
      type: faker.helpers.arrayElement(['transfer', 'contract_call', 'contract_creation']),
      network,
      ...overrides
    }
  }

  generateTransactions(count: number): MockTransaction[] {
    return Array.from({ length: count }, () => this.generateTransaction())
  }

  generateMarketData(overrides: Partial<MockMarketData> = {}): MockMarketData {
    const symbols = ['BTC', 'ETH', 'SOL', 'MATIC', 'LINK', 'UNI', 'AAVE', 'COMP']
    const symbol = faker.helpers.arrayElement(symbols)
    const basePrice = { BTC: 45000, ETH: 2500, SOL: 100, MATIC: 0.8, LINK: 15, UNI: 8, AAVE: 80, COMP: 50 }[symbol] || 100
    
    const change = faker.datatype.float({ min: -20, max: 20, precision: 0.01 })
    const price = basePrice * (1 + change / 100)
    
    return {
      symbol,
      price,
      volume24h: faker.datatype.number({ min: 1000000, max: 1000000000 }),
      marketCap: price * faker.datatype.number({ min: 1000000, max: 1000000000 }),
      change24h: change,
      high24h: price * faker.datatype.float({ min: 1, max: 1.1 }),
      low24h: price * faker.datatype.float({ min: 0.9, max: 1 }),
      timestamp: faker.date.recent({ days: 1 }),
      exchange: faker.helpers.arrayElement(['binance', 'coinbase', 'kraken', 'uniswap']),
      ...overrides
    }
  }

  generateMarketDataSeries(symbol: string, hours: number): MockMarketData[] {
    const basePrice = faker.datatype.number({ min: 100, max: 50000 })
    const data: MockMarketData[] = []
    
    for (let i = 0; i < hours; i++) {
      const timestamp = new Date(Date.now() - (hours - i) * 60 * 60 * 1000)
      const volatility = faker.datatype.float({ min: 0.98, max: 1.02 })
      const price = i === 0 ? basePrice : data[i - 1].price * volatility
      
      data.push(this.generateMarketData({
        symbol,
        price,
        timestamp
      }))
    }
    
    return data
  }

  generateLog(overrides: Partial<MockLog> = {}): MockLog {
    const level = faker.helpers.arrayElement(['debug', 'info', 'warn', 'error', 'fatal'])
    const sources = ['gateway', 'auth-service', 'agent-runtime', 'notifier', 'database']
    
    const messages = {
      debug: () => `Debug information: ${faker.hacker.phrase()}`,
      info: () => `Operation completed: ${faker.hacker.verb()} ${faker.hacker.noun()}`,
      warn: () => `Warning: ${faker.hacker.phrase()} may cause issues`,
      error: () => `Error: Failed to ${faker.hacker.verb()} ${faker.hacker.noun()}`,
      fatal: () => `Fatal error: System ${faker.hacker.noun()} crashed`
    }

    return {
      id: faker.string.uuid(),
      level,
      message: messages[level](),
      timestamp: faker.date.recent({ days: 1 }),
      source: faker.helpers.arrayElement(sources),
      metadata: {
        userId: faker.string.uuid(),
        requestId: faker.string.uuid(),
        duration: faker.datatype.number({ min: 10, max: 5000 }),
        statusCode: level === 'error' ? faker.helpers.arrayElement([400, 401, 403, 404, 500, 502, 503]) : 200
      },
      traceId: faker.string.uuid(),
      spanId: faker.string.alphanumeric(16),
      ...overrides
    }
  }

  generateLogs(count: number): MockLog[] {
    return Array.from({ length: count }, () => this.generateLog())
  }

  // Utility method to generate realistic time series data
  generateTimeSeries<T>(
    generator: () => T,
    count: number,
    interval: number = 60000 // 1 minute default
  ): (T & { timestamp: Date })[] {
    return Array.from({ length: count }, (_, i) => ({
      ...generator(),
      timestamp: new Date(Date.now() - (count - i) * interval)
    }))
  }

  // Generate complete test dataset
  generateTestDataset(): {
    users: MockUser[]
    agents: MockAgent[]
    missions: MockMission[]
    transactions: MockTransaction[]
    marketData: MockMarketData[]
    logs: MockLog[]
  } {
    return {
      users: this.generateUsers(faker.datatype.number({ min: 10, max: 50 })),
      agents: this.generateAgents(faker.datatype.number({ min: 5, max: 20 })),
      missions: this.generateMissions(faker.datatype.number({ min: 5, max: 30 })),
      transactions: this.generateTransactions(faker.datatype.number({ min: 100, max: 500 })),
      marketData: this.generateMarketDataSeries('BTC', 24),
      logs: this.generateLogs(faker.datatype.number({ min: 50, max: 200 }))
    }
  }
}

// Export singleton instance
export const mockDataGenerator = new MockDataGenerator()

// Utility functions
export function createSeededGenerator(seed: string): MockDataGenerator {
  return new MockDataGenerator(seed)
}

export function generateBulkTestData(counts: {
  users?: number
  agents?: number
  missions?: number
  transactions?: number
  logs?: number
}): any {
  const generator = new MockDataGenerator()
  
  return {
    users: counts.users ? generator.generateUsers(counts.users) : [],
    agents: counts.agents ? generator.generateAgents(counts.agents) : [],
    missions: counts.missions ? generator.generateMissions(counts.missions) : [],
    transactions: counts.transactions ? generator.generateTransactions(counts.transactions) : [],
    logs: counts.logs ? generator.generateLogs(counts.logs) : []
  }
}