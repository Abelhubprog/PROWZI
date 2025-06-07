
import { z } from 'zod'

const MissionPlanSchema = z.object({
  objectives: z.array(z.object({
    id: z.string(),
    description: z.string(),
    priority: z.enum(['critical', 'high', 'medium', 'low']),
    dependencies: z.array(z.string()),
  })),
  agents: z.array(z.object({
    type: z.string(),
    count: z.number(),
    configuration: z.record(z.any()),
  })),
  resources: z.object({
    estimatedTokens: z.number(),
    estimatedComputeHours: z.number(),
    requiredDataSources: z.array(z.string()),
  }),
  timeline: z.object({
    phases: z.array(z.object({
      name: z.string(),
      duration: z.number(),
      objectives: z.array(z.string()),
    })),
  }),
})

export type MissionPlan = z.infer<typeof MissionPlanSchema>

export interface MissionConstraints {
  maxDuration?: number
  tokenBudget?: number
  dataSources?: string[]
  priority?: 'low' | 'medium' | 'high' | 'critical'
}

export interface MissionTemplate {
  id: string
  name: string
  description: string
  category: string
  defaultAgents: Array<{
    type: string
    count: number
    config: Record<string, any>
  }>
  estimatedResources: {
    tokens: number
    computeHours: number
    duration: number
  }
}

export interface MissionClassification {
  type: string
  category: string
  complexity: 'simple' | 'moderate' | 'complex'
  domains: string[]
  confidence: number
}

export class MissionPlanner {
  private templates: Map<string, MissionTemplate>
  private missionHistory: Map<string, any[]>
  
  constructor() {
    this.templates = new Map()
    this.missionHistory = new Map()
    this.loadTemplates()
  }
  
  private loadTemplates() {
    const templates: MissionTemplate[] = [
      {
        id: 'crypto_monitoring',
        name: 'Crypto Market Monitoring',
        description: 'Monitor cryptocurrency markets for significant events',
        category: 'monitoring',
        defaultAgents: [
          { type: 'solana_sensor', count: 2, config: { watchlist: 'top_100' } },
          { type: 'ethereum_sensor', count: 1, config: { watchlist: 'defi' } },
          { type: 'whale_detector', count: 1, config: { threshold: 1000000 } },
          { type: 'pattern_analyzer', count: 1, config: { window: '1h' } },
        ],
        estimatedResources: {
          tokens: 5000,
          computeHours: 24,
          duration: 48,
        }
      },
      {
        id: 'ai_research_tracking',
        name: 'AI Research Tracking',
        description: 'Track AI research papers and model releases',
        category: 'research',
        defaultAgents: [
          { type: 'arxiv_sensor', count: 1, config: { categories: ['cs.AI', 'cs.LG'] } },
          { type: 'github_sensor', count: 1, config: { topics: ['machine-learning'] } },
          { type: 'paper_analyzer', count: 1, config: { min_citations: 10 } },
          { type: 'trend_detector', count: 1, config: { window: '7d' } },
        ],
        estimatedResources: {
          tokens: 3000,
          computeHours: 12,
          duration: 168,
        }
      },
      {
        id: 'token_launch_hunter',
        name: 'Token Launch Hunter',
        description: 'Detect and analyze new token launches',
        category: 'detection',
        defaultAgents: [
          { type: 'solana_mempool_sensor', count: 3, config: { filter: 'token_creation' } },
          { type: 'rug_risk_analyzer', count: 2, config: { threshold: 0.7 } },
          { type: 'social_sentiment_sensor', count: 1, config: { platforms: ['twitter', 'discord'] } },
          { type: 'liquidity_analyzer', count: 1, config: { min_liquidity: 10000 } },
        ],
        estimatedResources: {
          tokens: 8000,
          computeHours: 6,
          duration: 24,
        }
      }
    ]
    
    templates.forEach(template => {
      this.templates.set(template.id, template)
    })
  }
  
  async generatePlan(prompt: string, constraints?: MissionConstraints): Promise<MissionPlan> {
    // Classify the mission type
    const classification = await this.classifyMission(prompt)
    
    // Get relevant template if exists
    const template = this.templates.get(classification.type)
    
    // Generate detailed plan
    const plan = await this.createDetailedPlan(prompt, classification, template, constraints)
    
    // Optimize based on historical performance
    const optimizedPlan = await this.optimizePlan(plan, classification)
    
    // Validate resource requirements
    this.validateResources(optimizedPlan, constraints)
    
    return optimizedPlan
  }
  
  private async classifyMission(prompt: string): Promise<MissionClassification> {
    const promptLower = prompt.toLowerCase()
    
    // Simple keyword-based classification
    let type = 'general_monitoring'
    let category = 'monitoring'
    let domains: string[] = []
    let complexity: 'simple' | 'moderate' | 'complex' = 'moderate'
    
    // Crypto indicators
    if (promptLower.includes('token') || promptLower.includes('crypto') || 
        promptLower.includes('solana') || promptLower.includes('ethereum')) {
      domains.push('crypto')
      
      if (promptLower.includes('launch') || promptLower.includes('new token')) {
        type = 'token_launch_hunter'
        category = 'detection'
        complexity = 'simple'
      } else if (promptLower.includes('whale') || promptLower.includes('large')) {
        type = 'crypto_monitoring'
        category = 'monitoring'
        complexity = 'moderate'
      }
    }
    
    // AI indicators
    if (promptLower.includes('ai') || promptLower.includes('model') || 
        promptLower.includes('research') || promptLower.includes('paper')) {
      domains.push('ai')
      
      if (promptLower.includes('research') || promptLower.includes('paper')) {
        type = 'ai_research_tracking'
        category = 'research'
        complexity = 'moderate'
      }
    }
    
    // Security indicators
    if (promptLower.includes('exploit') || promptLower.includes('hack') || 
        promptLower.includes('vulnerability')) {
      category = 'security'
      complexity = 'complex'
    }
    
    // Complexity indicators
    if (promptLower.includes('deep') || promptLower.includes('comprehensive') || 
        promptLower.includes('analyze')) {
      complexity = 'complex'
    }
    
    return {
      type,
      category,
      complexity,
      domains: domains.length > 0 ? domains : ['general'],
      confidence: 0.8
    }
  }
  
  private async createDetailedPlan(
    prompt: string,
    classification: MissionClassification,
    template?: MissionTemplate,
    constraints?: MissionConstraints
  ): Promise<MissionPlan> {
    
    // Start with template or create from scratch
    let agents = template?.defaultAgents || []
    let estimatedTokens = template?.estimatedResources.tokens || 5000
    let estimatedComputeHours = template?.estimatedResources.computeHours || 12
    let duration = template?.estimatedResources.duration || 48
    
    // Adjust based on classification
    switch (classification.complexity) {
      case 'simple':
        agents = agents.slice(0, 2) // Fewer agents
        estimatedTokens *= 0.5
        estimatedComputeHours *= 0.5
        break
      case 'complex':
        agents = [...agents, { type: 'deep_analyzer', count: 1, config: {} }]
        estimatedTokens *= 1.5
        estimatedComputeHours *= 2
        duration *= 1.5
        break
    }
    
    // Apply constraints
    if (constraints) {
      if (constraints.maxDuration && duration > constraints.maxDuration) {
        duration = constraints.maxDuration
        // Reduce scope to fit duration
        agents = agents.slice(0, Math.max(1, Math.floor(agents.length * 0.7)))
      }
      
      if (constraints.tokenBudget && estimatedTokens > constraints.tokenBudget) {
        const scale = constraints.tokenBudget / estimatedTokens
        estimatedTokens = constraints.tokenBudget
        estimatedComputeHours *= scale
        agents.forEach(agent => {
          agent.count = Math.max(1, Math.floor(agent.count * scale))
        })
      }
    }
    
    // Generate objectives
    const objectives = this.generateObjectives(prompt, classification, agents)
    
    // Create timeline phases
    const phases = this.createPhases(objectives, duration)
    
    // Determine required data sources
    const dataSources = this.determineDataSources(agents, classification)
    
    return {
      objectives,
      agents,
      resources: {
        estimatedTokens: Math.floor(estimatedTokens),
        estimatedComputeHours: Math.floor(estimatedComputeHours),
        requiredDataSources: dataSources,
      },
      timeline: {
        phases,
      },
    }
  }
  
  private generateObjectives(
    prompt: string,
    classification: MissionClassification,
    agents: any[]
  ) {
    const objectives = []
    
    // Primary objective based on prompt
    objectives.push({
      id: 'primary',
      description: `Execute: ${prompt}`,
      priority: 'high' as const,
      dependencies: [],
    })
    
    // Data collection objectives
    if (agents.some(a => a.type.includes('sensor'))) {
      objectives.push({
        id: 'data_collection',
        description: 'Establish data collection pipelines',
        priority: 'high' as const,
        dependencies: [],
      })
    }
    
    // Analysis objectives
    if (agents.some(a => a.type.includes('analyzer'))) {
      objectives.push({
        id: 'analysis',
        description: 'Analyze collected data for patterns and insights',
        priority: 'medium' as const,
        dependencies: ['data_collection'],
      })
    }
    
    // Alerting objectives
    objectives.push({
      id: 'alerting',
      description: 'Generate and deliver actionable alerts',
      priority: 'high' as const,
      dependencies: ['analysis'],
    })
    
    return objectives
  }
  
  private createPhases(objectives: any[], duration: number) {
    const phases = []
    const setupTime = Math.floor(duration * 0.1)
    const executionTime = Math.floor(duration * 0.8)
    const wrapupTime = duration - setupTime - executionTime
    
    phases.push({
      name: 'Setup',
      duration: setupTime,
      objectives: ['data_collection'],
    })
    
    phases.push({
      name: 'Execution',
      duration: executionTime,
      objectives: ['primary', 'analysis'],
    })
    
    phases.push({
      name: 'Wrapup',
      duration: wrapupTime,
      objectives: ['alerting'],
    })
    
    return phases
  }
  
  private determineDataSources(agents: any[], classification: MissionClassification) {
    const sources = new Set<string>()
    
    agents.forEach(agent => {
      switch (agent.type) {
        case 'solana_sensor':
        case 'solana_mempool_sensor':
          sources.add('solana_rpc')
          sources.add('solana_mempool')
          break
        case 'ethereum_sensor':
          sources.add('ethereum_rpc')
          sources.add('ethereum_mempool')
          break
        case 'github_sensor':
          sources.add('github_api')
          break
        case 'arxiv_sensor':
          sources.add('arxiv_rss')
          break
        case 'social_sentiment_sensor':
          sources.add('twitter_api')
          sources.add('discord_webhooks')
          break
      }
    })
    
    return Array.from(sources)
  }
  
  private async optimizePlan(plan: MissionPlan, classification: MissionClassification): Promise<MissionPlan> {
    // Load historical mission performance
    const history = this.missionHistory.get(classification.type) || []
    
    if (history.length === 0) {
      return plan // No optimization without history
    }
    
    // Calculate average performance metrics
    const avgSuccess = history.reduce((sum, h) => sum + (h.success ? 1 : 0), 0) / history.length
    const avgDuration = history.reduce((sum, h) => sum + h.actualDuration, 0) / history.length
    const avgTokens = history.reduce((sum, h) => sum + h.tokensUsed, 0) / history.length
    
    // Adjust estimates based on historical performance
    if (avgDuration > plan.timeline.phases.reduce((sum, p) => sum + p.duration, 0)) {
      // Historical missions took longer, increase duration estimates
      plan.timeline.phases.forEach(phase => {
        phase.duration = Math.floor(phase.duration * 1.2)
      })
    }
    
    if (avgTokens > plan.resources.estimatedTokens) {
      // Historical missions used more tokens
      plan.resources.estimatedTokens = Math.floor(avgTokens * 1.1)
    }
    
    // Adjust agent counts based on success rate
    if (avgSuccess < 0.7) {
      // Low success rate, add redundancy
      plan.agents.forEach(agent => {
        if (agent.type.includes('analyzer') || agent.type.includes('detector')) {
          agent.count = Math.min(agent.count + 1, 3)
        }
      })
    }
    
    return plan
  }
  
  private validateResources(plan: MissionPlan, constraints?: MissionConstraints) {
    if (constraints?.tokenBudget && plan.resources.estimatedTokens > constraints.tokenBudget) {
      throw new Error(`Plan requires ${plan.resources.estimatedTokens} tokens but budget is ${constraints.tokenBudget}`)
    }
    
    if (constraints?.maxDuration) {
      const totalDuration = plan.timeline.phases.reduce((sum, phase) => sum + phase.duration, 0)
      if (totalDuration > constraints.maxDuration) {
        throw new Error(`Plan duration ${totalDuration}h exceeds maximum ${constraints.maxDuration}h`)
      }
    }
  }
  
  async recordMissionHistory(missionType: string, result: any) {
    if (!this.missionHistory.has(missionType)) {
      this.missionHistory.set(missionType, [])
    }
    
    const history = this.missionHistory.get(missionType)!
    history.push(result)
    
    // Keep only last 50 results
    if (history.length > 50) {
      history.splice(0, history.length - 50)
    }
  }
}

export const missionPlanner = new MissionPlanner()
