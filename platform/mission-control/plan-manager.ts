
import { EventEmitter } from 'events';
import { Pool } from 'pg';

interface Mission {
  id: string;
  name: string;
  prompt: string;
  status: 'planning' | 'active' | 'paused' | 'completed' | 'failed';
  plan: MissionPlan;
  config: MissionConfig;
  resource_usage: ResourceUsage;
  created_at: Date;
  updated_at: Date;
  completed_at?: Date;
  user_id?: string;
  tenant_id?: string;
}

interface MissionPlan {
  objectives: Objective[];
  phases: Phase[];
  agents: AgentSpec[];
  timeline: Timeline;
  success_criteria: SuccessCriteria;
  risk_assessment: RiskAssessment;
}

interface Objective {
  id: string;
  description: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  success_metrics: string[];
  dependencies: string[];
  assigned_agents: string[];
}

interface Phase {
  id: string;
  name: string;
  description: string;
  duration_hours: number;
  objectives: string[];
  prerequisites: string[];
  deliverables: string[];
}

interface AgentSpec {
  type: string;
  count: number;
  config: Record<string, any>;
  resource_allocation: {
    tokens: number;
    api_calls: number;
    compute_hours: number;
    memory_mb: number;
  };
  capabilities: string[];
}

interface Timeline {
  start_time: Date;
  estimated_duration_hours: number;
  milestones: Milestone[];
  buffer_time_hours: number;
}

interface Milestone {
  id: string;
  name: string;
  description: string;
  target_date: Date;
  dependencies: string[];
  success_criteria: string[];
}

interface SuccessCriteria {
  primary_goals: string[];
  quality_thresholds: Record<string, number>;
  performance_targets: Record<string, number>;
  acceptable_failure_rate: number;
}

interface RiskAssessment {
  high_risks: Risk[];
  medium_risks: Risk[];
  low_risks: Risk[];
  mitigation_strategies: Record<string, string>;
}

interface Risk {
  id: string;
  description: string;
  probability: number;
  impact: number;
  mitigation: string[];
}

interface MissionConfig {
  domains: ('crypto' | 'ai')[];
  sources: string[];
  filters: Record<string, any>;
  notification_settings: NotificationSettings;
  quality_settings: QualitySettings;
  resource_limits: ResourceLimits;
}

interface NotificationSettings {
  channels: ('email' | 'slack' | 'webhook')[];
  urgency_levels: ('critical' | 'high' | 'medium' | 'low')[];
  frequency: 'real_time' | 'hourly' | 'daily';
}

interface QualitySettings {
  min_confidence_score: number;
  min_novelty_score: number;
  required_sources: number;
  fact_checking_enabled: boolean;
}

interface ResourceLimits {
  max_tokens: number;
  max_api_calls: number;
  max_compute_hours: number;
  max_agents: number;
  budget_usd: number;
}

interface ResourceUsage {
  tokens_used: number;
  api_calls_made: number;
  compute_hours_used: number;
  active_agents: number;
  cost_usd: number;
  last_updated: Date;
}

export class PlanManager extends EventEmitter {
  private db: Pool;
  private activeMissions: Map<string, Mission> = new Map();

  constructor(dbConfig: any) {
    super();
    this.db = new Pool(dbConfig);
  }

  async createMission(request: {
    name: string;
    prompt: string;
    user_id?: string;
    tenant_id?: string;
    config?: Partial<MissionConfig>;
  }): Promise<Mission> {
    const missionId = this.generateId();
    
    // Generate plan based on prompt
    const plan = await this.generatePlan(request.prompt, request.config);
    
    const mission: Mission = {
      id: missionId,
      name: request.name,
      prompt: request.prompt,
      status: 'planning',
      plan,
      config: this.buildDefaultConfig(request.config),
      resource_usage: this.initializeResourceUsage(),
      created_at: new Date(),
      updated_at: new Date(),
      user_id: request.user_id,
      tenant_id: request.tenant_id
    };

    // Save to database
    await this.saveMission(mission);
    
    this.activeMissions.set(missionId, mission);
    this.emit('missionCreated', mission);
    
    return mission;
  }

  private async generatePlan(prompt: string, config?: Partial<MissionConfig>): Promise<MissionPlan> {
    // Analyze prompt to extract objectives and requirements
    const analysis = this.analyzePrompt(prompt);
    
    const objectives = this.generateObjectives(analysis);
    const phases = this.generatePhases(objectives);
    const agents = this.generateAgentSpecs(analysis, objectives);
    const timeline = this.generateTimeline(phases);
    const successCriteria = this.generateSuccessCriteria(objectives);
    const riskAssessment = this.generateRiskAssessment(analysis, agents);

    return {
      objectives,
      phases,
      agents,
      timeline,
      success_criteria: successCriteria,
      risk_assessment: riskAssessment
    };
  }

  private analyzePrompt(prompt: string): {
    domain: 'crypto' | 'ai' | 'mixed';
    complexity: 'simple' | 'moderate' | 'complex';
    urgency: 'low' | 'medium' | 'high' | 'critical';
    scope: 'narrow' | 'medium' | 'broad';
    keywords: string[];
    entities: string[];
  } {
    const lowerPrompt = prompt.toLowerCase();
    
    // Determine domain
    const cryptoKeywords = ['bitcoin', 'ethereum', 'defi', 'nft', 'token', 'blockchain', 'crypto'];
    const aiKeywords = ['ai', 'ml', 'machine learning', 'neural', 'model', 'algorithm'];
    
    const hasCrypto = cryptoKeywords.some(kw => lowerPrompt.includes(kw));
    const hasAI = aiKeywords.some(kw => lowerPrompt.includes(kw));
    
    let domain: 'crypto' | 'ai' | 'mixed' = 'mixed';
    if (hasCrypto && !hasAI) domain = 'crypto';
    else if (hasAI && !hasCrypto) domain = 'ai';

    // Determine complexity
    const complexityIndicators = prompt.split(' ').length;
    let complexity: 'simple' | 'moderate' | 'complex' = 'simple';
    if (complexityIndicators > 50) complexity = 'moderate';
    if (complexityIndicators > 100) complexity = 'complex';

    // Determine urgency
    const urgencyKeywords = ['urgent', 'immediate', 'asap', 'critical', 'emergency'];
    const urgency = urgencyKeywords.some(kw => lowerPrompt.includes(kw)) ? 'high' : 'medium';

    return {
      domain,
      complexity,
      urgency,
      scope: 'medium',
      keywords: this.extractKeywords(prompt),
      entities: this.extractEntities(prompt)
    };
  }

  private extractKeywords(prompt: string): string[] {
    // Simple keyword extraction
    return prompt
      .toLowerCase()
      .split(/\W+/)
      .filter(word => word.length > 3)
      .slice(0, 10);
  }

  private extractEntities(prompt: string): string[] {
    // Simple entity extraction - in production would use NLP
    const entities: string[] = [];
    
    // Common crypto entities
    const cryptoRegex = /\b(bitcoin|ethereum|solana|polygon|chainlink|uniswap|aave|compound)\b/gi;
    const cryptoMatches = prompt.match(cryptoRegex);
    if (cryptoMatches) entities.push(...cryptoMatches);

    // Common AI entities  
    const aiRegex = /\b(openai|anthropic|google|microsoft|meta|nvidia|huggingface)\b/gi;
    const aiMatches = prompt.match(aiRegex);
    if (aiMatches) entities.push(...aiMatches);

    return [...new Set(entities)];
  }

  private generateObjectives(analysis: any): Objective[] {
    const baseObjectives: Objective[] = [
      {
        id: 'obj-1',
        description: 'Data Collection and Monitoring',
        priority: 'high',
        success_metrics: ['Data coverage > 90%', 'Latency < 5 seconds'],
        dependencies: [],
        assigned_agents: ['sensor']
      },
      {
        id: 'obj-2', 
        description: 'Event Analysis and Enrichment',
        priority: 'high',
        success_metrics: ['Accuracy > 85%', 'Processing time < 2 seconds'],
        dependencies: ['obj-1'],
        assigned_agents: ['analyzer', 'evaluator']
      },
      {
        id: 'obj-3',
        description: 'Intelligence Generation',
        priority: 'medium',
        success_metrics: ['Brief quality > 80%', 'Relevance score > 0.7'],
        dependencies: ['obj-2'],
        assigned_agents: ['curator']
      }
    ];

    // Customize based on analysis
    if (analysis.domain === 'crypto') {
      baseObjectives.push({
        id: 'obj-4',
        description: 'Crypto Market Monitoring',
        priority: 'high',
        success_metrics: ['Market coverage > 95%', 'Alert latency < 30 seconds'],
        dependencies: ['obj-1'],
        assigned_agents: ['solana_sensor', 'graph_analyzer']
      });
    }

    if (analysis.domain === 'ai') {
      baseObjectives.push({
        id: 'obj-5',
        description: 'AI Research Tracking',
        priority: 'medium',
        success_metrics: ['Paper coverage > 80%', 'Relevance filtering > 70%'],
        dependencies: ['obj-1'],
        assigned_agents: ['arxiv_sensor', 'github_sensor']
      });
    }

    return baseObjectives;
  }

  private generatePhases(objectives: Objective[]): Phase[] {
    return [
      {
        id: 'phase-1',
        name: 'Setup and Initialization',
        description: 'Deploy agents and establish monitoring infrastructure',
        duration_hours: 1,
        objectives: ['obj-1'],
        prerequisites: [],
        deliverables: ['Active monitoring streams', 'Agent health checks']
      },
      {
        id: 'phase-2',
        name: 'Data Collection',
        description: 'Begin data collection and initial processing',
        duration_hours: 4,
        objectives: ['obj-1', 'obj-2'],
        prerequisites: ['phase-1'],
        deliverables: ['Event streams', 'Initial analysis']
      },
      {
        id: 'phase-3',
        name: 'Intelligence Generation',
        description: 'Generate actionable intelligence from collected data',
        duration_hours: 8,
        objectives: ['obj-3'],
        prerequisites: ['phase-2'],
        deliverables: ['Intelligence briefs', 'Trend analysis']
      }
    ];
  }

  private generateAgentSpecs(analysis: any, objectives: Objective[]): AgentSpec[] {
    const specs: AgentSpec[] = [
      {
        type: 'evaluator',
        count: 1,
        config: { evi_weights: { freshness: 0.25, novelty: 0.25, impact: 0.3, confidence: 0.15, gap: 0.05 } },
        resource_allocation: { tokens: 5000, api_calls: 1000, compute_hours: 2, memory_mb: 512 },
        capabilities: ['evi_scoring', 'event_classification']
      },
      {
        type: 'curator',
        count: 1,
        config: { min_events_for_brief: 3, quality_threshold: 0.7 },
        resource_allocation: { tokens: 10000, api_calls: 500, compute_hours: 4, memory_mb: 1024 },
        capabilities: ['brief_generation', 'content_analysis']
      }
    ];

    if (analysis.domain === 'crypto' || analysis.domain === 'mixed') {
      specs.push({
        type: 'solana_sensor',
        count: 1,
        config: { rpc_endpoint: 'mainnet-beta', monitor_programs: [] },
        resource_allocation: { tokens: 2000, api_calls: 2000, compute_hours: 6, memory_mb: 256 },
        capabilities: ['blockchain_monitoring', 'mempool_analysis']
      });
    }

    if (analysis.domain === 'ai' || analysis.domain === 'mixed') {
      specs.push({
        type: 'github_sensor',
        count: 1,
        config: { repositories: [], keywords: analysis.keywords },
        resource_allocation: { tokens: 1000, api_calls: 1000, compute_hours: 2, memory_mb: 256 },
        capabilities: ['repository_monitoring', 'code_analysis']
      });
    }

    return specs;
  }

  private generateTimeline(phases: Phase[]): Timeline {
    const totalDuration = phases.reduce((sum, phase) => sum + phase.duration_hours, 0);
    
    return {
      start_time: new Date(),
      estimated_duration_hours: totalDuration,
      milestones: phases.map((phase, i) => ({
        id: `milestone-${i + 1}`,
        name: `${phase.name} Complete`,
        description: phase.description,
        target_date: new Date(Date.now() + (i + 1) * phase.duration_hours * 60 * 60 * 1000),
        dependencies: phase.prerequisites,
        success_criteria: phase.deliverables
      })),
      buffer_time_hours: Math.ceil(totalDuration * 0.2) // 20% buffer
    };
  }

  private generateSuccessCriteria(objectives: Objective[]): SuccessCriteria {
    return {
      primary_goals: objectives.map(obj => obj.description),
      quality_thresholds: {
        'data_accuracy': 0.85,
        'brief_relevance': 0.7,
        'processing_speed': 0.9
      },
      performance_targets: {
        'events_per_hour': 1000,
        'briefs_per_day': 10,
        'agent_uptime': 0.99
      },
      acceptable_failure_rate: 0.05
    };
  }

  private generateRiskAssessment(analysis: any, agents: AgentSpec[]): RiskAssessment {
    return {
      high_risks: [
        {
          id: 'risk-1',
          description: 'API rate limits exceeded',
          probability: 0.3,
          impact: 0.8,
          mitigation: ['Implement rate limiting', 'Use multiple API keys']
        }
      ],
      medium_risks: [
        {
          id: 'risk-2',
          description: 'Data source becomes unavailable',
          probability: 0.2,
          impact: 0.6,
          mitigation: ['Add backup data sources', 'Implement retry logic']
        }
      ],
      low_risks: [
        {
          id: 'risk-3',
          description: 'Minor performance degradation',
          probability: 0.4,
          impact: 0.3,
          mitigation: ['Monitor performance', 'Scale resources as needed']
        }
      ],
      mitigation_strategies: {
        'rate_limits': 'Implement exponential backoff and circuit breakers',
        'data_quality': 'Use multiple sources and cross-validation',
        'system_failure': 'Deploy across multiple regions with failover'
      }
    };
  }

  private buildDefaultConfig(config?: Partial<MissionConfig>): MissionConfig {
    return {
      domains: config?.domains || ['crypto', 'ai'],
      sources: config?.sources || ['github', 'arxiv', 'solana'],
      filters: config?.filters || {},
      notification_settings: {
        channels: ['webhook'],
        urgency_levels: ['critical', 'high'],
        frequency: 'real_time'
      },
      quality_settings: {
        min_confidence_score: 0.6,
        min_novelty_score: 0.5,
        required_sources: 2,
        fact_checking_enabled: true
      },
      resource_limits: {
        max_tokens: 50000,
        max_api_calls: 10000,
        max_compute_hours: 24,
        max_agents: 10,
        budget_usd: 100
      }
    };
  }

  private initializeResourceUsage(): ResourceUsage {
    return {
      tokens_used: 0,
      api_calls_made: 0,
      compute_hours_used: 0,
      active_agents: 0,
      cost_usd: 0,
      last_updated: new Date()
    };
  }

  private async saveMission(mission: Mission) {
    const query = `
      INSERT INTO missions (id, name, prompt, status, plan, config, resource_usage, user_id, tenant_id)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    `;
    
    await this.db.query(query, [
      mission.id,
      mission.name,
      mission.prompt,
      mission.status,
      JSON.stringify(mission.plan),
      JSON.stringify(mission.config),
      JSON.stringify(mission.resource_usage),
      mission.user_id,
      mission.tenant_id
    ]);
  }

  async updateMissionStatus(missionId: string, status: Mission['status']) {
    const mission = this.activeMissions.get(missionId);
    if (!mission) {
      throw new Error(`Mission ${missionId} not found`);
    }

    mission.status = status;
    mission.updated_at = new Date();
    
    if (status === 'completed' || status === 'failed') {
      mission.completed_at = new Date();
    }

    await this.db.query(
      'UPDATE missions SET status = $1, updated_at = $2, completed_at = $3 WHERE id = $4',
      [status, mission.updated_at, mission.completed_at, missionId]
    );

    this.emit('missionStatusChanged', { missionId, status });
  }

  async getMission(missionId: string): Promise<Mission | null> {
    if (this.activeMissions.has(missionId)) {
      return this.activeMissions.get(missionId)!;
    }

    const result = await this.db.query('SELECT * FROM missions WHERE id = $1', [missionId]);
    if (result.rows.length === 0) return null;

    const row = result.rows[0];
    return {
      id: row.id,
      name: row.name,
      prompt: row.prompt,
      status: row.status,
      plan: row.plan,
      config: row.config,
      resource_usage: row.resource_usage,
      created_at: row.created_at,
      updated_at: row.updated_at,
      completed_at: row.completed_at,
      user_id: row.user_id,
      tenant_id: row.tenant_id
    };
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }
}
