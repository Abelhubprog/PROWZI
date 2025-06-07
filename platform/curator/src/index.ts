import { Redis } from 'ioredis'
import { createHash } from 'crypto'
import { WebSocketServer } from 'ws'
import { z } from 'zod'
import { EVIEnvelope, Brief, BriefSchema } from '@prowzi/messages'
import { createNatsConsumer, createNatsPublisher } from '@prowzi/nats'
import { chooseModel } from '../../shared/modelRouter'
import { call, LLMRequestParams, LLMResponse, LLMAPIError } from '../../shared/llmClients'

// Cache TTL in seconds (3 hours)
const CACHE_TTL = 3 * 60 * 60

// Templates for different models
const BRIEF_TEMPLATES = {
  default: `
Generate a concise intelligence brief for the following event:

Domain: {domain}
Source: {source}
Impact: {impact}
Key Data: {data}

Requirements:
- Headline: Max 15 words, action-oriented
- Summary: 2-3 sentences explaining what happened and why it matters
- Evidence: List 2-3 specific data points
- Suggested Actions: 1-2 actionable next steps
- Risk Factors: If applicable, highlight risks

Format as JSON matching the Brief schema.
`,
  claude: `
You are an expert intelligence analyst. Create a detailed brief for this high-impact event:

Domain: {domain}
Source: {source}
Impact Score: {impact} (Very High)
Event Data: {data}

Your brief must include:
- Headline: Precise, action-oriented (max 15 words)
- Summary: Comprehensive yet concise explanation (2-3 sentences)
- Evidence: 3-4 key data points with source attribution
- Risk Analysis: Potential implications and second-order effects
- Suggested Actions: Strategic recommendations (2-3 items)
- Confidence Level: Assessment of information reliability

Return as valid JSON matching the Brief schema.
`,
  minimal: `
Create a minimal intelligence brief with just the essential information:

Domain: {domain}
Source: {source}
Event: {data}

Format as JSON with:
- headline: Short title describing the event
- summary: Brief factual description
- evidence: List of data points
- suggestedActions: Basic next steps

Return valid JSON only.
`
}

export class CuratorService {
  private redis: Redis
  private wss: WebSocketServer
  private briefCounter = 0
  private processedEvents = new Set<string>()

  constructor() {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: 6379,
    })

    this.wss = new WebSocketServer({ port: 8082 })
  }

  async start() {
    console.log('Starting Curator Service...')

    // Subscribe to all bands
    const consumer = await createNatsConsumer([
      'evaluator.instant',
      'evaluator.same-day',
      'evaluator.weekly',
      'evaluator.archive'
    ])

    // Create publishers
    const briefPublisher = await createNatsPublisher('briefs')
    
    // Process incoming events
    await consumer.subscribe(async (envelope: EVIEnvelope) => {
      try {
        // Skip already processed events
        const eventId = envelope.event.id
        if (this.processedEvents.has(eventId)) {
          console.log(`Skipping already processed event: ${eventId}`)
          return
        }
        this.processedEvents.add(eventId)
        
        // Keep set size manageable
        if (this.processedEvents.size > 10000) {
          this.processedEvents.clear()
        }

        console.log(`Processing event: ${eventId}, band: ${envelope.band}, impact: ${envelope.scores.impact}`)
        
        // Check cache first
        const cacheKey = this.generateCacheKey(envelope)
        const cachedBrief = await this.checkCache(cacheKey)
        
        if (cachedBrief) {
          console.log(`Cache hit for event: ${eventId}`)
          await this.publishBrief(cachedBrief, briefPublisher)
          return
        }

        // Generate brief using appropriate model
        const brief = await this.generateBrief(envelope)
        
        // Cache the brief
        await this.cacheResult(cacheKey, brief)
        
        // Publish and broadcast
        await this.publishBrief(brief, briefPublisher)
        
        this.briefCounter++
        console.log(`Processed brief #${this.briefCounter}: ${brief.headline}`)
      } catch (error) {
        console.error('Error processing event:', error)
        
        // If LLM API error and model was Llama, try minimal template fallback
        if (error instanceof LLMAPIError && error.message.includes('Llama')) {
          try {
            console.log('Attempting minimal brief fallback...')
            const minimalBrief = await this.generateMinimalBrief(envelope)
            await this.publishBrief(minimalBrief, briefPublisher)
          } catch (fallbackError) {
            console.error('Fallback brief generation failed:', fallbackError)
          }
        }
      }
    })

    console.log('Curator Service started')
  }

  /**
   * Generates a SHA-256 hash as cache key for an envelope
   */
  private generateCacheKey(envelope: EVIEnvelope): string {
    // Include only the parts that determine the content of the brief
    const relevantData = {
      eventId: envelope.event.id,
      domain: envelope.event.domain,
      source: envelope.event.source,
      data: envelope.event.data,
      band: envelope.band,
      impact: envelope.scores.impact
    }
    
    return createHash('sha256')
      .update(JSON.stringify(relevantData))
      .digest('hex')
  }

  /**
   * Checks Redis cache for existing brief
   */
  private async checkCache(key: string): Promise<Brief | null> {
    const cached = await this.redis.get(`brief:${key}`)
    if (cached) {
      try {
        const brief = JSON.parse(cached)
        return BriefSchema.parse(brief)
      } catch (error) {
        console.warn('Invalid cached brief, ignoring', error)
        return null
      }
    }
    return null
  }

  /**
   * Caches a brief with TTL
   */
  private async cacheResult(key: string, brief: Brief): Promise<void> {
    await this.redis.set(
      `brief:${key}`,
      JSON.stringify(brief),
      'EX',
      CACHE_TTL
    )
  }

  /**
   * Generates a brief using the appropriate model based on band and impact
   */
  private async generateBrief(envelope: EVIEnvelope): Promise<Brief> {
    const { band, scores } = envelope
    const impact = scores.impact
    
    // Get tenant overrides from Redis if available
    const tenantId = envelope.event.tenantId
    let tenantOverrides: Record<string, string> | undefined
    
    if (tenantId) {
      const overridesStr = await this.redis.get(`tenant:${tenantId}:model_overrides`)
      if (overridesStr) {
        try {
          tenantOverrides = JSON.parse(overridesStr)
        } catch (error) {
          console.warn(`Invalid tenant overrides for ${tenantId}`, error)
        }
      }
    }
    
    // Choose the appropriate model based on band and impact
    const model = chooseModel('summarise', band as any, impact, tenantOverrides)
    console.log(`Selected model for brief generation: ${model.provider}/${model.model}`)
    
    // Select the appropriate template based on model
    let templateKey = 'default'
    if (model.provider === 'anthropic') {
      templateKey = 'claude'
    }
    
    const template = BRIEF_TEMPLATES[templateKey]
      .replace('{domain}', envelope.event.domain)
      .replace('{source}', envelope.event.source)
      .replace('{impact}', scores.impact.toString())
      .replace('{data}', JSON.stringify(envelope.event.data, null, 2))
    
    // Call the selected model
    const params: LLMRequestParams = {
      prompt: template,
      temperature: 0.4,
      maxTokens: model.maxTokens
    }
    
    try {
      const response = await call(model, params)
      
      // Parse and validate the response
      const briefData = JSON.parse(response.text)
      
      // Add metadata
      const brief: Brief = {
        ...briefData,
        id: `brief-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
        createdAt: new Date().toISOString(),
        eventId: envelope.event.id,
        band: envelope.band,
        domain: envelope.event.domain,
        source: envelope.event.source,
        modelUsed: `${model.provider}/${model.model}`,
        tenantId: envelope.event.tenantId,
        expiresAt: this.calculateExpiryTime(envelope.band)
      }
      
      return BriefSchema.parse(brief)
    } catch (error) {
      if (error instanceof LLMAPIError) {
        // If the selected model is Llama and it fails, we'll handle this in the main try/catch
        if (model.provider === 'llama-local') {
          throw error
        }
        
        // For other models, fall back to Llama
        console.warn(`${model.provider} API error, falling back to Llama`, error)
        const llamaModel = chooseModel('summarise', 'weekly', 0, undefined)
        
        const fallbackParams: LLMRequestParams = {
          prompt: template,
          temperature: 0.4,
          maxTokens: llamaModel.maxTokens
        }
        
        const fallbackResponse = await call(llamaModel, fallbackParams)
        const briefData = JSON.parse(fallbackResponse.text)
        
        const brief: Brief = {
          ...briefData,
          id: `brief-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
          createdAt: new Date().toISOString(),
          eventId: envelope.event.id,
          band: envelope.band,
          domain: envelope.event.domain,
          source: envelope.event.source,
          modelUsed: `${llamaModel.provider}/${llamaModel.model} (fallback)`,
          tenantId: envelope.event.tenantId,
          expiresAt: this.calculateExpiryTime(envelope.band)
        }
        
        return BriefSchema.parse(brief)
      }
      
      // Re-throw other errors
      throw error
    }
  }

  /**
   * Generates a minimal brief when all LLM calls fail
   */
  private async generateMinimalBrief(envelope: EVIEnvelope): Promise<Brief> {
    const template = BRIEF_TEMPLATES.minimal
      .replace('{domain}', envelope.event.domain)
      .replace('{source}', envelope.event.source)
      .replace('{data}', JSON.stringify(envelope.event.data, null, 2))
    
    // Try to extract basic information directly from the event
    const eventData = envelope.event.data
    
    // Generate a basic brief without LLM
    const headline = eventData.title || 
                    eventData.name || 
                    `${envelope.event.domain} update from ${envelope.event.source}`
    
    const summary = eventData.description || 
                   eventData.summary || 
                   `New ${envelope.event.domain} event detected from ${envelope.event.source}`
    
    // Extract evidence points
    const evidence = []
    if (eventData.url) evidence.push(`Source URL: ${eventData.url}`)
    if (eventData.timestamp) evidence.push(`Timestamp: ${eventData.timestamp}`)
    
    // For different domains, extract relevant data
    if (envelope.event.domain === 'crypto') {
      if (eventData.token) evidence.push(`Token: ${eventData.token}`)
      if (eventData.price) evidence.push(`Price: ${eventData.price}`)
      if (eventData.change) evidence.push(`Change: ${eventData.change}%`)
    } else if (envelope.event.domain === 'github') {
      if (eventData.repo) evidence.push(`Repository: ${eventData.repo}`)
      if (eventData.author) evidence.push(`Author: ${eventData.author}`)
      if (eventData.action) evidence.push(`Action: ${eventData.action}`)
    }
    
    const brief: Brief = {
      id: `brief-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      headline: headline.substring(0, 100), // Ensure reasonable length
      summary,
      evidence,
      suggestedActions: ['Review raw event data for more details'],
      createdAt: new Date().toISOString(),
      eventId: envelope.event.id,
      band: envelope.band,
      domain: envelope.event.domain,
      source: envelope.event.source,
      modelUsed: 'minimal-template-fallback',
      tenantId: envelope.event.tenantId,
      expiresAt: this.calculateExpiryTime(envelope.band)
    }
    
    return BriefSchema.parse(brief)
  }

  /**
   * Calculates expiry time based on band
   */
  private calculateExpiryTime(band: string): string {
    const now = new Date()
    
    switch (band) {
      case 'instant':
        // Expires in 24 hours
        now.setHours(now.getHours() + 24)
        break
      case 'same-day':
        // Expires in 3 days
        now.setDate(now.getDate() + 3)
        break
      case 'weekly':
        // Expires in 14 days
        now.setDate(now.getDate() + 14)
        break
      case 'archive':
        // Expires in 30 days
        now.setDate(now.getDate() + 30)
        break
      default:
        // Default: 7 days
        now.setDate(now.getDate() + 7)
    }
    
    return now.toISOString()
  }

  /**
   * Publishes a brief to NATS and Redis, and broadcasts to WebSocket clients
   */
  private async publishBrief(brief: Brief, publisher: any): Promise<void> {
    // Publish to NATS
    await publisher.publish(brief)
    
    // Store in Redis stream
    await this.redis.xadd(
      'briefs',
      '*',
      'data',
      JSON.stringify(brief)
    )
    
    // Broadcast to WebSocket clients
    this.wss.clients.forEach(client => {
      if (client.readyState === 1) { // OPEN
        client.send(JSON.stringify({
          type: 'new_brief',
          data: brief
        }))
      }
    })
  }
}
