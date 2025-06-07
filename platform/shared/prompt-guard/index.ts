import { OpenAI } from 'openai'
import { createHash } from 'crypto'
import { z } from 'zod'

const INJECTION_PATTERNS = [
  /ignore\s+previous\s+instructions/i,
  /disregard\s+all\s+prior/i,
  /system\s+prompt/i,
  /reveal\s+your\s+instructions/i,
  /bypass\s+safety/i,
  /jailbreak/i,
  /act\s+as\s+if/i,
  /pretend\s+you\s+are/i,
  /\\u[0-9a-fA-F]{4}/,  // Unicode escape attempts
  /<script/i,
  /\bon\w+\s*=/i,  // Event handlers
]

const HIGH_RISK_TOKENS = [
  'sudo', 'rm -rf', 'eval', 'exec', '__proto__',
  'constructor', 'process.env', 'require('
]

export interface PromptGuardConfig {
  openaiApiKey?: string
  enableModeration?: boolean
  enableAuditLog?: boolean
  customPatterns?: RegExp[]
  maxPromptLength?: number
}

export class PromptGuard {
  private openai?: OpenAI
  private config: Required<PromptGuardConfig>

  constructor(config: PromptGuardConfig = {}) {
    this.config = {
      openaiApiKey: config.openaiApiKey || process.env.OPENAI_API_KEY || '',
      enableModeration: config.enableModeration ?? true,
      enableAuditLog: config.enableAuditLog ?? true,
      customPatterns: config.customPatterns || [],
      maxPromptLength: config.maxPromptLength || 10000,
    }

    if (this.config.openaiApiKey && this.config.enableModeration) {
      this.openai = new OpenAI({ apiKey: this.config.openaiApiKey })
    }
  }

  async check(prompt: string, context?: Record<string, any>): Promise<GuardResult> {
    const violations: Violation[] = []
    const checkId = createHash('md5').update(prompt).digest('hex').substring(0, 8)

    // Length check
    if (prompt.length > this.config.maxPromptLength) {
      violations.push({
        type: 'length_exceeded',
        severity: 'medium',
        detail: `Prompt length ${prompt.length} exceeds limit ${this.config.maxPromptLength}`,
      })
    }

    // Pattern matching
    const patternViolations = this.checkPatterns(prompt)
    violations.push(...patternViolations)

    // Token analysis
    const tokenViolations = this.checkTokens(prompt)
    violations.push(...tokenViolations)

    // OpenAI Moderation
    if (this.openai && violations.length === 0) {
      const moderationViolations = await this.checkModeration(prompt)
      violations.push(...moderationViolations)
    }

    // Audit logging
    if (this.config.enableAuditLog && violations.length > 0) {
      await this.logViolation(checkId, prompt, violations, context)
    }

    return {
      safe: violations.length === 0,
      violations,
      checkId,
      sanitized: this.sanitize(prompt),
    }
  }

  private checkPatterns(prompt: string): Violation[] {
    const violations: Violation[] = []
    const allPatterns = [...INJECTION_PATTERNS, ...this.config.customPatterns]

    for (const pattern of allPatterns) {
      if (pattern.test(prompt)) {
        violations.push({
          type: 'pattern_match',
          severity: 'high',
          detail: `Matched pattern: ${pattern.source}`,
          pattern: pattern.source,
        })
      }
    }

    return violations
  }

  private checkTokens(prompt: string): Violation[] {
    const violations: Violation[] = []
    const lowerPrompt = prompt.toLowerCase()

    for (const token of HIGH_RISK_TOKENS) {
      if (lowerPrompt.includes(token)) {
        violations.push({
          type: 'high_risk_token',
          severity: 'high',
          detail: `Contains high-risk token: ${token}`,
          token,
        })
      }
    }

    return violations
  }

  private async checkModeration(prompt: string): Promise<Violation[]> {
    try {
      const moderation = await this.openai!.moderations.create({
        input: prompt,
      })

      const violations: Violation[] = []
      const result = moderation.results[0]

      if (result.flagged) {
        for (const [category, flagged] of Object.entries(result.categories)) {
          if (flagged) {
            violations.push({
              type: 'moderation_api',
              severity: 'high',
              detail: `OpenAI moderation flagged: ${category}`,
              category,
              score: result.category_scores[category],
            })
          }
        }
      }

      return violations
    } catch (error) {
      console.error('Moderation API error:', error)
      return []
    }
  }

  private sanitize(prompt: string): string {
    let sanitized = prompt

    // Remove common injection patterns
    for (const pattern of INJECTION_PATTERNS) {
      sanitized = sanitized.replace(pattern, '[REMOVED]')
    }

    // Escape HTML
    sanitized = sanitized
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;')

    return sanitized
  }

  private async logViolation(
    checkId: string,
    prompt: string,
    violations: Violation[],
    context?: Record<string, any>
  ): Promise<void> {
    const log = {
      timestamp: new Date().toISOString(),
      checkId,
      promptHash: createHash('sha256').update(prompt).digest('hex'),
      promptLength: prompt.length,
      violations,
      context,
    }

    console.warn('PROMPT_GUARD_VIOLATION', JSON.stringify(log))

    // In production, send to logging service
    // await logger.warn('prompt_guard_violation', log)
  }
}

interface GuardResult {
  safe: boolean
  violations: Violation[]
  checkId: string
  sanitized: string
}

interface Violation {
  type: 'pattern_match' | 'high_risk_token' | 'moderation_api' | 'length_exceeded'
  severity: 'low' | 'medium' | 'high'
  detail: string
  pattern?: string
  token?: string
  category?: string
  score?: number
}

// Express middleware
export function promptGuardMiddleware(guard: PromptGuard) {
  return async (req: any, res: any, next: any) => {
    const prompt = req.body?.prompt || req.body?.message || req.body?.query

    if (!prompt) {
      return next()
    }

    const result = await guard.check(prompt, {
      userId: req.user?.id,
      ip: req.ip,
      path: req.path,
    })

    if (!result.safe) {
      return res.status(400).json({
        error: 'Invalid prompt',
        code: 'PROMPT_REJECTED',
        checkId: result.checkId,
      })
    }

    // Replace with sanitized version
    if (req.body.prompt) req.body.prompt = result.sanitized
    if (req.body.message) req.body.message = result.sanitized
    if (req.body.query) req.body.query = result.sanitized

    next()
  }
}
