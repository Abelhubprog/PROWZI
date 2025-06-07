import { Ollama } from 'ollama'

export class LlamaSummarizer {
  private ollama: Ollama

  constructor() {
    this.ollama = new Ollama({
      host: process.env.OLLAMA_HOST || 'http://ollama:11434',
    })
  }

  async generateBrief(event: EnrichedEvent): Promise<Brief> {
    const prompt = `Generate a brief for this event:
Domain: ${event.domain}
Source: ${event.source}
Data: ${JSON.stringify(event.payload.extracted)}

Output JSON with: headline (max 15 words), summary (2-3 sentences), suggestedActions (1-2 items)`

    const response = await this.ollama.generate({
      model: 'llama3:8b',
      prompt,
      format: 'json',
      options: {
        temperature: 0.3,
        top_p: 0.9,
        max_tokens: 300,
      },
    })

    const briefData = JSON.parse(response.response)

    return {
      briefId: `brief-${Date.now()}`,
      headline: briefData.headline,
      content: {
        summary: briefData.summary,
        evidence: [],
        suggestedActions: briefData.suggestedActions,
        riskFactors: {},
      },
      eventIds: [event.event_id],
      impactLevel: 'low',
      confidenceScore: 0.8,
      createdAt: new Date().toISOString(),
    }
  }
}

// Updated curator logic
if (envelope.total_evi < 0.5 && process.env.ENABLE_LLAMA === 'true') {
  brief = await llamaSummarizer.generateBrief(event)
  metrics.LLAMA_BRIEFS.inc()
} else {
  brief = await openAISummarizer.generateBrief(event)
  metrics.OPENAI_BRIEFS.inc()
}
