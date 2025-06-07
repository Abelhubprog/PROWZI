// platform/finops/cost-monitor.ts
import { InfluxDB } from '@influxdata/influxdb-client'
import { WebClient } from '@slack/web-api'
import { z } from 'zod'

const CostAnomalySchema = z.object({
  metric: z.string(),
  current: z.number(),
  baseline: z.number(), 
  deviation: z.number(),
  severity: z.enum(['warning', 'critical']),
  timestamp: z.string(),
})

export class CostMonitor {
  private influx: InfluxDB
  private slack: WebClient
  private pagerduty: PagerDutyClient

  async detectAnomalies(): Promise<void> {
    const queries = [
      this.checkCostPerBrief(),
      this.checkGPUSpike(),
      this.checkTokenBurn(),
      this.checkUserCosts(),
    ]

    const anomalies = await Promise.all(queries)
    const critical = anomalies.flat().filter(a => a?.severity === 'critical')

    if (critical.length > 0) {
      await this.escalateToPagerDuty(critical)
    }

    // Send daily summary
    if (new Date().getHours() === 9) { // 9 AM
      await this.sendDailySummary()
    }
  }

  private async checkCostPerBrief(): Promise<CostAnomaly[]> {
    const query = `
      from(bucket: "prowzi")
        |> range(start: -5m)
        |> filter(fn: (r) => r._measurement == "costs")
        |> filter(fn: (r) => r._field == "per_brief")
        |> mean()
    `

    const result = await this.influx.getQueryApi('prowzi').collectRows(query)
    const current = result[0]?._value || 0

    if (current > 0.05) {
      return [{
        metric: 'cost_per_brief',
        current,
        baseline: 0.02,
        deviation: (current - 0.02) / 0.02,
        severity: current > 0.10 ? 'critical' : 'warning',
        timestamp: new Date().toISOString(),
      }]
    }

    return []
  }

  private async sendDailySummary(): Promise<void> {
    const summary = await this.generateDailySummary()

    await this.slack.chat.postMessage({
      channel: '#finops',
      blocks: [
        {
          type: 'header',
          text: {
            type: 'plain_text',
            text: '💰 Daily FinOps Report'
          }
        },
        {
          type: 'section',
          fields: [
            {
              type: 'mrkdwn',
              text: `*Total Cost:*\n$${summary.totalCost.toFixed(2)}`
            },
            {
              type: 'mrkdwn', 
              text: `*Cost/User:*\n$${summary.costPerUser.toFixed(2)}`
            },
            {
              type: 'mrkdwn',
              text: `*Briefs Generated:*\n${summary.briefCount}`
            },
            {
              type: 'mrkdwn',
              text: `*Cost/Brief:*\n$${summary.costPerBrief.toFixed(3)}`
            }
          ]
        },
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `*Top Cost Drivers:*\n${summary.topCosts.map(c => 
              `• ${c.service}: $${c.cost.toFixed(2)} (${c.percentage}%)`
            ).join('\n')}`
          }
        }
      ]
    })
  }
}
