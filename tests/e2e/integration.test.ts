// tests/e2e/integration.test.ts
import { ProwziClient } from '@prowzi/sdk'
import { expect } from '@playwright/test'

describe('Prowzi E2E Integration', () => {
  let client: ProwziClient
  let authToken: string

  beforeAll(async () => {
    // Authenticate
    const auth = new ProwziAuth(process.env.API_URL!)
    const tokens = await auth.authenticateEthereum(mockWallet)
    authToken = tokens.access_token

    client = new ProwziClient({
      apiKey: authToken,
      baseUrl: process.env.API_URL!,
    })
  })

  test('complete mission flow with RLS', async () => {
    // Create mission
    const mission = await client.createMission({
      prompt: 'Track new Solana token launches',
      constraints: {
        maxDuration: 1,
        tokenBudget: 1000,
      },
    })

    expect(mission.status).toBe('planning')

    // Wait for first brief
    const brief = await waitForBrief(client, {
      missionId: mission.id,
      timeout: 30000,
    })

    expect(brief.impactLevel).toBeDefined()
    expect(brief.content.summary).toBeTruthy()

    // Verify RLS - try to access another tenant's data
    const otherClient = new ProwziClient({
      apiKey: otherTenantToken,
      baseUrl: process.env.API_URL!,
    })

    await expect(
      otherClient.getMission(mission.id)
    ).rejects.toThrow('404')

    // Test webhook signature
    const webhook = await setupWebhook()
    const notification = await webhook.waitForNotification()

    const isValid = verifyWebhook(
      process.env.WEBHOOK_SECRET!,
      notification.headers['x-prowzi-signature'],
      notification.body
    )

    expect(isValid).toBe(true)

    // Test plan hot-swap
    const newPlan = await client.updateMissionPlan(mission.id, {
      dag: modifiedDAG,
    })

    expect(newPlan.version).toBe(2)

    // Verify no events lost during swap
    const metrics = await client.getMissionMetrics(mission.id)
    expect(metrics.eventsLost).toBe(0)
  })

  test('GDPR erasure compliance', async () => {
    const tenantId = 'test-tenant-gdpr'

    // Seed data
    await seedTenantData(tenantId)

    // Request erasure
    const erasureId = await client.requestDataErasure({
      reason: 'User request',
      confirmedBy: 'test@example.com',
    })

    // Poll for completion
    await waitForErasureCompletion(erasureId)

    // Verify data is gone
    const remainingData = await verifyNoDataRemains(tenantId)
    expect(remainingData).toBe(0)

    // Verify audit logs are anonymized
    const auditLogs = await getAuditLogs(tenantId)
    expect(auditLogs.every(log => log.user_id === 'ERASED')).toBe(true)
  })

  test('cost controls and alerts', async () => {
    // Simulate high-cost scenario
    const mission = await client.createMission({
      prompt: 'Analyze all Ethereum transactions', // Expensive
      constraints: {
        tokenBudget: 100, // Low budget
      },
    })

    // Should get throttled
    await expect(
      waitForEvent(client, {
        type: 'mission.throttled',
        missionId: mission.id,
        timeout: 60000,
      })
    ).resolves.toBeTruthy()

    // Check cost metrics
    const costs = await client.getMissionCosts(mission.id)
    expect(costs.totalCost).toBeLessThan(0.10) // Under $0.10

    // Verify alert was triggered
    const alerts = await getTriggeredAlerts()
    expect(alerts).toContainEqual(
      expect.objectContaining({
        alertname: 'BudgetNearExhaustion',
      })
    )
  })
})
