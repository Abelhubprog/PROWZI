// platform/compliance/gdpr-service.ts
import { Queue, Worker } from 'bullmq'
import { db, weaviate, s3, pulsar } from '@/lib/clients'

export class GDPRErasureService {
  private queue: Queue
  private worker: Worker

  constructor() {
    this.queue = new Queue('gdpr-erasure', {
      connection: {
        host: process.env.REDIS_HOST,
        port: 6379,
      },
    })

    this.worker = new Worker('gdpr-erasure', this.processErasure.bind(this), {
      connection: {
        host: process.env.REDIS_HOST,
        port: 6379,
      },
      concurrency: 1, // Process one at a time for safety
    })
  }

  async requestErasure(tenantId: string, requestedBy: string): Promise<string> {
    const erasureId = `erasure-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

    // Create audit record
    await db('erasure_requests').insert({
      id: erasureId,
      tenant_id: tenantId,
      requested_by: requestedBy,
      status: 'pending',
      created_at: new Date(),
    })

    // Queue the job
    await this.queue.add('erase', {
      erasureId,
      tenantId,
    }, {
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 5000,
      },
    })

    return erasureId
  }

  private async processErasure(job: Job): Promise<void> {
    const { erasureId, tenantId } = job.data
    const steps = [
      { name: 'postgres', fn: () => this.erasePostgres(tenantId) },
      { name: 'weaviate', fn: () => this.eraseVectorDB(tenantId) },
      { name: 's3', fn: () => this.eraseObjectStorage(tenantId) },
      { name: 'pulsar', fn: () => this.eraseEventStream(tenantId) },
      { name: 'neo4j', fn: () => this.eraseGraph(tenantId) },
      { name: 'audit', fn: () => this.anonymizeAuditLogs(tenantId) },
    ]

    let completedSteps = 0

    try {
      for (const step of steps) {
        await this.updateProgress(erasureId, step.name, completedSteps / steps.length * 100)
        await step.fn()
        completedSteps++
      }

      await this.completeErasure(erasureId)

    } catch (error) {
      await this.failErasure(erasureId, error)
      throw error
    }
  }

  private async erasePostgres(tenantId: string): Promise<void> {
    await db.transaction(async (trx) => {
      // Delete in dependency order
      const tables = [
        'feedback',
        'brief_events',
        'briefs', 
        'event_embeddings',
        'events',
        'mission_agents',
        'missions',
        'user_preferences',
        'users',
      ]

      for (const table of tables) {
        const deleted = await trx(table)
          .where('tenant_id', tenantId)
          .delete()

        console.log(`Deleted ${deleted} rows from ${table}`)
      }

      // Delete tenant record itself
      await trx('tenants').where('id', tenantId).delete()
    })
  }

  private async eraseVectorDB(tenantId: string): Promise<void> {
    const client = await weaviate.client()

    // Delete all objects for tenant
    await client.batch
      .objectsBatchDeleter()
      .withClassName('Event')
      .withWhere({
        path: ['tenantId'],
        operator: 'Equal',
        valueString: tenantId,
      })
      .do()

    await client.batch
      .objectsBatchDeleter()
      .withClassName('Brief')
      .withWhere({
        path: ['tenantId'],
        operator: 'Equal',
        valueString: tenantId,
      })
      .do()
  }

  private async eraseObjectStorage(tenantId: string): Promise<void> {
    const prefix = `tenants/${tenantId}/`

    // List all objects
    let continuationToken: string | undefined

    do {
      const response = await s3.listObjectsV2({
        Bucket: 'prowzi-data',
        Prefix: prefix,
        ContinuationToken: continuationToken,
      }).promise()

      if (response.Contents && response.Contents.length > 0) {
        // Delete in batches of 1000
        const deleteParams = {
          Bucket: 'prowzi-data',
          Delete: {
            Objects: response.Contents.map(obj => ({ Key: obj.Key! })),
          },
        }

        await s3.deleteObjects(deleteParams).promise()
      }

      continuationToken = response.NextContinuationToken
    } while (continuationToken)
  }

  private async anonymizeAuditLogs(tenantId: string): Promise<void> {
    // Don't delete audit logs, just anonymize PII
    await db('audit_logs')
      .where('tenant_id', tenantId)
      .update({
        user_id: 'ERASED',
        user_email: 'erased@prowzi.io',
        ip_address: '0.0.0.0',
        user_agent: 'ERASED',
        details: db.raw(`
          jsonb_set(
            jsonb_set(details, '{personal_data}', '"ERASED"'),
            '{user_info}', 
            '{"erased": true}'
          )
        `),
      })
  }
}
