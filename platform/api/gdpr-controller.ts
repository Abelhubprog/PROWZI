import { Router } from 'express'
import { z } from 'zod'
import { Queue } from 'bullmq'
import { db, weaviate, s3 } from '@/lib/clients'

const erasureQueue = new Queue('gdpr-erasure')

const ErasureRequestSchema = z.object({
  reason: z.string(),
  confirmedBy: z.string().email(),
})

export const gdprRouter = Router()

gdprRouter.post('/tenants/:id/erase', async (req, res) => {
  const { id: tenantId } = req.params
  const request = ErasureRequestSchema.parse(req.body)

  // Verify authorization
  if (req.user?.role !== 'admin') {
    return res.status(403).json({ error: 'Unauthorized' })
  }

  try {
    // Create erasure record
    const [erasure] = await db('erasure_requests').insert({
      tenant_id: tenantId,
      requested_by: req.user.id,
      reason: request.reason,
      status: 'pending',
      created_at: new Date(),
    }).returning('*')

    // Queue erasure job
    await erasureQueue.add('erase-tenant', {
      erasureId: erasure.id,
      tenantId: tenantId,
    }, {
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 5000,
      },
    })

    res.json({
      erasureId: erasure.id,
      status: 'pending',
      message: 'Erasure request queued',
    })

  } catch (error) {
    console.error('Erasure request error:', error)
    res.status(500).json({ error: 'Failed to create erasure request' })
  }
})

gdprRouter.get('/erasures/:id/status', async (req, res) => {
  const { id } = req.params

  const [erasure] = await db('erasure_requests')
    .where({ id })
    .select('*')

  if (!erasure) {
    return res.status(404).json({ error: 'Erasure request not found' })
  }

  res.json({
    id: erasure.id,
    status: erasure.status,
    progress: erasure.progress,
    completedAt: erasure.completed_at,
    errors: erasure.errors,
  })
})

// Worker process
erasureQueue.process('erase-tenant', async (job) => {
  const { erasureId, tenantId } = job.data

  const updateProgress = async (step: string, percent: number) => {
    await db('erasure_requests')
      .where({ id: erasureId })
      .update({
        progress: { step, percent },
        updated_at: new Date(),
      })
  }

  try {
    // 1. Delete from Postgres
    await updateProgress('postgres', 0)

    await db.transaction(async (trx) => {
      // Delete in order of dependencies
      await trx('feedback').where({ tenant_id: tenantId }).delete()
      await trx('briefs').where({ tenant_id: tenantId }).delete()
      await trx('events').where({ tenant_id: tenantId }).delete()
      await trx('missions').where({ tenant_id: tenantId }).delete()
      await trx('users').where({ tenant_id: tenantId }).delete()
    })

    await updateProgress('postgres', 25)

    // 2. Delete from Weaviate
    await updateProgress('weaviate', 25)

    const weaviateClient = await weaviate.client()
    await weaviateClient.batch
      .objectsBatchDeleter()
      .withClassName('Event')
      .withWhere({
        path: ['tenantId'],
        operator: 'Equal',
        valueString: tenantId,
      })
      .do()

    await updateProgress('weaviate', 50)

    // 3. Delete from S3
    await updateProgress('s3', 50)

    const objects = await s3.listObjectsV2({
      Bucket: 'prowzi-data',
      Prefix: `tenants/${tenantId}/`,
    }).promise()

    if (objects.Contents?.length) {
      await s3.deleteObjects({
        Bucket: 'prowzi-data',
        Delete: {
          Objects: objects.Contents.map(obj => ({ Key: obj.Key! })),
        },
      }).promise()
    }

    await updateProgress('s3', 75)

    // 4. Delete from audit logs (anonymize)
    await updateProgress('audit', 75)

    await db('audit_logs')
      .where({ tenant_id: tenantId })
      .update({
        user_id: 'ERASED',
        ip_address: '0.0.0.0',
        user_agent: 'ERASED',
        details: db.raw("jsonb_set(details, '{personal_data}', '\"ERASED\"')")
      })

    await updateProgress('complete', 100)

    // Mark complete
    await db('erasure_requests')
      .where({ id: erasureId })
      .update({
        status: 'completed',
        completed_at: new Date(),
      })

  } catch (error) {
    console.error('Erasure error:', error)

    await db('erasure_requests')
      .where({ id: erasureId })
      .update({
        status: 'failed',
        errors: { message: error.message, stack: error.stack },
      })

    throw error
  }
})
