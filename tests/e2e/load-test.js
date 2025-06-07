import http from 'k6/http'
import { check, sleep } from 'k6'
import { Rate } from 'k6/metrics'

const errorRate = new Rate('errors')

export const options = {
  stages: [
    { duration: '30s', target: 100 },
    { duration: '4m', target: 100 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'],
    errors: ['rate<0.005'],
  },
}

export default function() {
  // Generate synthetic event
  const event = {
    event_id: `test-${__VU}-${__ITER}`,
    domain: Math.random() > 0.5 ? 'crypto' : 'ai',
    source: 'test_sensor',
    topic_hints: ['test'],
    payload: {
      raw: { test: true },
      extracted: {
        entities: [],
        metrics: { value: Math.random() * 1000 },
      },
      embeddings: Array(768).fill(0).map(() => Math.random()),
    },
    metadata: {
      content_hash: `hash-${Date.now()}`,
      language: 'en',
      processing_time_ms: 10,
    },
  }

  const res = http.post('http://localhost:8080/events', JSON.stringify(event), {
    headers: { 'Content-Type': 'application/json' },
  })

  const success = check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 1s': (r) => r.timings.duration < 1000,
  })

  errorRate.add(!success)

  sleep(0.01) // 100 RPS per VU
}
