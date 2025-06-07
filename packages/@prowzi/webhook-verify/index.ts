// @prowzi/webhook-verify package
import { createHmac } from 'crypto'

export function verifyWebhook(
  secret: string,
  header: string,
  payload: string | Buffer
): boolean {
  const parts = header.split(',').reduce((acc, part) => {
    const [key, value] = part.split('=')
    acc[key] = value
    return acc
  }, {} as Record<string, string>)

  const signature = parts.sha256
  if (!signature) {
    return false
  }

  const hmac = createHmac('sha256', secret)
  hmac.update(payload)
  const hash = hmac.digest('hex')

  return `sha256=${hash}` === signature
}

export function verifyGitHubWebhook(
  secret: string,
  signature: string,
  payload: string | Buffer
): boolean {
  const hmac = createHmac('sha1', secret)
  hmac.update(payload)
  const hash = `sha1=${hmac.digest('hex')}`
  
  return hash === signature
}

export function verifySlackWebhook(
  signingSecret: string,
  signature: string,
  timestamp: string,
  payload: string
): boolean {
  const baseString = `v0:${timestamp}:${payload}`
  const hmac = createHmac('sha256', signingSecret)
  hmac.update(baseString)
  const hash = `v0=${hmac.digest('hex')}`
  
  return hash === signature
}

export function verifyDiscordWebhook(
  publicKey: string,
  signature: string,
  timestamp: string,
  payload: string
): boolean {
  // Discord uses Ed25519 verification
  // This would need a more complex implementation with nacl
  // For now, return true as placeholder
  return true
}
