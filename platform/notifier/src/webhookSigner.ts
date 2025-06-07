/**
 * Webhook Signer Module
 * 
 * Provides HMAC-SHA256 signing and verification for outgoing webhook payloads.
 * Uses the header format: X-Prowzi-Signature: t=<unix-seconds>,v1=<hex-digest>
 */

import * as crypto from 'crypto';

/**
 * Error thrown when webhook signature operations fail
 */
export class WebhookSignatureError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebhookSignatureError';
  }
}

/**
 * Parsed signature components
 */
export interface ParsedSignature {
  timestamp: number;
  signatures: { [key: string]: string };
}

/**
 * Webhook channel types supported by the notifier
 */
export type WebhookChannel = 'telegram' | 'discord' | 'slack' | 'generic';

/**
 * Gets the HMAC secret for a specific channel
 * 
 * @param channel The webhook channel
 * @param tenantId Optional tenant ID for multi-tenant deployments
 * @returns The HMAC secret for the channel
 * @throws {WebhookSignatureError} If the secret is not configured
 */
export function getChannelSecret(channel: WebhookChannel, tenantId?: string): string {
  // Format: NOTIFIER_HMAC_<CHANNEL> or NOTIFIER_HMAC_<TENANT>_<CHANNEL>
  const envVar = tenantId 
    ? `NOTIFIER_HMAC_${tenantId.toUpperCase()}_${channel.toUpperCase()}`
    : `NOTIFIER_HMAC_${channel.toUpperCase()}`;
  
  const secret = process.env[envVar];
  
  if (!secret) {
    // Fall back to the global secret if tenant-specific not found
    const globalSecret = process.env[`NOTIFIER_HMAC_${channel.toUpperCase()}`];
    
    if (!globalSecret) {
      throw new WebhookSignatureError(`HMAC secret not configured for channel: ${channel}`);
    }
    
    return globalSecret;
  }
  
  return secret;
}

/**
 * Signs a webhook payload using HMAC-SHA256
 * 
 * @param payload The raw payload to sign (Buffer or string)
 * @param channel The webhook channel
 * @param tenantId Optional tenant ID for multi-tenant deployments
 * @returns The signature header value in format: t=<timestamp>,v1=<signature>
 */
export function signPayload(
  payload: Buffer | string, 
  channel: WebhookChannel,
  tenantId?: string
): string {
  // Get the secret for this channel
  const secret = getChannelSecret(channel, tenantId);
  
  // Convert string payload to Buffer if needed
  const payloadBuffer = Buffer.isBuffer(payload) ? payload : Buffer.from(payload);
  
  // Get current timestamp in seconds
  const timestamp = Math.floor(Date.now() / 1000);
  
  // Create HMAC using SHA-256
  const hmac = crypto.createHmac('sha256', secret);
  hmac.update(payloadBuffer);
  
  // Get hex digest
  const signature = hmac.digest('hex');
  
  // Return formatted signature header
  return `t=${timestamp},v1=${signature}`;
}

/**
 * Parses a signature header string into its components
 * 
 * @param signatureHeader The signature header string
 * @returns Parsed signature with timestamp and signature values
 * @throws {WebhookSignatureError} If the signature header is invalid
 */
export function parseSignatureHeader(signatureHeader: string): ParsedSignature {
  if (!signatureHeader || typeof signatureHeader !== 'string') {
    throw new WebhookSignatureError('Invalid signature header');
  }
  
  const result: ParsedSignature = {
    timestamp: 0,
    signatures: {}
  };
  
  // Split the header by commas
  const pairs = signatureHeader.split(',');
  
  for (const pair of pairs) {
    const [key, value] = pair.split('=');
    
    if (!key || !value) {
      throw new WebhookSignatureError(`Invalid signature format: ${pair}`);
    }
    
    if (key === 't') {
      const timestamp = parseInt(value, 10);
      
      if (isNaN(timestamp)) {
        throw new WebhookSignatureError(`Invalid timestamp: ${value}`);
      }
      
      result.timestamp = timestamp;
    } else {
      // Store signature with its version (e.g., 'v1')
      result.signatures[key] = value;
    }
  }
  
  // Ensure we have a timestamp and at least one signature
  if (result.timestamp === 0) {
    throw new WebhookSignatureError('Missing timestamp in signature header');
  }
  
  if (Object.keys(result.signatures).length === 0) {
    throw new WebhookSignatureError('Missing signature value in header');
  }
  
  return result;
}

/**
 * Verifies a webhook signature against a payload
 * 
 * @param payload The raw payload (Buffer or string)
 * @param signatureHeader The signature header from the request
 * @param channel The webhook channel
 * @param tenantId Optional tenant ID for multi-tenant deployments
 * @param tolerance Optional tolerance in seconds (default: 300s = 5min)
 * @returns True if the signature is valid, false otherwise
 */
export function verifySignature(
  payload: Buffer | string,
  signatureHeader: string,
  channel: WebhookChannel,
  tenantId?: string,
  tolerance: number = 300
): boolean {
  try {
    // Get the secret for this channel
    const secret = getChannelSecret(channel, tenantId);
    
    // Parse the signature header
    const parsed = parseSignatureHeader(signatureHeader);
    
    // Check timestamp tolerance
    const now = Math.floor(Date.now() / 1000);
    const diff = Math.abs(now - parsed.timestamp);
    
    if (diff > tolerance) {
      console.warn(`Webhook signature timestamp too old: ${diff}s`);
      return false;
    }
    
    // Convert string payload to Buffer if needed
    const payloadBuffer = Buffer.isBuffer(payload) ? payload : Buffer.from(payload);
    
    // Create HMAC using SHA-256
    const hmac = crypto.createHmac('sha256', secret);
    hmac.update(payloadBuffer);
    
    // Get hex digest
    const expectedSignature = hmac.digest('hex');
    
    // Check if any of the provided signatures match
    // (supporting multiple signature versions)
    for (const [version, signature] of Object.entries(parsed.signatures)) {
      if (crypto.timingSafeEqual(
        Buffer.from(signature),
        Buffer.from(expectedSignature)
      )) {
        return true;
      }
    }
    
    return false;
  } catch (error) {
    console.error('Signature verification error:', error);
    return false;
  }
}

/**
 * Creates verification middleware for Express.js
 * 
 * @param channel The webhook channel
 * @param tenantId Optional tenant ID for multi-tenant deployments
 * @param tolerance Optional tolerance in seconds (default: 300s = 5min)
 * @returns Express middleware function
 */
export function createVerificationMiddleware(
  channel: WebhookChannel,
  tenantId?: string,
  tolerance: number = 300
) {
  return (req: any, res: any, next: any) => {
    try {
      const signatureHeader = req.headers['x-prowzi-signature'];
      
      if (!signatureHeader) {
        return res.status(401).json({ error: 'Missing signature header' });
      }
      
      // Get raw body from the request
      // Note: requires bodyParser with { verify: (req, res, buf) => { req.rawBody = buf; } }
      const rawBody = req.rawBody;
      
      if (!rawBody) {
        return res.status(400).json({ error: 'Missing raw request body' });
      }
      
      const isValid = verifySignature(
        rawBody,
        signatureHeader,
        channel,
        tenantId,
        tolerance
      );
      
      if (!isValid) {
        return res.status(401).json({ error: 'Invalid signature' });
      }
      
      next();
    } catch (error) {
      console.error('Webhook verification middleware error:', error);
      return res.status(500).json({ error: 'Signature verification failed' });
    }
  };
}

/**
 * Helper for TypeScript clients to verify Prowzi webhook signatures
 */
export const ProwziWebhookVerifier = {
  /**
   * Verify a webhook signature
   * 
   * @param payload The raw payload (string or Buffer)
   * @param signatureHeader The X-Prowzi-Signature header value
   * @param secret The HMAC secret
   * @param tolerance Optional tolerance in seconds (default: 300s = 5min)
   * @returns True if signature is valid, false otherwise
   */
  verify(
    payload: string | Buffer,
    signatureHeader: string,
    secret: string,
    tolerance: number = 300
  ): boolean {
    try {
      // Parse the signature header
      const parsed = parseSignatureHeader(signatureHeader);
      
      // Check timestamp tolerance
      const now = Math.floor(Date.now() / 1000);
      const diff = Math.abs(now - parsed.timestamp);
      
      if (diff > tolerance) {
        return false;
      }
      
      // Convert string payload to Buffer if needed
      const payloadBuffer = Buffer.isBuffer(payload) ? payload : Buffer.from(payload);
      
      // Create HMAC using SHA-256
      const hmac = crypto.createHmac('sha256', secret);
      hmac.update(payloadBuffer);
      
      // Get hex digest
      const expectedSignature = hmac.digest('hex');
      
      // Check if v1 signature matches
      if (parsed.signatures.v1) {
        return crypto.timingSafeEqual(
          Buffer.from(parsed.signatures.v1),
          Buffer.from(expectedSignature)
        );
      }
      
      return false;
    } catch (error) {
      console.error('Signature verification error:', error);
      return false;
    }
  }
};
