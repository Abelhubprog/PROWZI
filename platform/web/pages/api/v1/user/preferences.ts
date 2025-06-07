import type { NextApiRequest, NextApiResponse } from 'next';
import { z } from 'zod';

// Schema for user preferences validation
const UserPreferencesSchema = z.object({
  // Domain toggles for filtering content
  domains: z.object({
    crypto: z.boolean().default(true),
    ai: z.boolean().default(true),
    defi: z.boolean().optional(),
    nft: z.boolean().optional(),
    security: z.boolean().optional(),
    research: z.boolean().optional(),
  }).optional(),
  
  // Alert cadence preferences
  alertCadence: z.object({
    instant: z.boolean().default(true),
    sameDay: z.boolean().default(true),
    weekly: z.boolean().default(true),
    maxAlertsPerHour: z.number().min(0).max(100).optional(),
    quietHours: z.object({
      enabled: z.boolean().default(false),
      start: z.string().regex(/^([01]\d|2[0-3]):([0-5]\d)$/).optional(), // HH:MM format
      end: z.string().regex(/^([01]\d|2[0-3]):([0-5]\d)$/).optional(),   // HH:MM format
      timezone: z.string().optional(),
    }).optional(),
  }).optional(),
  
  // Notification channels
  notificationChannels: z.object({
    email: z.object({
      enabled: z.boolean().default(true),
      address: z.string().email().optional(),
      digest: z.enum(['instant', 'hourly', 'daily', 'weekly']).optional(),
    }).optional(),
    telegram: z.object({
      enabled: z.boolean().default(false),
      chatId: z.string().optional(),
    }).optional(),
    discord: z.object({
      enabled: z.boolean().default(false),
      webhookUrl: z.string().url().optional(),
    }).optional(),
    slack: z.object({
      enabled: z.boolean().default(false),
      webhookUrl: z.string().url().optional(),
    }).optional(),
    mobilePush: z.object({
      enabled: z.boolean().default(false),
      deviceTokens: z.array(z.string()).optional(),
      platform: z.enum(['ios', 'android', 'both']).optional(),
    }).optional(),
  }).optional(),
  
  // Model overrides for LLM selection
  modelOverrides: z.object({
    search: z.enum(['perplexity', 'deepseek-r1']).optional(),
    reasoning: z.enum(['gpt-4.1', 'claude-4-sonnet', 'gemini-flash']).optional(),
    summarise: z.enum(['claude-4-sonnet', 'qwen-2.5', 'llama-3-8b']).optional(),
  }).optional(),
  
  // Additional user preferences
  theme: z.enum(['light', 'dark', 'system']).optional(),
  language: z.string().optional(),
  timezone: z.string().optional(),
});

type UserPreferences = z.infer<typeof UserPreferencesSchema>;

/**
 * API route handler for user preferences
 * 
 * GET /api/v1/user/preferences - Get user preferences
 * PUT /api/v1/user/preferences - Update user preferences
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Only allow GET and PUT methods
  if (req.method !== 'GET' && req.method !== 'PUT') {
    return res.status(405).json({ error: 'Method not allowed', message: 'Only GET and PUT requests are supported' });
  }

  try {
    // Get gateway URL from environment
    const gatewayUrl = process.env.GATEWAY_URL || 'http://gateway.prowzi-system.svc.cluster.local:8080';
    const preferencesUrl = `${gatewayUrl}/api/v1/user/preferences`;
    
    // Check for authorization header
    if (!req.headers.authorization) {
      return res.status(401).json({ error: 'Unauthorized', message: 'Authentication required' });
    }

    // Handle PUT request (update preferences)
    if (req.method === 'PUT') {
      // Validate request body
      const validationResult = UserPreferencesSchema.safeParse(req.body);
      if (!validationResult.success) {
        return res.status(400).json({
          error: 'Invalid preferences data',
          details: validationResult.error.format()
        });
      }

      const preferencesData: UserPreferences = validationResult.data;
      
      // Forward the PUT request to the gateway service
      const response = await fetch(preferencesUrl, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': req.headers.authorization as string,
          // Forward tenant-specific headers if present
          ...(req.headers['x-tenant-id'] ? { 'X-Tenant-ID': req.headers['x-tenant-id'] as string } : {}),
          // Add tracking headers
          'X-Source': 'web-ui',
          'X-Request-ID': req.headers['x-request-id'] as string || generateRequestId()
        },
        body: JSON.stringify(preferencesData)
      });

      // Get response data
      const responseData = await response.json();

      // Forward the gateway response status and body
      res.status(response.status).json(responseData);

      // Log preferences update (only on success)
      if (response.ok) {
        console.log(`User preferences updated successfully`);
      }
    } 
    // Handle GET request (fetch preferences)
    else {
      // Forward the GET request to the gateway service
      const response = await fetch(preferencesUrl, {
        method: 'GET',
        headers: {
          'Authorization': req.headers.authorization as string,
          // Forward tenant-specific headers if present
          ...(req.headers['x-tenant-id'] ? { 'X-Tenant-ID': req.headers['x-tenant-id'] as string } : {}),
          // Add tracking headers
          'X-Source': 'web-ui',
          'X-Request-ID': req.headers['x-request-id'] as string || generateRequestId()
        }
      });

      // Check for 404 (no preferences found)
      if (response.status === 404) {
        // Return default preferences
        return res.status(200).json({
          domains: {
            crypto: true,
            ai: true
          },
          alertCadence: {
            instant: true,
            sameDay: true,
            weekly: true
          },
          notificationChannels: {
            email: {
              enabled: true
            }
          },
          modelOverrides: {}
        });
      }

      // Get response data
      const responseData = await response.json();

      // Forward the gateway response status and body
      res.status(response.status).json(responseData);
    }
  } catch (error) {
    console.error('Error handling user preferences:', error);
    
    // Return appropriate error response
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to process user preferences due to an internal error'
    });
  }
}

/**
 * Generate a unique request ID
 */
function generateRequestId(): string {
  return `web-${Date.now()}-${Math.random().toString(36).substring(2, 10)}`;
}
