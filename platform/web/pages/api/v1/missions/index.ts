import type { NextApiRequest, NextApiResponse } from 'next';
import { z } from 'zod';

// Schema for mission creation payload validation
const MissionCreateSchema = z.object({
  prompt: z.string().min(5).max(2000),
  duration: z.string().or(z.number()),
  budget: z.object({
    tokens: z.number().optional(),
    compute: z.enum(['low', 'medium', 'high']).optional()
  }).optional()
});

type MissionCreateRequest = z.infer<typeof MissionCreateSchema>;

/**
 * API route handler for mission creation
 * 
 * POST /api/v1/missions - Create a new mission
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Only allow POST method
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed', message: 'Only POST requests are supported' });
  }

  try {
    // Get gateway URL from environment
    const gatewayUrl = process.env.GATEWAY_URL || 'http://gateway.prowzi-system.svc.cluster.local:8080';
    
    // Validate request body
    const validationResult = MissionCreateSchema.safeParse(req.body);
    if (!validationResult.success) {
      return res.status(400).json({
        error: 'Invalid request body',
        details: validationResult.error.format()
      });
    }

    const missionData: MissionCreateRequest = validationResult.data;
    
    // Forward the request to the gateway service
    const response = await fetch(`${gatewayUrl}/api/v1/missions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Forward authorization header if present
        ...(req.headers.authorization ? { 'Authorization': req.headers.authorization as string } : {}),
        // Forward tenant-specific headers if present
        ...(req.headers['x-tenant-id'] ? { 'X-Tenant-ID': req.headers['x-tenant-id'] as string } : {}),
        // Add tracking headers
        'X-Source': 'web-ui',
        'X-Request-ID': req.headers['x-request-id'] as string || generateRequestId()
      },
      body: JSON.stringify(missionData)
    });

    // Get response data
    const responseData = await response.json();

    // Forward the gateway response status and body
    res.status(response.status).json(responseData);

    // Log mission creation (only on success)
    if (response.ok) {
      console.log(`Mission created: ${responseData.missionId || 'unknown'}`);
    }
  } catch (error) {
    console.error('Error creating mission:', error);
    
    // Return appropriate error response
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to create mission due to an internal error'
    });
  }
}

/**
 * Generate a unique request ID
 */
function generateRequestId(): string {
  return `web-${Date.now()}-${Math.random().toString(36).substring(2, 10)}`;
}
