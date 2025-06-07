import type { NextApiRequest, NextApiResponse } from 'next';
import { Readable } from 'stream';

// Maximum retry attempts for reconnection
const MAX_RETRIES = 3;
// Initial retry delay in milliseconds (with exponential backoff)
const INITIAL_RETRY_DELAY = 1000;

/**
 * API route handler for mission streaming
 * 
 * GET /api/v1/missions/[id]/stream - Stream real-time mission updates
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Only allow GET method
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed', message: 'Only GET requests are supported' });
  }

  // Get mission ID from URL
  const { id } = req.query;
  if (!id || Array.isArray(id)) {
    return res.status(400).json({ error: 'Invalid mission ID', message: 'Mission ID must be a string' });
  }

  // Get gateway URL from environment
  const gatewayUrl = process.env.GATEWAY_URL || 'http://gateway.prowzi-system.svc.cluster.local:8080';
  const streamUrl = `${gatewayUrl}/api/v1/missions/${id}/stream`;

  // Set headers for SSE
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // Disable buffering for Nginx

  // Flag to track if the client has disconnected
  let clientDisconnected = false;

  // Handle client disconnect
  req.on('close', () => {
    clientDisconnected = true;
    console.log(`Client disconnected from mission stream: ${id}`);
  });

  // Connect to gateway with retry logic
  let retryCount = 0;
  let connected = false;

  while (retryCount < MAX_RETRIES && !clientDisconnected && !connected) {
    try {
      console.log(`Connecting to mission stream: ${id} (attempt ${retryCount + 1}/${MAX_RETRIES})`);

      // Connect to the gateway stream
      const response = await fetch(streamUrl, {
        headers: {
          // Forward authorization header if present
          ...(req.headers.authorization ? { 'Authorization': req.headers.authorization as string } : {}),
          // Forward tenant-specific headers if present
          ...(req.headers['x-tenant-id'] ? { 'X-Tenant-ID': req.headers['x-tenant-id'] as string } : {}),
          // Add tracking headers
          'Accept': 'text/event-stream',
          'X-Source': 'web-ui',
          'X-Request-ID': req.headers['x-request-id'] as string || generateRequestId()
        }
      });

      // Check if the response is successful
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Gateway returned ${response.status}: ${errorText}`);
      }

      // Get the response body as a readable stream
      if (!response.body) {
        throw new Error('No response body from gateway');
      }

      connected = true;
      console.log(`Connected to mission stream: ${id}`);

      // Send initial connection message
      res.write(`data: ${JSON.stringify({ type: 'connected', missionId: id })}\n\n`);

      // Forward the stream to the client
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      // Process the stream
      while (!clientDisconnected) {
        const { done, value } = await reader.read();
        
        if (done) {
          console.log(`Mission stream ended: ${id}`);
          break;
        }

        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          res.write(chunk);
          
          // Ensure the chunk is flushed
          if (res.flush && typeof res.flush === 'function') {
            res.flush();
          }
        }
      }

      // Clean up the reader
      reader.releaseLock();
      
      // If we've reached here and the client is still connected, the stream ended naturally
      if (!clientDisconnected) {
        res.write(`data: ${JSON.stringify({ type: 'complete', missionId: id })}\n\n`);
        res.end();
      }
      
      return;
    } catch (error) {
      console.error(`Error connecting to mission stream: ${id}`, error);
      retryCount++;

      // If client disconnected during retry wait, stop retrying
      if (clientDisconnected) {
        break;
      }

      // If we've reached max retries, send error and end
      if (retryCount >= MAX_RETRIES) {
        if (!clientDisconnected) {
          res.write(`data: ${JSON.stringify({ 
            type: 'error', 
            error: 'Failed to connect to mission stream after multiple attempts',
            missionId: id 
          })}\n\n`);
          res.end();
        }
        return;
      }

      // Wait before retrying with exponential backoff
      const retryDelay = INITIAL_RETRY_DELAY * Math.pow(2, retryCount - 1);
      console.log(`Retrying in ${retryDelay}ms...`);
      await new Promise(resolve => setTimeout(resolve, retryDelay));
    }
  }

  // If client disconnected before we could connect, just end the response
  if (clientDisconnected && !connected) {
    res.end();
  }
}

/**
 * Generate a unique request ID
 */
function generateRequestId(): string {
  return `web-${Date.now()}-${Math.random().toString(36).substring(2, 10)}`;
}

// Disable Next.js body parsing for this route
export const config = {
  api: {
    bodyParser: false,
  },
};
