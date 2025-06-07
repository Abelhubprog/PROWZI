import { SiweMessage } from 'siwe';
import { Connection, PublicKey } from '@solana/web3.js';
import bs58 from 'bs58';

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  scope: string[];
  user: UserInfo;
}

export interface UserInfo {
  id: string;
  email?: string;
  wallet_address?: string;
  tier: string;
  tenant_id: string;
}

export interface ProwziAuthConfig {
  apiUrl: string;
  tenantId?: string;
}

export class ProwziAuth {
  private apiUrl: string;
  private tenantId?: string;
  private tokens?: AuthTokens;

  constructor(config: ProwziAuthConfig) {
    this.apiUrl = config.apiUrl.replace(/\/$/, ''); // Remove trailing slash
    this.tenantId = config.tenantId;
  }

  /**
   * Get a nonce for wallet authentication
   */
  async getNonce(address?: string): Promise<{ nonce: string; message: string }> {
    const params = address ? `?address=${encodeURIComponent(address)}` : '';
    const response = await fetch(`${this.apiUrl}/auth/nonce${params}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get nonce: ${response.statusText}`);
    }
    
    return response.json();
  }

  /**
   * Authenticate with Ethereum wallet using SIWE
   */
  async authenticateEthereum(signer: any): Promise<AuthTokens> {
    try {
      const address = await signer.getAddress();
      const { message } = await this.getNonce(address);

      // Sign the SIWE message
      const signature = await signer.signMessage(message);

      const response = await fetch(`${this.apiUrl}/auth/wallet`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'ethereum',
          address,
          message,
          signature,
          tenant_id: this.tenantId,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Authentication failed');
      }

      this.tokens = await response.json();
      return this.tokens;
    } catch (error) {
      throw new Error(`Ethereum authentication failed: ${error.message}`);
    }
  }

  /**
   * Authenticate with Solana wallet
   */
  async authenticateSolana(wallet: any): Promise<AuthTokens> {
    try {
      const { message } = await this.getNonce();
      
      // Encode message for signing
      const encodedMessage = new TextEncoder().encode(message);
      
      // Sign message
      const signature = await wallet.signMessage(encodedMessage);

      const response = await fetch(`${this.apiUrl}/auth/wallet`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'solana',
          address: wallet.publicKey.toBase58(),
          message,
          signature: bs58.encode(signature),
          tenant_id: this.tenantId,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Authentication failed');
      }

      this.tokens = await response.json();
      return this.tokens;
    } catch (error) {
      throw new Error(`Solana authentication failed: ${error.message}`);
    }
  }

  /**
   * Refresh access token using refresh token
   */
  async refreshToken(refreshToken?: string): Promise<AuthTokens> {
    const token = refreshToken || this.tokens?.refresh_token;
    
    if (!token) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${this.apiUrl}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: token }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Token refresh failed');
    }

    this.tokens = await response.json();
    return this.tokens;
  }

  /**
   * Introspect a token to check if it's valid
   */
  async introspectToken(token?: string): Promise<any> {
    const accessToken = token || this.tokens?.access_token;
    
    if (!accessToken) {
      throw new Error('No access token available');
    }

    const response = await fetch(`${this.apiUrl}/auth/introspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: accessToken }),
    });

    if (!response.ok) {
      throw new Error('Token introspection failed');
    }

    return response.json();
  }

  /**
   * Get current tokens
   */
  getTokens(): AuthTokens | undefined {
    return this.tokens;
  }

  /**
   * Set tokens (useful for restoring from storage)
   */
  setTokens(tokens: AuthTokens): void {
    this.tokens = tokens;
  }

  /**
   * Clear current tokens
   */
  clearTokens(): void {
    this.tokens = undefined;
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!this.tokens?.access_token;
  }

  /**
   * Get access token for API calls
   */
  getAccessToken(): string | undefined {
    return this.tokens?.access_token;
  }

  /**
   * Get Authorization header value
   */
  getAuthHeader(): string | undefined {
    const token = this.getAccessToken();
    return token ? `Bearer ${token}` : undefined;
  }

  /**
   * Auto-refresh token if it's about to expire
   */
  async ensureValidToken(): Promise<string> {
    if (!this.tokens) {
      throw new Error('Not authenticated');
    }

    // Check if token expires in the next 5 minutes
    const expiresAt = Date.now() + (this.tokens.expires_in * 1000);
    const fiveMinutesFromNow = Date.now() + (5 * 60 * 1000);

    if (expiresAt < fiveMinutesFromNow) {
      await this.refreshToken();
    }

    return this.tokens!.access_token;
  }
}

// Convenience function for creating authenticated fetch requests
export function createAuthenticatedFetch(auth: ProwziAuth) {
  return async (url: string, options: RequestInit = {}): Promise<Response> => {
    const token = await auth.ensureValidToken();
    
    const authHeaders = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      ...options.headers,
    };

    return fetch(url, {
      ...options,
      headers: authHeaders,
    });
  };
}

// Export default instance for convenience
export default ProwziAuth;
