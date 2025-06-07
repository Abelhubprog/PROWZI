// SDK Authentication Module for Prowzi Platform
// Provides authentication utilities for client applications

export interface ProwziAuth {
  authenticate(credentials: AuthCredentials): Promise<AuthResult>;
  refreshToken(token: string): Promise<AuthResult>;
  validateToken(token: string): Promise<boolean>;
  logout(token: string): Promise<void>;
}

export interface AuthCredentials {
  email?: string;
  apiKey?: string;
  clientId?: string;
  clientSecret?: string;
  provider?: 'email' | 'oauth' | 'api-key';
}

export interface AuthResult {
  token: string;
  refreshToken: string;
  expiresIn: number;
  user: UserInfo;
  permissions: string[];
}

export interface UserInfo {
  id: string;
  email: string;
  name: string;
  organization?: string;
  plan: 'free' | 'pro' | 'enterprise';
  verified: boolean;
}

export interface TokenPayload {
  sub: string;
  email: string;
  org?: string;
  plan: string;
  perms: string[];
  iat: number;
  exp: number;
}

class ProwziAuthClient implements ProwziAuth {
  private baseUrl: string;
  private clientId: string;
  private timeout: number;

  constructor(config: AuthConfig) {
    this.baseUrl = config.baseUrl || 'https://api.prowzi.com';
    this.clientId = config.clientId;
    this.timeout = config.timeout || 30000;
  }

  async authenticate(credentials: AuthCredentials): Promise<AuthResult> {
    const endpoint = this.getAuthEndpoint(credentials.provider || 'email');
    
    try {
      const response = await this.makeRequest('POST', endpoint, {
        ...credentials,
        clientId: this.clientId,
      });

      if (!response.ok) {
        throw new AuthError(`Authentication failed: ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateAuthResult(data);
    } catch (error) {
      throw new AuthError(`Authentication error: ${error.message}`);
    }
  }

  async refreshToken(refreshToken: string): Promise<AuthResult> {
    try {
      const response = await this.makeRequest('POST', '/auth/refresh', {
        refreshToken,
        clientId: this.clientId,
      });

      if (!response.ok) {
        throw new AuthError(`Token refresh failed: ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateAuthResult(data);
    } catch (error) {
      throw new AuthError(`Token refresh error: ${error.message}`);
    }
  }

  async validateToken(token: string): Promise<boolean> {
    try {
      const response = await this.makeRequest('POST', '/auth/validate', {
        token,
      });

      return response.ok;
    } catch (error) {
      return false;
    }
  }

  async logout(token: string): Promise<void> {
    try {
      await this.makeRequest('POST', '/auth/logout', {
        token,
      });
    } catch (error) {
      // Ignore logout errors - token may already be invalid
      console.warn('Logout error:', error.message);
    }
  }

  private getAuthEndpoint(provider: string): string {
    switch (provider) {
      case 'email':
        return '/auth/login';
      case 'oauth':
        return '/auth/oauth';
      case 'api-key':
        return '/auth/api-key';
      default:
        return '/auth/login';
    }
  }

  private async makeRequest(method: string, endpoint: string, body?: any): Promise<Response> {
    const url = `${this.baseUrl}${endpoint}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'Prowzi-SDK/1.0.0',
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  private validateAuthResult(data: any): AuthResult {
    if (!data.token || !data.user) {
      throw new AuthError('Invalid authentication response');
    }

    return {
      token: data.token,
      refreshToken: data.refreshToken || '',
      expiresIn: data.expiresIn || 3600,
      user: {
        id: data.user.id,
        email: data.user.email,
        name: data.user.name || '',
        organization: data.user.organization,
        plan: data.user.plan || 'free',
        verified: data.user.verified || false,
      },
      permissions: data.permissions || [],
    };
  }
}

export class AuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AuthError';
  }
}

export interface AuthConfig {
  baseUrl?: string;
  clientId: string;
  timeout?: number;
}

// JWT Token utilities
export class TokenUtils {
  static decode(token: string): TokenPayload | null {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) return null;

      const payload = JSON.parse(atob(parts[1]));
      return payload as TokenPayload;
    } catch {
      return null;
    }
  }

  static isExpired(token: string): boolean {
    const payload = this.decode(token);
    if (!payload) return true;

    return Date.now() / 1000 > payload.exp;
  }

  static getExpirationTime(token: string): Date | null {
    const payload = this.decode(token);
    if (!payload) return null;

    return new Date(payload.exp * 1000);
  }

  static hasPermission(token: string, permission: string): boolean {
    const payload = this.decode(token);
    if (!payload) return false;

    return payload.perms?.includes(permission) || false;
  }
}

// Storage utilities for tokens
export class TokenStorage {
  private static readonly TOKEN_KEY = 'prowzi_token';
  private static readonly REFRESH_KEY = 'prowzi_refresh_token';

  static setTokens(token: string, refreshToken: string): void {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(this.TOKEN_KEY, token);
      localStorage.setItem(this.REFRESH_KEY, refreshToken);
    }
  }

  static getToken(): string | null {
    if (typeof localStorage !== 'undefined') {
      return localStorage.getItem(this.TOKEN_KEY);
    }
    return null;
  }

  static getRefreshToken(): string | null {
    if (typeof localStorage !== 'undefined') {
      return localStorage.getItem(this.REFRESH_KEY);
    }
    return null;
  }

  static clearTokens(): void {
    if (typeof localStorage !== 'undefined') {
      localStorage.removeItem(this.TOKEN_KEY);
      localStorage.removeItem(this.REFRESH_KEY);
    }
  }
}

// Main export
export function createAuthClient(config: AuthConfig): ProwziAuth {
  return new ProwziAuthClient(config);
}

export default createAuthClient;