/**
 * Mobile Push Notification Module
 * 
 * Provides a unified interface for sending push notifications to iOS (APNS) and Android (FCM) devices.
 * Supports high-priority notifications for urgent briefs and includes retry logic for reliability.
 */

import * as admin from 'firebase-admin';
import * as jwt from 'jsonwebtoken';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { Brief } from '@prowzi/messages';
import { setTimeout } from 'timers/promises';

// Maximum retry attempts for failed notifications
const MAX_RETRIES = 3;

// Retry delay in milliseconds (exponential backoff)
const BASE_RETRY_DELAY = 500;

/**
 * Notification priority levels
 */
export enum NotificationPriority {
  NORMAL = 'normal',
  HIGH = 'high'
}

/**
 * Device platform types
 */
export enum DevicePlatform {
  IOS = 'ios',
  ANDROID = 'android'
}

/**
 * Device registration information
 */
export interface DeviceRegistration {
  id: string;
  token: string;
  platform: DevicePlatform;
  userId: string;
  tenantId: string;
  createdAt: Date;
  updatedAt: Date;
}

/**
 * Push notification payload
 */
export interface PushNotificationPayload {
  title: string;
  body: string;
  data?: Record<string, string>;
  priority?: NotificationPriority;
  badge?: number;
  sound?: string;
  category?: string;
  threadId?: string;
}

/**
 * Result of a push notification send attempt
 */
export interface PushNotificationResult {
  deviceId: string;
  platform: DevicePlatform;
  success: boolean;
  messageId?: string;
  error?: Error;
  retryCount?: number;
}

/**
 * Error thrown when push notification operations fail
 */
export class PushNotificationError extends Error {
  platform: DevicePlatform;
  deviceId?: string;
  retryable: boolean;
  
  constructor(message: string, platform: DevicePlatform, deviceId?: string, retryable = true) {
    super(message);
    this.name = 'PushNotificationError';
    this.platform = platform;
    this.deviceId = deviceId;
    this.retryable = retryable;
  }
}

/**
 * APNS (Apple Push Notification Service) client
 */
export class APNSClient {
  private baseUrl: string;
  private teamId: string;
  private keyId: string;
  private authKey: Buffer;
  private jwt: string | null = null;
  private jwtExpiry: number = 0;
  private initialized: boolean = false;
  
  constructor() {
    // Production vs. development environment
    this.baseUrl = process.env.NODE_ENV === 'production'
      ? 'https://api.push.apple.com'
      : 'https://api.sandbox.push.apple.com';
    
    // Get required environment variables
    this.teamId = process.env.APNS_TEAM_ID || '';
    this.keyId = process.env.APNS_KEY_ID || '';
    const authKeyBase64 = process.env.APNS_AUTH_KEY_B64 || '';
    
    // Decode the base64 auth key
    this.authKey = Buffer.from(authKeyBase64, 'base64');
    
    // Validate configuration
    if (!this.teamId || !this.keyId || !authKeyBase64) {
      console.warn('APNS configuration incomplete. iOS push notifications will be disabled.');
    } else {
      this.initialized = true;
    }
  }
  
  /**
   * Generate a JWT token for APNS authentication
   * @returns The JWT token
   */
  private generateToken(): string {
    if (!this.initialized) {
      throw new PushNotificationError(
        'APNS not properly configured',
        DevicePlatform.IOS,
        undefined,
        false
      );
    }
    
    // Token expires in 50 minutes (APNS tokens are valid for 60 minutes)
    const now = Math.floor(Date.now() / 1000);
    const expiresAt = now + 50 * 60;
    
    const token = jwt.sign({
      iss: this.teamId,
      iat: now
    }, this.authKey, {
      algorithm: 'ES256',
      header: {
        alg: 'ES256',
        kid: this.keyId
      },
      expiresIn: '50m'
    });
    
    this.jwt = token;
    this.jwtExpiry = expiresAt;
    
    return token;
  }
  
  /**
   * Get a valid JWT token, generating a new one if needed
   * @returns A valid JWT token
   */
  private getToken(): string {
    const now = Math.floor(Date.now() / 1000);
    
    // Generate a new token if we don't have one or it's about to expire
    if (!this.jwt || now >= this.jwtExpiry - 60) {
      return this.generateToken();
    }
    
    return this.jwt;
  }
  
  /**
   * Send a push notification to an iOS device
   * 
   * @param deviceToken The device token
   * @param payload The notification payload
   * @param deviceId Optional device ID for tracking
   * @param retryCount Current retry attempt (for internal use)
   * @returns Result of the push notification attempt
   */
  async sendNotification(
    deviceToken: string,
    payload: PushNotificationPayload,
    deviceId: string = 'unknown',
    retryCount: number = 0
  ): Promise<PushNotificationResult> {
    if (!this.initialized) {
      return {
        deviceId,
        platform: DevicePlatform.IOS,
        success: false,
        error: new PushNotificationError(
          'APNS not properly configured',
          DevicePlatform.IOS,
          deviceId,
          false
        )
      };
    }
    
    try {
      // Generate a unique ID for this notification
      const apnsId = uuidv4();
      
      // Prepare the APNS payload
      const apnsPayload = {
        aps: {
          alert: {
            title: payload.title,
            body: payload.body
          },
          badge: payload.badge,
          sound: payload.sound || 'default',
          'content-available': 1,
          priority: payload.priority === NotificationPriority.HIGH ? 10 : 5,
          category: payload.category,
          'thread-id': payload.threadId
        },
        ...payload.data
      };
      
      // Send the notification
      const response = await axios({
        method: 'POST',
        url: `${this.baseUrl}/3/device/${deviceToken}`,
        headers: {
          'Authorization': `Bearer ${this.getToken()}`,
          'apns-id': apnsId,
          'apns-expiration': '0', // 0 means the message expires immediately if it can't be delivered
          'apns-priority': payload.priority === NotificationPriority.HIGH ? '10' : '5',
          'apns-topic': process.env.APNS_BUNDLE_ID || 'com.prowzi.app',
          'apns-push-type': 'alert'
        },
        data: apnsPayload
      });
      
      return {
        deviceId,
        platform: DevicePlatform.IOS,
        success: true,
        messageId: response.headers['apns-id'] || apnsId,
        retryCount
      };
    } catch (error: any) {
      // Handle specific APNS error codes
      const status = error.response?.status;
      const reason = error.response?.data?.reason;
      
      // Determine if we should retry based on the error
      const retryable = !(
        // Don't retry for these errors
        status === 400 || // Bad request
        status === 403 || // Certificate error
        status === 410 || // Token is no longer valid
        reason === 'BadDeviceToken' ||
        reason === 'DeviceTokenNotForTopic' ||
        reason === 'Unregistered'
      );
      
      const errorMessage = reason 
        ? `APNS Error: ${reason} (${status})` 
        : `APNS Error: ${error.message}`;
      
      const pushError = new PushNotificationError(
        errorMessage,
        DevicePlatform.IOS,
        deviceId,
        retryable
      );
      
      // Retry if appropriate
      if (retryable && retryCount < MAX_RETRIES) {
        const delay = BASE_RETRY_DELAY * Math.pow(2, retryCount);
        console.log(`Retrying APNS notification in ${delay}ms (attempt ${retryCount + 1}/${MAX_RETRIES})`);
        
        await setTimeout(delay);
        return this.sendNotification(deviceToken, payload, deviceId, retryCount + 1);
      }
      
      return {
        deviceId,
        platform: DevicePlatform.IOS,
        success: false,
        error: pushError,
        retryCount
      };
    }
  }
}

/**
 * FCM (Firebase Cloud Messaging) client
 */
export class FCMClient {
  private initialized: boolean = false;
  private app: admin.app.App | null = null;
  
  constructor() {
    try {
      // Check if FCM is already initialized
      try {
        this.app = admin.app();
        this.initialized = true;
      } catch (error) {
        // Initialize Firebase Admin SDK
        const serverKey = process.env.FCM_SERVER_KEY;
        
        if (!serverKey) {
          console.warn('FCM_SERVER_KEY not provided. Android push notifications will be disabled.');
          return;
        }
        
        // Initialize with service account if available
        if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
          this.app = admin.initializeApp({
            credential: admin.credential.applicationDefault()
          });
        } 
        // Otherwise use the server key
        else {
          this.app = admin.initializeApp({
            credential: admin.credential.cert({
              projectId: process.env.FCM_PROJECT_ID || 'prowzi',
              clientEmail: process.env.FCM_CLIENT_EMAIL || `firebase-adminsdk@prowzi.iam.gserviceaccount.com`,
              // Using a private key directly is not ideal, but this supports the server key approach
              privateKey: serverKey.replace(/\\n/g, '\n')
            })
          });
        }
        
        this.initialized = true;
      }
    } catch (error) {
      console.error('Failed to initialize FCM:', error);
    }
  }
  
  /**
   * Send a push notification to an Android device
   * 
   * @param deviceToken The FCM device token
   * @param payload The notification payload
   * @param deviceId Optional device ID for tracking
   * @param retryCount Current retry attempt (for internal use)
   * @returns Result of the push notification attempt
   */
  async sendNotification(
    deviceToken: string,
    payload: PushNotificationPayload,
    deviceId: string = 'unknown',
    retryCount: number = 0
  ): Promise<PushNotificationResult> {
    if (!this.initialized || !this.app) {
      return {
        deviceId,
        platform: DevicePlatform.ANDROID,
        success: false,
        error: new PushNotificationError(
          'FCM not properly initialized',
          DevicePlatform.ANDROID,
          deviceId,
          false
        )
      };
    }
    
    try {
      // Prepare the FCM message
      const message: admin.messaging.Message = {
        token: deviceToken,
        notification: {
          title: payload.title,
          body: payload.body
        },
        data: payload.data || {},
        android: {
          priority: payload.priority === NotificationPriority.HIGH ? 'high' : 'normal',
          notification: {
            sound: payload.sound || 'default',
            clickAction: 'FLUTTER_NOTIFICATION_CLICK'
          }
        }
      };
      
      // Send the message
      const response = await admin.messaging().send(message);
      
      return {
        deviceId,
        platform: DevicePlatform.ANDROID,
        success: true,
        messageId: response,
        retryCount
      };
    } catch (error: any) {
      // Determine if we should retry based on the error
      const errorCode = error.code || '';
      const retryable = !(
        // Don't retry for these errors
        errorCode.includes('invalid-argument') ||
        errorCode.includes('registration-token-not-registered')
      );
      
      const pushError = new PushNotificationError(
        `FCM Error: ${error.message}`,
        DevicePlatform.ANDROID,
        deviceId,
        retryable
      );
      
      // Retry if appropriate
      if (retryable && retryCount < MAX_RETRIES) {
        const delay = BASE_RETRY_DELAY * Math.pow(2, retryCount);
        console.log(`Retrying FCM notification in ${delay}ms (attempt ${retryCount + 1}/${MAX_RETRIES})`);
        
        await setTimeout(delay);
        return this.sendNotification(deviceToken, payload, deviceId, retryCount + 1);
      }
      
      return {
        deviceId,
        platform: DevicePlatform.ANDROID,
        success: false,
        error: pushError,
        retryCount
      };
    }
  }
}

// Create singleton instances
const apnsClient = new APNSClient();
const fcmClient = new FCMClient();

/**
 * Send a push notification to a device
 * 
 * @param device The device registration information
 * @param payload The notification payload
 * @returns Result of the push notification attempt
 */
export async function sendPushNotification(
  device: DeviceRegistration,
  payload: PushNotificationPayload
): Promise<PushNotificationResult> {
  if (device.platform === DevicePlatform.IOS) {
    return apnsClient.sendNotification(device.token, payload, device.id);
  } else if (device.platform === DevicePlatform.ANDROID) {
    return fcmClient.sendNotification(device.token, payload, device.id);
  } else {
    return {
      deviceId: device.id,
      platform: device.platform,
      success: false,
      error: new PushNotificationError(
        `Unsupported platform: ${device.platform}`,
        device.platform as DevicePlatform,
        device.id,
        false
      )
    };
  }
}

/**
 * Send push notifications to multiple devices
 * 
 * @param devices List of device registrations
 * @param payload The notification payload
 * @returns Results for each notification attempt
 */
export async function sendPushNotifications(
  devices: DeviceRegistration[],
  payload: PushNotificationPayload
): Promise<PushNotificationResult[]> {
  // Send notifications in parallel
  const promises = devices.map(device => sendPushNotification(device, payload));
  return Promise.all(promises);
}

/**
 * Create a push notification payload from a brief
 * 
 * @param brief The brief to create a notification for
 * @returns A push notification payload
 */
export function createBriefNotification(brief: Brief): PushNotificationPayload {
  // Determine priority based on band
  const priority = brief.band === 'instant' 
    ? NotificationPriority.HIGH 
    : NotificationPriority.NORMAL;
  
  // Create the notification payload
  return {
    title: brief.headline,
    body: brief.summary,
    data: {
      briefId: brief.id,
      eventId: brief.eventId,
      domain: brief.domain,
      source: brief.source,
      band: brief.band,
      createdAt: brief.createdAt
    },
    priority,
    sound: priority === NotificationPriority.HIGH ? 'critical' : 'default',
    category: `BRIEF_${brief.domain.toUpperCase()}`,
    threadId: brief.domain
  };
}

/**
 * Initialize the push notification services
 * This should be called at application startup
 */
export function initializePushServices(): void {
  console.log('Initializing push notification services...');
  
  // Check APNS configuration
  const apnsConfigured = process.env.APNS_TEAM_ID && 
                         process.env.APNS_KEY_ID && 
                         process.env.APNS_AUTH_KEY_B64;
  
  console.log(`APNS (iOS) push notifications: ${apnsConfigured ? 'ENABLED' : 'DISABLED'}`);
  
  // Check FCM configuration
  const fcmConfigured = process.env.FCM_SERVER_KEY || process.env.GOOGLE_APPLICATION_CREDENTIALS;
  console.log(`FCM (Android) push notifications: ${fcmConfigured ? 'ENABLED' : 'DISABLED'}`);
  
  if (!apnsConfigured && !fcmConfigured) {
    console.warn('WARNING: Both APNS and FCM are not configured. Mobile push notifications will be disabled.');
  }
}
