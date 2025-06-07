// mobile/src/services/notifications.ts
import messaging from '@react-native-firebase/messaging'
import { Platform } from 'react-native'
import ProwziAuth from '@prowzi/sdk'

export class NotificationService {
  async requestPermission(): Promise<boolean> {
    const authStatus = await messaging().requestPermission()
    return authStatus === messaging.AuthorizationStatus.AUTHORIZED
  }

  async registerDevice(): Promise<void> {
    const token = await messaging().getToken()
    const platform = Platform.OS

    await fetch(`${API_URL}/devices/register`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${await getAuthToken()}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ token, platform }),
    })
  }

  setupHandlers(): void {
    // Foreground messages
    messaging().onMessage(async remoteMessage => {
      const { briefId, impactLevel } = remoteMessage.data

      // Show local notification
      await notifee.displayNotification({
        title: remoteMessage.notification?.title,
        body: remoteMessage.notification?.body,
        ios: {
          sound: 'default',
          criticalVolume: impactLevel === 'critical' ? 1.0 : 0.5,
        },
        android: {
          channelId: `prowzi-${impactLevel}`,
          sound: 'default',
          importance: impactLevel === 'critical' 
            ? AndroidImportance.HIGH 
            : AndroidImportance.DEFAULT,
        },
      })
    })

    // Background/quit message handling
    messaging().setBackgroundMessageHandler(async remoteMessage => {
      const { briefId } = remoteMessage.data
      // Cache brief for quick open
      await AsyncStorage.setItem('pending_brief', briefId)
    })
  }
}
