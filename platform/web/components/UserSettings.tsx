import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/router';

// Types for user preferences
interface UserPreferences {
  domains?: {
    crypto: boolean;
    ai: boolean;
    defi?: boolean;
    nft?: boolean;
    security?: boolean;
    research?: boolean;
  };
  alertCadence?: {
    instant: boolean;
    sameDay: boolean;
    weekly: boolean;
    maxAlertsPerHour?: number;
    quietHours?: {
      enabled: boolean;
      start?: string;
      end?: string;
      timezone?: string;
    };
  };
  notificationChannels?: {
    email?: {
      enabled: boolean;
      address?: string;
      digest?: 'instant' | 'hourly' | 'daily' | 'weekly';
    };
    telegram?: {
      enabled: boolean;
      chatId?: string;
    };
    discord?: {
      enabled: boolean;
      webhookUrl?: string;
    };
    slack?: {
      enabled: boolean;
      webhookUrl?: string;
    };
    mobilePush?: {
      enabled: boolean;
      deviceTokens?: string[];
      platform?: 'ios' | 'android' | 'both';
    };
  };
  modelOverrides?: {
    search?: 'perplexity' | 'deepseek-r1';
    reasoning?: 'gpt-4.1' | 'claude-4-sonnet' | 'gemini-flash';
    summarise?: 'claude-4-sonnet' | 'qwen-2.5' | 'llama-3-8b';
  };
  theme?: 'light' | 'dark' | 'system';
  language?: string;
  timezone?: string;
}

// Component props
interface UserSettingsProps {
  className?: string;
}

const UserSettings: React.FC<UserSettingsProps> = ({ className }) => {
  const router = useRouter();
  
  // State for preferences
  const [preferences, setPreferences] = useState<UserPreferences>({
    domains: {
      crypto: true,
      ai: true,
      defi: false,
      nft: false,
      security: false,
      research: false,
    },
    alertCadence: {
      instant: true,
      sameDay: true,
      weekly: true,
      maxAlertsPerHour: 10,
      quietHours: {
        enabled: false,
        start: '22:00',
        end: '08:00',
        timezone: 'UTC',
      },
    },
    notificationChannels: {
      email: {
        enabled: true,
        digest: 'daily',
      },
      telegram: {
        enabled: false,
      },
      discord: {
        enabled: false,
      },
      slack: {
        enabled: false,
      },
      mobilePush: {
        enabled: false,
        platform: 'both',
      },
    },
    modelOverrides: {
      search: 'perplexity',
      reasoning: 'gpt-4.1',
      summarise: 'llama-3-8b',
    },
    theme: 'system',
  });
  
  // UI states
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);
  
  // Alert cadence slider value (1-100)
  const [alertSliderValue, setAlertSliderValue] = useState(50);
  
  // Fetch user preferences
  const fetchPreferences = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/user/preferences', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token') || ''}`,
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch preferences: ${response.statusText}`);
      }
      
      const data = await response.json();
      setPreferences(data);
      
      // Set alert slider based on maxAlertsPerHour
      if (data.alertCadence?.maxAlertsPerHour) {
        // Convert maxAlertsPerHour (0-100) to slider value (1-100)
        setAlertSliderValue(Math.min(100, Math.max(1, data.alertCadence.maxAlertsPerHour)));
      }
    } catch (err) {
      console.error('Error fetching preferences:', err);
      setError('Failed to load your preferences. Please try again later.');
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Save user preferences
  const savePreferences = async () => {
    setSaving(true);
    setError(null);
    setSaveSuccess(false);
    
    try {
      // Update maxAlertsPerHour from slider value
      const updatedPreferences = {
        ...preferences,
        alertCadence: {
          ...preferences.alertCadence,
          maxAlertsPerHour: alertSliderValue,
        },
      };
      
      const response = await fetch('/api/v1/user/preferences', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token') || ''}`,
        },
        body: JSON.stringify(updatedPreferences),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to save preferences: ${response.statusText}`);
      }
      
      setSaveSuccess(true);
      
      // Hide success message after 3 seconds
      setTimeout(() => {
        setSaveSuccess(false);
      }, 3000);
    } catch (err) {
      console.error('Error saving preferences:', err);
      setError('Failed to save your preferences. Please try again later.');
    } finally {
      setSaving(false);
    }
  };
  
  // Load preferences on mount
  useEffect(() => {
    fetchPreferences();
  }, [fetchPreferences]);
  
  // Handle domain toggle change
  const handleDomainToggle = (domain: keyof UserPreferences['domains']) => {
    setPreferences(prev => ({
      ...prev,
      domains: {
        ...prev.domains,
        [domain]: !prev.domains?.[domain],
      },
    }));
  };
  
  // Handle alert cadence toggle change
  const handleAlertCadenceToggle = (cadence: keyof UserPreferences['alertCadence']) => {
    if (cadence === 'quietHours' || cadence === 'maxAlertsPerHour') return;
    
    setPreferences(prev => ({
      ...prev,
      alertCadence: {
        ...prev.alertCadence,
        [cadence]: !prev.alertCadence?.[cadence],
      },
    }));
  };
  
  // Handle quiet hours toggle
  const handleQuietHoursToggle = () => {
    setPreferences(prev => ({
      ...prev,
      alertCadence: {
        ...prev.alertCadence,
        quietHours: {
          ...prev.alertCadence?.quietHours,
          enabled: !prev.alertCadence?.quietHours?.enabled,
        },
      },
    }));
  };
  
  // Handle quiet hours time change
  const handleQuietHoursChange = (field: 'start' | 'end', value: string) => {
    setPreferences(prev => ({
      ...prev,
      alertCadence: {
        ...prev.alertCadence,
        quietHours: {
          ...prev.alertCadence?.quietHours,
          [field]: value,
        },
      },
    }));
  };
  
  // Handle notification channel toggle
  const handleChannelToggle = (channel: keyof UserPreferences['notificationChannels']) => {
    setPreferences(prev => ({
      ...prev,
      notificationChannels: {
        ...prev.notificationChannels,
        [channel]: {
          ...prev.notificationChannels?.[channel],
          enabled: !prev.notificationChannels?.[channel]?.enabled,
        },
      },
    }));
  };
  
  // Handle notification channel field change
  const handleChannelFieldChange = (
    channel: keyof UserPreferences['notificationChannels'],
    field: string,
    value: string
  ) => {
    setPreferences(prev => ({
      ...prev,
      notificationChannels: {
        ...prev.notificationChannels,
        [channel]: {
          ...prev.notificationChannels?.[channel],
          [field]: value,
        },
      },
    }));
  };
  
  // Handle model override change
  const handleModelOverrideChange = (
    type: keyof UserPreferences['modelOverrides'],
    value: string
  ) => {
    setPreferences(prev => ({
      ...prev,
      modelOverrides: {
        ...prev.modelOverrides,
        [type]: value,
      },
    }));
  };
  
  // Get alert cadence label based on slider value
  const getAlertCadenceLabel = () => {
    if (alertSliderValue <= 20) return 'Low (Few alerts per hour)';
    if (alertSliderValue <= 50) return 'Medium (Balanced alert frequency)';
    if (alertSliderValue <= 80) return 'High (More frequent alerts)';
    return 'Firehose (Maximum alert frequency)';
  };
  
  // Render loading state
  if (loading) {
    return (
      <div className={`user-settings ${className || ''}`}>
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading your preferences...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className={`user-settings ${className || ''}`}>
      <h1 className="settings-title">User Settings</h1>
      
      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={fetchPreferences}>Try Again</button>
        </div>
      )}
      
      {saveSuccess && (
        <div className="success-message">
          <p>Your preferences have been saved successfully!</p>
        </div>
      )}
      
      <div className="settings-section">
        <h2>Domain Preferences</h2>
        <p className="section-description">Select which domains you want to receive intelligence about.</p>
        
        <div className="toggle-group">
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.domains?.crypto || false}
                onChange={() => handleDomainToggle('crypto')}
              />
              <span className="toggle-text">Crypto</span>
            </label>
          </div>
          
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.domains?.ai || false}
                onChange={() => handleDomainToggle('ai')}
              />
              <span className="toggle-text">AI</span>
            </label>
          </div>
          
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.domains?.defi || false}
                onChange={() => handleDomainToggle('defi')}
              />
              <span className="toggle-text">DeFi</span>
            </label>
          </div>
          
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.domains?.nft || false}
                onChange={() => handleDomainToggle('nft')}
              />
              <span className="toggle-text">NFT</span>
            </label>
          </div>
          
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.domains?.security || false}
                onChange={() => handleDomainToggle('security')}
              />
              <span className="toggle-text">Security</span>
            </label>
          </div>
          
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.domains?.research || false}
                onChange={() => handleDomainToggle('research')}
              />
              <span className="toggle-text">Research</span>
            </label>
          </div>
        </div>
      </div>
      
      <div className="settings-section">
        <h2>Alert Cadence</h2>
        <p className="section-description">Configure how frequently you want to receive alerts.</p>
        
        <div className="toggle-group">
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.alertCadence?.instant || false}
                onChange={() => handleAlertCadenceToggle('instant')}
              />
              <span className="toggle-text">Instant</span>
            </label>
          </div>
          
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.alertCadence?.sameDay || false}
                onChange={() => handleAlertCadenceToggle('sameDay')}
              />
              <span className="toggle-text">Same-Day</span>
            </label>
          </div>
          
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.alertCadence?.weekly || false}
                onChange={() => handleAlertCadenceToggle('weekly')}
              />
              <span className="toggle-text">Weekly</span>
            </label>
          </div>
        </div>
        
        <div className="slider-container">
          <label htmlFor="alert-frequency" className="slider-label">
            Alert Frequency: <span className="slider-value">{getAlertCadenceLabel()}</span>
          </label>
          <input
            id="alert-frequency"
            type="range"
            min="1"
            max="100"
            value={alertSliderValue}
            onChange={(e) => setAlertSliderValue(parseInt(e.target.value))}
            className="slider"
          />
          <div className="slider-labels">
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
            <span>Firehose</span>
          </div>
        </div>
        
        <div className="quiet-hours">
          <div className="toggle-item">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={preferences.alertCadence?.quietHours?.enabled || false}
                onChange={handleQuietHoursToggle}
              />
              <span className="toggle-text">Enable Quiet Hours</span>
            </label>
          </div>
          
          {preferences.alertCadence?.quietHours?.enabled && (
            <div className="quiet-hours-times">
              <div className="time-input">
                <label htmlFor="quiet-start">Start Time:</label>
                <input
                  id="quiet-start"
                  type="time"
                  value={preferences.alertCadence?.quietHours?.start || '22:00'}
                  onChange={(e) => handleQuietHoursChange('start', e.target.value)}
                />
              </div>
              
              <div className="time-input">
                <label htmlFor="quiet-end">End Time:</label>
                <input
                  id="quiet-end"
                  type="time"
                  value={preferences.alertCadence?.quietHours?.end || '08:00'}
                  onChange={(e) => handleQuietHoursChange('end', e.target.value)}
                />
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div className="settings-section">
        <h2>Notification Channels</h2>
        <p className="section-description">Configure where you want to receive notifications.</p>
        
        <div className="notification-channels">
          <div className="channel-item">
            <div className="channel-header">
              <h3>Email</h3>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={preferences.notificationChannels?.email?.enabled || false}
                  onChange={() => handleChannelToggle('email')}
                />
                <span className="switch-slider"></span>
              </label>
            </div>
            
            {preferences.notificationChannels?.email?.enabled && (
              <div className="channel-settings">
                <div className="form-group">
                  <label htmlFor="email-address">Email Address:</label>
                  <input
                    id="email-address"
                    type="email"
                    value={preferences.notificationChannels?.email?.address || ''}
                    onChange={(e) => handleChannelFieldChange('email', 'address', e.target.value)}
                    placeholder="your@email.com"
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="email-digest">Digest Frequency:</label>
                  <select
                    id="email-digest"
                    value={preferences.notificationChannels?.email?.digest || 'daily'}
                    onChange={(e) => handleChannelFieldChange('email', 'digest', e.target.value)}
                  >
                    <option value="instant">Instant</option>
                    <option value="hourly">Hourly</option>
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                  </select>
                </div>
              </div>
            )}
          </div>
          
          <div className="channel-item">
            <div className="channel-header">
              <h3>Telegram</h3>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={preferences.notificationChannels?.telegram?.enabled || false}
                  onChange={() => handleChannelToggle('telegram')}
                />
                <span className="switch-slider"></span>
              </label>
            </div>
            
            {preferences.notificationChannels?.telegram?.enabled && (
              <div className="channel-settings">
                <div className="form-group">
                  <label htmlFor="telegram-chat-id">Chat ID:</label>
                  <input
                    id="telegram-chat-id"
                    type="text"
                    value={preferences.notificationChannels?.telegram?.chatId || ''}
                    onChange={(e) => handleChannelFieldChange('telegram', 'chatId', e.target.value)}
                    placeholder="Your Telegram Chat ID"
                  />
                </div>
                <p className="help-text">
                  To get your Chat ID, message @ProwziBot on Telegram and type /start
                </p>
              </div>
            )}
          </div>
          
          <div className="channel-item">
            <div className="channel-header">
              <h3>Discord</h3>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={preferences.notificationChannels?.discord?.enabled || false}
                  onChange={() => handleChannelToggle('discord')}
                />
                <span className="switch-slider"></span>
              </label>
            </div>
            
            {preferences.notificationChannels?.discord?.enabled && (
              <div className="channel-settings">
                <div className="form-group">
                  <label htmlFor="discord-webhook">Webhook URL:</label>
                  <input
                    id="discord-webhook"
                    type="text"
                    value={preferences.notificationChannels?.discord?.webhookUrl || ''}
                    onChange={(e) => handleChannelFieldChange('discord', 'webhookUrl', e.target.value)}
                    placeholder="https://discord.com/api/webhooks/..."
                  />
                </div>
                <p className="help-text">
                  Create a webhook in your Discord server's channel settings
                </p>
              </div>
            )}
          </div>
          
          <div className="channel-item">
            <div className="channel-header">
              <h3>Slack</h3>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={preferences.notificationChannels?.slack?.enabled || false}
                  onChange={() => handleChannelToggle('slack')}
                />
                <span className="switch-slider"></span>
              </label>
            </div>
            
            {preferences.notificationChannels?.slack?.enabled && (
              <div className="channel-settings">
                <div className="form-group">
                  <label htmlFor="slack-webhook">Webhook URL:</label>
                  <input
                    id="slack-webhook"
                    type="text"
                    value={preferences.notificationChannels?.slack?.webhookUrl || ''}
                    onChange={(e) => handleChannelFieldChange('slack', 'webhookUrl', e.target.value)}
                    placeholder="https://hooks.slack.com/services/..."
                  />
                </div>
                <p className="help-text">
                  Create a webhook in your Slack workspace's app settings
                </p>
              </div>
            )}
          </div>
          
          <div className="channel-item">
            <div className="channel-header">
              <h3>Mobile Push</h3>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={preferences.notificationChannels?.mobilePush?.enabled || false}
                  onChange={() => handleChannelToggle('mobilePush')}
                />
                <span className="switch-slider"></span>
              </label>
            </div>
            
            {preferences.notificationChannels?.mobilePush?.enabled && (
              <div className="channel-settings">
                <div className="form-group">
                  <label htmlFor="mobile-platform">Platform:</label>
                  <select
                    id="mobile-platform"
                    value={preferences.notificationChannels?.mobilePush?.platform || 'both'}
                    onChange={(e) => handleChannelFieldChange('mobilePush', 'platform', e.target.value)}
                  >
                    <option value="ios">iOS</option>
                    <option value="android">Android</option>
                    <option value="both">Both</option>
                  </select>
                </div>
                <p className="help-text">
                  Download our mobile app and sign in to enable push notifications
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="settings-section">
        <h2>Model Selection</h2>
        <p className="section-description">Customize which AI models are used for different tasks.</p>
        
        <div className="model-overrides">
          <div className="model-item">
            <label htmlFor="search-model">Search Model:</label>
            <select
              id="search-model"
              value={preferences.modelOverrides?.search || 'perplexity'}
              onChange={(e) => handleModelOverrideChange('search', e.target.value)}
            >
              <option value="perplexity">Perplexity (Default)</option>
              <option value="deepseek-r1">Deepseek R1</option>
            </select>
            <p className="model-description">Used for retrieving information from external sources</p>
          </div>
          
          <div className="model-item">
            <label htmlFor="reasoning-model">Reasoning Model:</label>
            <select
              id="reasoning-model"
              value={preferences.modelOverrides?.reasoning || 'gpt-4.1'}
              onChange={(e) => handleModelOverrideChange('reasoning', e.target.value)}
            >
              <option value="gpt-4.1">GPT-4.1 (Default)</option>
              <option value="claude-4-sonnet">Claude 4 Sonnet</option>
              <option value="gemini-flash">Gemini Flash</option>
            </select>
            <p className="model-description">Used for complex analysis and planning</p>
          </div>
          
          <div className="model-item">
            <label htmlFor="summarise-model">Summarization Model:</label>
            <select
              id="summarise-model"
              value={preferences.modelOverrides?.summarise || 'llama-3-8b'}
              onChange={(e) => handleModelOverrideChange('summarise', e.target.value)}
            >
              <option value="claude-4-sonnet">Claude 4 Sonnet (High quality)</option>
              <option value="qwen-2.5">Qwen 2.5 (Balanced)</option>
              <option value="llama-3-8b">Llama 3 8B (Default - Efficient)</option>
            </select>
            <p className="model-description">Used for generating intelligence briefs</p>
          </div>
        </div>
      </div>
      
      <div className="settings-actions">
        <button
          className="cancel-button"
          onClick={() => router.back()}
          disabled={saving}
        >
          Cancel
        </button>
        <button
          className="save-button"
          onClick={savePreferences}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save Preferences'}
        </button>
      </div>
      
      <style jsx>{`
        .user-settings {
          max-width: 800px;
          margin: 0 auto;
          padding: 2rem;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .settings-title {
          font-size: 2rem;
          margin-bottom: 2rem;
          color: #333;
        }
        
        .settings-section {
          margin-bottom: 2.5rem;
          padding: 1.5rem;
          background-color: #fff;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .settings-section h2 {
          font-size: 1.5rem;
          margin-top: 0;
          margin-bottom: 0.5rem;
          color: #333;
        }
        
        .section-description {
          color: #666;
          margin-bottom: 1.5rem;
        }
        
        .toggle-group {
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
          margin-bottom: 1.5rem;
        }
        
        .toggle-item {
          min-width: 150px;
        }
        
        .toggle-label {
          display: flex;
          align-items: center;
          cursor: pointer;
        }
        
        .toggle-text {
          margin-left: 0.5rem;
        }
        
        .slider-container {
          margin-top: 1.5rem;
        }
        
        .slider-label {
          display: block;
          margin-bottom: 0.5rem;
        }
        
        .slider-value {
          font-weight: 600;
          color: #0070f3;
        }
        
        .slider {
          width: 100%;
          height: 8px;
          -webkit-appearance: none;
          appearance: none;
          background: #ddd;
          outline: none;
          border-radius: 4px;
        }
        
        .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #0070f3;
          cursor: pointer;
        }
        
        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #0070f3;
          cursor: pointer;
        }
        
        .slider-labels {
          display: flex;
          justify-content: space-between;
          margin-top: 0.5rem;
          color: #666;
          font-size: 0.8rem;
        }
        
        .quiet-hours {
          margin-top: 1.5rem;
          padding-top: 1.5rem;
          border-top: 1px solid #eee;
        }
        
        .quiet-hours-times {
          display: flex;
          gap: 1rem;
          margin-top: 1rem;
        }
        
        .time-input {
          flex: 1;
        }
        
        .time-input label {
          display: block;
          margin-bottom: 0.5rem;
          color: #666;
        }
        
        .time-input input {
          width: 100%;
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px;
        }
        
        .notification-channels {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }
        
        .channel-item {
          padding: 1rem;
          border: 1px solid #eee;
          border-radius: 6px;
        }
        
        .channel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .channel-header h3 {
          margin: 0;
        }
        
        .toggle-switch {
          position: relative;
          display: inline-block;
          width: 50px;
          height: 24px;
        }
        
        .toggle-switch input {
          opacity: 0;
          width: 0;
          height: 0;
        }
        
        .switch-slider {
          position: absolute;
          cursor: pointer;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: #ccc;
          transition: .4s;
          border-radius: 24px;
        }
        
        .switch-slider:before {
          position: absolute;
          content: "";
          height: 16px;
          width: 16px;
          left: 4px;
          bottom: 4px;
          background-color: white;
          transition: .4s;
          border-radius: 50%;
        }
        
        input:checked + .switch-slider {
          background-color: #0070f3;
        }
        
        input:checked + .switch-slider:before {
          transform: translateX(26px);
        }
        
        .channel-settings {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #eee;
        }
        
        .form-group {
          margin-bottom: 1rem;
        }
        
        .form-group label {
          display: block;
          margin-bottom: 0.5rem;
          color: #666;
        }
        
        .form-group input,
        .form-group select {
          width: 100%;
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 1rem;
        }
        
        .help-text {
          margin-top: 0.5rem;
          font-size: 0.8rem;
          color: #666;
        }
        
        .model-overrides {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }
        
        .model-item {
          margin-bottom: 1rem;
        }
        
        .model-item label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 500;
        }
        
        .model-item select {
          width: 100%;
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 1rem;
          margin-bottom: 0.5rem;
        }
        
        .model-description {
          margin: 0;
          font-size: 0.8rem;
          color: #666;
        }
        
        .settings-actions {
          display: flex;
          justify-content: flex-end;
          gap: 1rem;
          margin-top: 2rem;
        }
        
        .cancel-button,
        .save-button {
          padding: 0.75rem 1.5rem;
          border-radius: 4px;
          font-size: 1rem;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .cancel-button {
          background-color: #f5f5f5;
          border: 1px solid #ddd;
          color: #333;
        }
        
        .cancel-button:hover {
          background-color: #eee;
        }
        
        .save-button {
          background-color: #0070f3;
          border: none;
          color: white;
        }
        
        .save-button:hover {
          background-color: #0060df;
        }
        
        .save-button:disabled,
        .cancel-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
        
        .loading-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 300px;
        }
        
        .loading-spinner {
          border: 4px solid #f3f3f3;
          border-top: 4px solid #0070f3;
          border-radius: 50%;
          width: 40px;
          height: 40px;
          animation: spin 1s linear infinite;
          margin-bottom: 1rem;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .error-message {
          padding: 1rem;
          background-color: #fee;
          border-left: 4px solid #f44;
          margin-bottom: 2rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .error-message button {
          background-color: #f44;
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .success-message {
          padding: 1rem;
          background-color: #e6f7e6;
          border-left: 4px solid #4caf50;
          margin-bottom: 2rem;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
          .user-settings {
            padding: 1rem;
          }
          
          .toggle-group {
            flex-direction: column;
            gap: 0.5rem;
          }
          
          .quiet-hours-times {
            flex-direction: column;
          }
          
          .settings-actions {
            flex-direction: column-reverse;
          }
          
          .save-button,
          .cancel-button {
            width: 100%;
          }
        }
      `}</style>
    </div>
  );
};

export default UserSettings;
