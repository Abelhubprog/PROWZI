package channels

import (
	"context"
	"fmt"
	"os"

	"go.uber.org/zap"
	"github.com/prowzi/notifier/internal/metrics"
)

// TelegramConfig represents Telegram channel configuration
type TelegramConfig struct {
	Enabled    bool   `json:"enabled" yaml:"enabled"`
	BotToken   string `json:"bot_token" yaml:"bot_token"`
	WebhookURL string `json:"webhook_url" yaml:"webhook_url"`
	ChatID     string `json:"chat_id" yaml:"chat_id"`
}

// TelegramChannel implements the Channel interface for Telegram
type TelegramChannel struct {
	config    *TelegramConfig
	logger    *zap.Logger
	metrics   *metrics.Metrics
	notifier  *TelegramNotifier
}

// NewTelegramChannel creates a new Telegram channel
func NewTelegramChannel(config *TelegramConfig, logger *zap.Logger, metrics *metrics.Metrics) (*TelegramChannel, error) {
	if !config.Enabled {
		return nil, fmt.Errorf("telegram channel is disabled")
	}

	if config.BotToken == "" {
		return nil, fmt.Errorf("telegram bot token is required")
	}

	// Create Telegram notifier
	notifier := NewTelegramNotifier(config.BotToken)

	// Set webhook if configured
	if config.WebhookURL != "" {
		if err := notifier.SetWebhook(config.WebhookURL); err != nil {
			logger.Warn("failed to set telegram webhook", zap.Error(err))
		} else {
			logger.Info("telegram webhook set successfully", zap.String("url", config.WebhookURL))
		}
	}

	channel := &TelegramChannel{
		config:   config,
		logger:   logger,
		metrics:  metrics,
		notifier: notifier,
	}

	return channel, nil
}

// Send sends a message through the Telegram channel
func (tc *TelegramChannel) Send(ctx context.Context, message *Message) error {
	tc.logger.Debug("sending telegram message",
		zap.String("id", message.ID),
		zap.String("type", message.Type),
	)

	// Determine chat ID - use configured default or from metadata
	chatID := tc.config.ChatID
	if messageChId, ok := message.Metadata["chat_id"].(string); ok && messageChId != "" {
		chatID = messageChId
	}

	if chatID == "" {
		return fmt.Errorf("no chat ID specified for telegram message")
	}

	// Handle different message types
	switch message.Type {
	case "alert":
		if alertData, ok := message.Metadata["alert"].(map[string]interface{}); ok {
			err := tc.notifier.SendAlert(chatID, alertData)
			if err != nil {
				tc.metrics.IncrementNotificationsFailed("telegram", "send_error")
				return fmt.Errorf("failed to send telegram alert: %w", err)
			}
		} else {
			// Fall back to regular message
			err := tc.notifier.Send(chatID, message.Content)
			if err != nil {
				tc.metrics.IncrementNotificationsFailed("telegram", "send_error")
				return fmt.Errorf("failed to send telegram message: %w", err)
			}
		}
	case "rich":
		if payload, ok := message.Metadata["telegram_payload"].(NotificationPayload); ok {
			err := tc.notifier.SendRichNotification(payload)
			if err != nil {
				tc.metrics.IncrementNotificationsFailed("telegram", "send_error")
				return fmt.Errorf("failed to send rich telegram notification: %w", err)
			}
		} else {
			err := tc.notifier.Send(chatID, message.Content)
			if err != nil {
				tc.metrics.IncrementNotificationsFailed("telegram", "send_error")
				return fmt.Errorf("failed to send telegram message: %w", err)
			}
		}
	default:
		// Standard text message
		err := tc.notifier.Send(chatID, message.Content)
		if err != nil {
			tc.metrics.IncrementNotificationsFailed("telegram", "send_error")
			return fmt.Errorf("failed to send telegram message: %w", err)
		}
	}

	tc.metrics.IncrementNotificationsSent("telegram")
	tc.logger.Info("telegram message sent successfully",
		zap.String("id", message.ID),
		zap.String("chat_id", chatID),
	)

	return nil
}

// Close closes the Telegram channel
func (tc *TelegramChannel) Close() error {
	tc.logger.Info("closing telegram channel")
	return nil
}

// Health checks the health of the Telegram channel
func (tc *TelegramChannel) Health() error {
	// Simple health check - could be enhanced to ping Telegram API
	if tc.config.BotToken == "" {
		return fmt.Errorf("telegram bot token not configured")
	}
	return nil
}

// GetNotifier returns the underlying TelegramNotifier for webhook handling
func (tc *TelegramChannel) GetNotifier() *TelegramNotifier {
	return tc.notifier
}

// HandleWebhook handles Telegram webhook updates via the channel
func (tc *TelegramChannel) HandleWebhook(ctx context.Context, update Update) error {
	return tc.notifier.HandleWebhook(ctx, update)
}

// LoadTelegramConfig loads Telegram configuration from environment variables
func LoadTelegramConfig() *TelegramConfig {
	return &TelegramConfig{
		Enabled:    os.Getenv("TELEGRAM_ENABLED") == "true",
		BotToken:   os.Getenv("TELEGRAM_BOT_TOKEN"),
		WebhookURL: os.Getenv("TELEGRAM_WEBHOOK_URL"),
		ChatID:     os.Getenv("TELEGRAM_CHAT_ID"),
	}
}