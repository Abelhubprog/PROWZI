package main

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prowzi/notifier/internal/channels"
	"github.com/prowzi/notifier/internal/queue"
	"go.uber.org/zap"
)

// NotificationRequest represents an incoming notification request
type NotificationRequest struct {
	Type      string                 `json:"type" binding:"required"`
	Channels  []string               `json:"channels" binding:"required"`
	Content   string                 `json:"content,omitempty"`
	Template  string                 `json:"template,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Priority  string                 `json:"priority,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	ScheduleFor *time.Time           `json:"schedule_for,omitempty"`
}

// NotificationResponse represents the response to a notification request
type NotificationResponse struct {
	ID      string `json:"id"`
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

// ChannelInfo represents information about a notification channel
type ChannelInfo struct {
	Name    string `json:"name"`
	Type    string `json:"type"`
	Enabled bool   `json:"enabled"`
	Config  map[string]interface{} `json:"config,omitempty"`
}

// StatsResponse represents service statistics
type StatsResponse struct {
	TotalNotifications     int64             `json:"total_notifications"`
	NotificationsSent      map[string]int64  `json:"notifications_sent"`
	NotificationsFailed    map[string]int64  `json:"notifications_failed"`
	ActiveChannels         int               `json:"active_channels"`
	QueueSize              int               `json:"queue_size"`
	Uptime                 string            `json:"uptime"`
	LastProcessedAt        *time.Time        `json:"last_processed_at,omitempty"`
}

// healthHandler handles health check requests
func (ns *NotificationService) healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "ok",
		"service":   "prowzi-notifier",
		"version":   "0.1.0",
		"timestamp": time.Now().UTC(),
	})
}

// readinessHandler handles readiness check requests
func (ns *NotificationService) readinessHandler(c *gin.Context) {
	// Check if all critical components are ready
	ready := true
	details := make(map[string]interface{})

	// Check channels
	channelStatus := make(map[string]bool)
	for name, channel := range ns.channels {
		// Assuming channels have a Health() method
		channelStatus[name] = true // Simplified check
	}
	details["channels"] = channelStatus

	// Check queue
	queueReady := true // Simplified check
	details["queue"] = queueReady

	if !ready || !queueReady {
		ready = false
	}

	status := http.StatusOK
	if !ready {
		status = http.StatusServiceUnavailable
	}

	c.JSON(status, gin.H{
		"ready":   ready,
		"details": details,
	})
}

// createNotificationHandler handles notification creation requests
func (ns *NotificationService) createNotificationHandler(c *gin.Context) {
	var req NotificationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		ns.logger.Error("invalid notification request", zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_request",
			"message": err.Error(),
		})
		return
	}

	// Validate channels
	for _, channelName := range req.Channels {
		if _, exists := ns.channels[channelName]; !exists {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "invalid_channel",
				"message": "channel '" + channelName + "' not found",
			})
			return
		}
	}

	// Create notification
	notification := &queue.Notification{
		ID:        generateNotificationID(),
		Type:      req.Type,
		Channels:  req.Channels,
		Content:   req.Content,
		Template:  req.Template,
		Data:      req.Data,
		Priority:  req.Priority,
		Metadata:  req.Metadata,
		Timestamp: time.Now().UTC(),
	}

	// Set default priority if not specified
	if notification.Priority == "" {
		notification.Priority = "normal"
	}

	// Send to queue
	if err := ns.queue.Send(c.Request.Context(), notification); err != nil {
		ns.logger.Error("failed to queue notification", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "queue_error",
			"message": "failed to queue notification",
		})
		return
	}

	ns.logger.Info("notification queued",
		zap.String("id", notification.ID),
		zap.String("type", notification.Type),
		zap.Strings("channels", notification.Channels),
	)

	c.JSON(http.StatusCreated, NotificationResponse{
		ID:     notification.ID,
		Status: "queued",
	})
}

// getNotificationStatusHandler handles notification status requests
func (ns *NotificationService) getNotificationStatusHandler(c *gin.Context) {
	notificationID := c.Param("id")
	if notificationID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "missing_id",
			"message": "notification ID is required",
		})
		return
	}

	// TODO: Implement actual status tracking
	// For now, return a mock response
	c.JSON(http.StatusOK, gin.H{
		"id":     notificationID,
		"status": "sent",
		"sent_at": time.Now().UTC(),
	})
}

// getChannelsHandler handles channel listing requests
func (ns *NotificationService) getChannelsHandler(c *gin.Context) {
	channels := make([]ChannelInfo, 0, len(ns.channels))

	for name := range ns.channels {
		channelInfo := ChannelInfo{
			Name:    name,
			Type:    name, // Simplified
			Enabled: true,
		}
		channels = append(channels, channelInfo)
	}

	c.JSON(http.StatusOK, gin.H{
		"channels": channels,
		"total":    len(channels),
	})
}

// testChannelHandler handles channel testing requests
func (ns *NotificationService) testChannelHandler(c *gin.Context) {
	channelName := c.Param("channel")
	channel, exists := ns.channels[channelName]
	if !exists {
		c.JSON(http.StatusNotFound, gin.H{
			"error":   "channel_not_found",
			"message": "channel '" + channelName + "' not found",
		})
		return
	}

	// Create test message
	testMessage := &channels.Message{
		ID:      "test-" + strconv.FormatInt(time.Now().Unix(), 10),
		Type:    "test",
		Content: "🧪 Test notification from Prowzi Notification Service\n\nThis is a test message to verify channel functionality.",
		Priority: "low",
		Metadata: map[string]interface{}{
			"test": true,
			"timestamp": time.Now().UTC(),
		},
		Timestamp: time.Now().UTC(),
	}

	// Send test message
	if err := channel.Send(c.Request.Context(), testMessage); err != nil {
		ns.logger.Error("failed to send test message",
			zap.String("channel", channelName),
			zap.Error(err),
		)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "send_failed",
			"message": "failed to send test message: " + err.Error(),
		})
		return
	}

	ns.logger.Info("test message sent", zap.String("channel", channelName))
	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "test message sent successfully",
		"channel": channelName,
	})
}

// telegramWebhookHandler handles incoming Telegram webhook updates
func (ns *NotificationService) telegramWebhookHandler(c *gin.Context) {
	var update channels.Update
	if err := c.ShouldBindJSON(&update); err != nil {
		ns.logger.Error("invalid telegram webhook payload", zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_payload"})
		return
	}

	// Get Telegram channel
	telegramChannel, exists := ns.channels["telegram"]
	if !exists {
		ns.logger.Error("telegram channel not found")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "telegram_channel_not_found"})
		return
	}

	// Cast to TelegramNotifier to access webhook handler
	if telegramNotifier, ok := telegramChannel.(*channels.TelegramNotifier); ok {
		if err := telegramNotifier.HandleWebhook(c.Request.Context(), update); err != nil {
			ns.logger.Error("failed to handle telegram webhook", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "webhook_handler_failed"})
			return
		}
	} else {
		ns.logger.Error("telegram channel type assertion failed")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "channel_type_error"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

// getStatsHandler handles statistics requests
func (ns *NotificationService) getStatsHandler(c *gin.Context) {
	// TODO: Implement actual metrics collection
	// For now, return mock data
	stats := StatsResponse{
		TotalNotifications: 1000,
		NotificationsSent: map[string]int64{
			"telegram": 450,
			"discord":  300,
			"slack":    150,
			"email":    100,
		},
		NotificationsFailed: map[string]int64{
			"telegram": 5,
			"discord":  3,
			"slack":    2,
			"email":    1,
		},
		ActiveChannels: len(ns.channels),
		QueueSize:      0, // Would get from queue
		Uptime:         "72h45m30s",
	}

	c.JSON(http.StatusOK, stats)
}

// reloadTemplatesHandler handles template reload requests
func (ns *NotificationService) reloadTemplatesHandler(c *gin.Context) {
	if err := ns.templates.Reload(); err != nil {
		ns.logger.Error("failed to reload templates", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "reload_failed",
			"message": err.Error(),
		})
		return
	}

	ns.logger.Info("templates reloaded successfully")
	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "templates reloaded successfully",
	})
}

// generateNotificationID generates a unique notification ID
func generateNotificationID() string {
	return "notif_" + strconv.FormatInt(time.Now().UnixNano(), 36)
}