package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prowzi/notifier/internal/channels"
	"github.com/prowzi/notifier/internal/config"
	"github.com/prowzi/notifier/internal/metrics"
	"github.com/prowzi/notifier/internal/queue"
	"github.com/prowzi/notifier/internal/templates"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
)

// NotificationService represents the main notification service
type NotificationService struct {
	config    *config.Config
	logger    *zap.Logger
	channels  map[string]channels.Channel
	queue     queue.Queue
	metrics   *metrics.Metrics
	templates *templates.Engine
	server    *http.Server
}

// NewNotificationService creates a new notification service instance
func NewNotificationService() (*NotificationService, error) {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// Initialize logger
	logger, err := initLogger(cfg.LogLevel)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize logger: %w", err)
	}

	// Initialize metrics
	metricsInstance := metrics.NewMetrics()

	// Initialize template engine
	templateEngine, err := templates.NewEngine(cfg.TemplatesPath)
	if err != nil {
		logger.Error("failed to initialize template engine", zap.Error(err))
		return nil, fmt.Errorf("failed to initialize templates: %w", err)
	}

	// Initialize notification channels
	notificationChannels := make(map[string]channels.Channel)

	// Telegram channel
	if cfg.Telegram.Enabled {
		telegramChannel, err := channels.NewTelegramChannel(&cfg.Telegram, logger, metricsInstance)
		if err != nil {
			logger.Error("failed to initialize Telegram channel", zap.Error(err))
		} else {
			notificationChannels["telegram"] = telegramChannel
			logger.Info("Telegram channel initialized")
		}
	}

	// Discord channel
	if cfg.Discord.Enabled {
		discordChannel, err := channels.NewDiscordChannel(&cfg.Discord, logger, metricsInstance)
		if err != nil {
			logger.Error("failed to initialize Discord channel", zap.Error(err))
		} else {
			notificationChannels["discord"] = discordChannel
			logger.Info("Discord channel initialized")
		}
	}

	// Slack channel
	if cfg.Slack.Enabled {
		slackChannel, err := channels.NewSlackChannel(&cfg.Slack, logger, metricsInstance)
		if err != nil {
			logger.Error("failed to initialize Slack channel", zap.Error(err))
		} else {
			notificationChannels["slack"] = slackChannel
			logger.Info("Slack channel initialized")
		}
	}

	// Email channel
	if cfg.Email.Enabled {
		emailChannel, err := channels.NewEmailChannel(&cfg.Email, logger, metricsInstance)
		if err != nil {
			logger.Error("failed to initialize Email channel", zap.Error(err))
		} else {
			notificationChannels["email"] = emailChannel
			logger.Info("Email channel initialized")
		}
	}

	// Webhook channel
	if cfg.Webhook.Enabled {
		webhookChannel, err := channels.NewWebhookChannel(&cfg.Webhook, logger, metricsInstance)
		if err != nil {
			logger.Error("failed to initialize Webhook channel", zap.Error(err))
		} else {
			notificationChannels["webhook"] = webhookChannel
			logger.Info("Webhook channel initialized")
		}
	}

	// Initialize queue
	messageQueue, err := queue.NewQueue(cfg.Queue, logger, metricsInstance)
	if err != nil {
		logger.Error("failed to initialize queue", zap.Error(err))
		return nil, fmt.Errorf("failed to initialize queue: %w", err)
	}

	service := &NotificationService{
		config:    cfg,
		logger:    logger,
		channels:  notificationChannels,
		queue:     messageQueue,
		metrics:   metricsInstance,
		templates: templateEngine,
	}

	return service, nil
}

// Start starts the notification service
func (ns *NotificationService) Start(ctx context.Context) error {
	ns.logger.Info("Starting Prowzi Notification Service",
		zap.String("version", "0.1.0"),
		zap.Int("port", ns.config.Server.Port),
		zap.Int("channels", len(ns.channels)),
	)

	// Start the queue consumer
	go ns.processNotifications(ctx)

	// Setup HTTP server
	router := ns.setupRoutes()
	ns.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", ns.config.Server.Port),
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start metrics server
	if ns.config.Metrics.Enabled {
		go ns.startMetricsServer()
	}

	// Start HTTP server
	go func() {
		if err := ns.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			ns.logger.Fatal("failed to start server", zap.Error(err))
		}
	}()

	ns.logger.Info("Notification service started successfully")
	return nil
}

// Stop gracefully stops the notification service
func (ns *NotificationService) Stop(ctx context.Context) error {
	ns.logger.Info("Stopping notification service...")

	// Stop HTTP server
	if ns.server != nil {
		if err := ns.server.Shutdown(ctx); err != nil {
			ns.logger.Error("failed to shutdown server", zap.Error(err))
		}
	}

	// Close queue
	if err := ns.queue.Close(); err != nil {
		ns.logger.Error("failed to close queue", zap.Error(err))
	}

	// Close channels
	for name, channel := range ns.channels {
		if err := channel.Close(); err != nil {
			ns.logger.Error("failed to close channel", zap.String("channel", name), zap.Error(err))
		}
	}

	ns.logger.Info("Notification service stopped")
	return nil
}

// setupRoutes configures the HTTP routes
func (ns *NotificationService) setupRoutes() *gin.Engine {
	if ns.config.Server.Mode == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// Health check endpoint
	router.GET("/health", ns.healthHandler)
	router.GET("/ready", ns.readinessHandler)

	// API endpoints
	v1 := router.Group("/api/v1")
	{
		v1.POST("/notifications", ns.createNotificationHandler)
		v1.GET("/notifications/:id/status", ns.getNotificationStatusHandler)
		v1.GET("/channels", ns.getChannelsHandler)
		v1.POST("/test/:channel", ns.testChannelHandler)
		
		// Telegram webhook endpoint
		v1.POST("/webhooks/telegram", ns.telegramWebhookHandler)
	}

	// Admin endpoints
	admin := router.Group("/admin")
	{
		admin.GET("/stats", ns.getStatsHandler)
		admin.POST("/reload-templates", ns.reloadTemplatesHandler)
	}

	return router
}

// processNotifications processes notifications from the queue
func (ns *NotificationService) processNotifications(ctx context.Context) {
	ns.logger.Info("Starting notification processor")

	for {
		select {
		case <-ctx.Done():
			ns.logger.Info("Stopping notification processor")
			return
		default:
			notification, err := ns.queue.Receive(ctx)
			if err != nil {
				ns.logger.Error("failed to receive notification", zap.Error(err))
				time.Sleep(5 * time.Second)
				continue
			}

			if notification != nil {
				ns.processNotification(ctx, notification)
			}
		}
	}
}

// processNotification processes a single notification
func (ns *NotificationService) processNotification(ctx context.Context, notification *queue.Notification) {
	ns.logger.Info("Processing notification",
		zap.String("id", notification.ID),
		zap.String("type", notification.Type),
		zap.Strings("channels", notification.Channels),
	)

	ns.metrics.IncrementNotificationsProcessed()

	for _, channelName := range notification.Channels {
		channel, exists := ns.channels[channelName]
		if !exists {
			ns.logger.Warn("channel not found", zap.String("channel", channelName))
			ns.metrics.IncrementNotificationsFailed(channelName, "channel_not_found")
			continue
		}

		// Render template if specified
		content := notification.Content
		if notification.Template != "" {
			rendered, err := ns.templates.Render(notification.Template, notification.Data)
			if err != nil {
				ns.logger.Error("failed to render template",
					zap.String("template", notification.Template),
					zap.Error(err),
				)
				ns.metrics.IncrementNotificationsFailed(channelName, "template_error")
				continue
			}
			content = rendered
		}

		// Send notification
		err := channel.Send(ctx, &channels.Message{
			ID:        notification.ID,
			Type:      notification.Type,
			Content:   content,
			Priority:  notification.Priority,
			Metadata:  notification.Metadata,
			Timestamp: notification.Timestamp,
		})

		if err != nil {
			ns.logger.Error("failed to send notification",
				zap.String("id", notification.ID),
				zap.String("channel", channelName),
				zap.Error(err),
			)
			ns.metrics.IncrementNotificationsFailed(channelName, "send_error")
		} else {
			ns.logger.Info("notification sent successfully",
				zap.String("id", notification.ID),
				zap.String("channel", channelName),
			)
			ns.metrics.IncrementNotificationsSent(channelName)
		}
	}
}

// startMetricsServer starts the Prometheus metrics server
func (ns *NotificationService) startMetricsServer() {
	metricsRouter := gin.New()
	metricsRouter.GET("/metrics", gin.WrapH(promhttp.Handler()))

	metricsServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", ns.config.Metrics.Port),
		Handler: metricsRouter,
	}

	ns.logger.Info("Starting metrics server", zap.Int("port", ns.config.Metrics.Port))

	if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		ns.logger.Error("metrics server failed", zap.Error(err))
	}
}

// initLogger initializes the structured logger
func initLogger(level string) (*zap.Logger, error) {
	config := zap.NewProductionConfig()

	switch level {
	case "debug":
		config.Level = zap.NewAtomicLevelAt(zap.DebugLevel)
	case "info":
		config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	case "warn":
		config.Level = zap.NewAtomicLevelAt(zap.WarnLevel)
	case "error":
		config.Level = zap.NewAtomicLevelAt(zap.ErrorLevel)
	default:
		config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	}

	return config.Build()
}

func main() {
	// Create notification service
	service, err := NewNotificationService()
	if err != nil {
		log.Fatalf("Failed to create notification service: %v", err)
	}

	// Create context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start service
	if err := service.Start(ctx); err != nil {
		log.Fatalf("Failed to start notification service: %v", err)
	}

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	service.logger.Info("Received shutdown signal")

	// Create shutdown context with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	// Stop service
	if err := service.Stop(shutdownCtx); err != nil {
		service.logger.Error("Failed to stop service gracefully", zap.Error(err))
		os.Exit(1)
	}

	service.logger.Info("Service stopped gracefully")
}