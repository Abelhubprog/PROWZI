package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config represents the application configuration
type Config struct {
	Server        ServerConfig        `json:"server"`
	Queue         QueueConfig         `json:"queue"`
	Discord       DiscordConfig       `json:"discord"`
	Slack         SlackConfig         `json:"slack"`
	Email         EmailConfig         `json:"email"`
	Webhook       WebhookConfig       `json:"webhook"`
	Metrics       MetricsConfig       `json:"metrics"`
	LogLevel      string              `json:"log_level"`
	TemplatesPath string              `json:"templates_path"`
	Database      DatabaseConfig      `json:"database"`
}

// ServerConfig represents HTTP server configuration
type ServerConfig struct {
	Port int    `json:"port"`
	Mode string `json:"mode"`
}

// QueueConfig represents message queue configuration
type QueueConfig struct {
	Type        string        `json:"type"`         // redis, memory, postgres
	URL         string        `json:"url"`
	MaxRetries  int           `json:"max_retries"`
	RetryDelay  time.Duration `json:"retry_delay"`
	BatchSize   int           `json:"batch_size"`
	WorkerCount int           `json:"worker_count"`
}

// DiscordConfig represents Discord notification configuration
type DiscordConfig struct {
	Enabled    bool   `json:"enabled"`
	WebhookURL string `json:"webhook_url"`
	Username   string `json:"username"`
	AvatarURL  string `json:"avatar_url"`
	Timeout    time.Duration `json:"timeout"`
}

// SlackConfig represents Slack notification configuration
type SlackConfig struct {
	Enabled    bool   `json:"enabled"`
	WebhookURL string `json:"webhook_url"`
	Channel    string `json:"channel"`
	Username   string `json:"username"`
	IconEmoji  string `json:"icon_emoji"`
	Timeout    time.Duration `json:"timeout"`
}

// EmailConfig represents email notification configuration
type EmailConfig struct {
	Enabled    bool   `json:"enabled"`
	SMTPHost   string `json:"smtp_host"`
	SMTPPort   int    `json:"smtp_port"`
	Username   string `json:"username"`
	Password   string `json:"password"`
	FromEmail  string `json:"from_email"`
	FromName   string `json:"from_name"`
	UseTLS     bool   `json:"use_tls"`
	Timeout    time.Duration `json:"timeout"`
}

// WebhookConfig represents webhook notification configuration
type WebhookConfig struct {
	Enabled     bool          `json:"enabled"`
	DefaultURL  string        `json:"default_url"`
	Timeout     time.Duration `json:"timeout"`
	MaxRetries  int           `json:"max_retries"`
	RetryDelay  time.Duration `json:"retry_delay"`
	SecretKey   string        `json:"secret_key"`
}

// MetricsConfig represents metrics configuration
type MetricsConfig struct {
	Enabled bool `json:"enabled"`
	Port    int  `json:"port"`
}

// DatabaseConfig represents database configuration
type DatabaseConfig struct {
	URL             string        `json:"url"`
	MaxConnections  int           `json:"max_connections"`
	ConnTimeout     time.Duration `json:"connection_timeout"`
	IdleTimeout     time.Duration `json:"idle_timeout"`
	MigrationsPath  string        `json:"migrations_path"`
}

// Load loads configuration from environment variables
func Load() (*Config, error) {
	config := &Config{
		Server: ServerConfig{
			Port: getEnvInt("SERVER_PORT", 8080),
			Mode: getEnvString("SERVER_MODE", "development"),
		},
		Queue: QueueConfig{
			Type:        getEnvString("QUEUE_TYPE", "memory"),
			URL:         getEnvString("QUEUE_URL", ""),
			MaxRetries:  getEnvInt("QUEUE_MAX_RETRIES", 3),
			RetryDelay:  getEnvDuration("QUEUE_RETRY_DELAY", 5*time.Second),
			BatchSize:   getEnvInt("QUEUE_BATCH_SIZE", 10),
			WorkerCount: getEnvInt("QUEUE_WORKER_COUNT", 4),
		},
		Discord: DiscordConfig{
			Enabled:    getEnvBool("DISCORD_ENABLED", false),
			WebhookURL: getEnvString("DISCORD_WEBHOOK_URL", ""),
			Username:   getEnvString("DISCORD_USERNAME", "Prowzi Bot"),
			AvatarURL:  getEnvString("DISCORD_AVATAR_URL", ""),
			Timeout:    getEnvDuration("DISCORD_TIMEOUT", 30*time.Second),
		},
		Slack: SlackConfig{
			Enabled:    getEnvBool("SLACK_ENABLED", false),
			WebhookURL: getEnvString("SLACK_WEBHOOK_URL", ""),
			Channel:    getEnvString("SLACK_CHANNEL", "#general"),
			Username:   getEnvString("SLACK_USERNAME", "Prowzi Bot"),
			IconEmoji:  getEnvString("SLACK_ICON_EMOJI", ":robot_face:"),
			Timeout:    getEnvDuration("SLACK_TIMEOUT", 30*time.Second),
		},
		Email: EmailConfig{
			Enabled:   getEnvBool("EMAIL_ENABLED", false),
			SMTPHost:  getEnvString("SMTP_HOST", ""),
			SMTPPort:  getEnvInt("SMTP_PORT", 587),
			Username:  getEnvString("SMTP_USERNAME", ""),
			Password:  getEnvString("SMTP_PASSWORD", ""),
			FromEmail: getEnvString("EMAIL_FROM", "noreply@prowzi.com"),
			FromName:  getEnvString("EMAIL_FROM_NAME", "Prowzi Notifications"),
			UseTLS:    getEnvBool("SMTP_USE_TLS", true),
			Timeout:   getEnvDuration("EMAIL_TIMEOUT", 30*time.Second),
		},
		Webhook: WebhookConfig{
			Enabled:    getEnvBool("WEBHOOK_ENABLED", false),
			DefaultURL: getEnvString("WEBHOOK_DEFAULT_URL", ""),
			Timeout:    getEnvDuration("WEBHOOK_TIMEOUT", 30*time.Second),
			MaxRetries: getEnvInt("WEBHOOK_MAX_RETRIES", 3),
			RetryDelay: getEnvDuration("WEBHOOK_RETRY_DELAY", 5*time.Second),
			SecretKey:  getEnvString("WEBHOOK_SECRET_KEY", ""),
		},
		Metrics: MetricsConfig{
			Enabled: getEnvBool("METRICS_ENABLED", true),
			Port:    getEnvInt("METRICS_PORT", 9090),
		},
		LogLevel:      getEnvString("LOG_LEVEL", "info"),
		TemplatesPath: getEnvString("TEMPLATES_PATH", "./templates"),
		Database: DatabaseConfig{
			URL:             getEnvString("DATABASE_URL", "postgres://localhost/prowzi_notifications"),
			MaxConnections:  getEnvInt("DB_MAX_CONNECTIONS", 20),
			ConnTimeout:     getEnvDuration("DB_CONN_TIMEOUT", 10*time.Second),
			IdleTimeout:     getEnvDuration("DB_IDLE_TIMEOUT", 10*time.Minute),
			MigrationsPath:  getEnvString("DB_MIGRATIONS_PATH", "./migrations"),
		},
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return config, nil
}

// Validate validates the configuration
func (c *Config) Validate() error {
	if c.Server.Port < 1 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid server port: %d", c.Server.Port)
	}

	if c.Discord.Enabled && c.Discord.WebhookURL == "" {
		return fmt.Errorf("Discord webhook URL is required when Discord is enabled")
	}

	if c.Slack.Enabled && c.Slack.WebhookURL == "" {
		return fmt.Errorf("Slack webhook URL is required when Slack is enabled")
	}

	if c.Email.Enabled {
		if c.Email.SMTPHost == "" {
			return fmt.Errorf("SMTP host is required when email is enabled")
		}
		if c.Email.Username == "" || c.Email.Password == "" {
			return fmt.Errorf("SMTP username and password are required when email is enabled")
		}
	}

	if c.Webhook.Enabled && c.Webhook.DefaultURL == "" {
		return fmt.Errorf("Default webhook URL is required when webhook is enabled")
	}

	if c.Queue.Type == "redis" && c.Queue.URL == "" {
		return fmt.Errorf("Queue URL is required for Redis queue type")
	}

	if c.Queue.Type == "postgres" && c.Database.URL == "" {
		return fmt.Errorf("Database URL is required for PostgreSQL queue type")
	}

	return nil
}

// Helper functions for environment variable parsing

func getEnvString(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}