package channels

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/prowzi/notifier/internal/config"
	"github.com/prowzi/notifier/internal/metrics"
	"go.uber.org/zap"
)

// DiscordChannel represents a Discord notification channel
type DiscordChannel struct {
	config  *config.DiscordConfig
	logger  *zap.Logger
	metrics *metrics.ChannelMetrics
	client  *http.Client
	bot     *DiscordBot
}

// DiscordBot represents the Discord bot instance
type DiscordBot struct {
	token          string
	applicationID  string
	guildID        string
	client         *http.Client
	commands       map[string]SlashCommandHandler
	components     map[string]ComponentHandler
}

// SlashCommandHandler handles Discord slash commands
type SlashCommandHandler func(ctx context.Context, interaction *DiscordInteraction) error

// ComponentHandler handles Discord component interactions
type ComponentHandler func(ctx context.Context, interaction *DiscordInteraction) error

// DiscordInteraction represents a Discord interaction
type DiscordInteraction struct {
	ID            string                      `json:"id"`
	ApplicationID string                      `json:"application_id"`
	Type          int                         `json:"type"`
	Data          *DiscordInteractionData     `json:"data,omitempty"`
	GuildID       string                      `json:"guild_id,omitempty"`
	ChannelID     string                      `json:"channel_id,omitempty"`
	Member        *DiscordMember              `json:"member,omitempty"`
	User          *DiscordUser                `json:"user,omitempty"`
	Token         string                      `json:"token"`
	Version       int                         `json:"version"`
	Message       *DiscordMessage             `json:"message,omitempty"`
}

// DiscordInteractionData represents interaction data
type DiscordInteractionData struct {
	ID            string                      `json:"id,omitempty"`
	Name          string                      `json:"name,omitempty"`
	Type          int                         `json:"type,omitempty"`
	Resolved      *DiscordResolvedData        `json:"resolved,omitempty"`
	Options       []DiscordApplicationCommandOption `json:"options,omitempty"`
	CustomID      string                      `json:"custom_id,omitempty"`
	ComponentType int                         `json:"component_type,omitempty"`
	Values        []string                    `json:"values,omitempty"`
	TargetID      string                      `json:"target_id,omitempty"`
}

// DiscordResolvedData represents resolved interaction data
type DiscordResolvedData struct {
	Users    map[string]*DiscordUser    `json:"users,omitempty"`
	Members  map[string]*DiscordMember  `json:"members,omitempty"`
	Roles    map[string]*DiscordRole    `json:"roles,omitempty"`
	Channels map[string]*DiscordChannel `json:"channels,omitempty"`
}

// DiscordUser represents a Discord user
type DiscordUser struct {
	ID            string `json:"id"`
	Username      string `json:"username"`
	Discriminator string `json:"discriminator"`
	Avatar        string `json:"avatar,omitempty"`
	Bot           bool   `json:"bot,omitempty"`
	System        bool   `json:"system,omitempty"`
}

// DiscordMember represents a Discord guild member
type DiscordMember struct {
	User         *DiscordUser `json:"user,omitempty"`
	Nick         string       `json:"nick,omitempty"`
	Avatar       string       `json:"avatar,omitempty"`
	Roles        []string     `json:"roles"`
	JoinedAt     string       `json:"joined_at"`
	PremiumSince string       `json:"premium_since,omitempty"`
	Permissions  string       `json:"permissions,omitempty"`
}

// DiscordRole represents a Discord role
type DiscordRole struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Color       int    `json:"color"`
	Hoist       bool   `json:"hoist"`
	Position    int    `json:"position"`
	Permissions string `json:"permissions"`
	Managed     bool   `json:"managed"`
	Mentionable bool   `json:"mentionable"`
}

// DiscordApplicationCommand represents a Discord application command
type DiscordApplicationCommand struct {
	ID                       string                             `json:"id,omitempty"`
	Type                     int                                `json:"type,omitempty"`
	ApplicationID            string                             `json:"application_id,omitempty"`
	GuildID                  string                             `json:"guild_id,omitempty"`
	Name                     string                             `json:"name"`
	NameLocalizations        map[string]string                  `json:"name_localizations,omitempty"`
	Description              string                             `json:"description"`
	DescriptionLocalizations map[string]string                  `json:"description_localizations,omitempty"`
	Options                  []DiscordApplicationCommandOption `json:"options,omitempty"`
	DefaultMemberPermissions string                             `json:"default_member_permissions,omitempty"`
	DMPermission             bool                               `json:"dm_permission,omitempty"`
	NSFW                     bool                               `json:"nsfw,omitempty"`
	Version                  string                             `json:"version,omitempty"`
}

// DiscordApplicationCommandOption represents a command option
type DiscordApplicationCommandOption struct {
	Type                     int                                `json:"type"`
	Name                     string                             `json:"name"`
	NameLocalizations        map[string]string                  `json:"name_localizations,omitempty"`
	Description              string                             `json:"description"`
	DescriptionLocalizations map[string]string                  `json:"description_localizations,omitempty"`
	Required                 bool                               `json:"required,omitempty"`
	Choices                  []DiscordApplicationCommandOptionChoice `json:"choices,omitempty"`
	Options                  []DiscordApplicationCommandOption `json:"options,omitempty"`
	ChannelTypes             []int                              `json:"channel_types,omitempty"`
	MinValue                 interface{}                        `json:"min_value,omitempty"`
	MaxValue                 interface{}                        `json:"max_value,omitempty"`
	MinLength                int                                `json:"min_length,omitempty"`
	MaxLength                int                                `json:"max_length,omitempty"`
	Autocomplete             bool                               `json:"autocomplete,omitempty"`
	Value                    interface{}                        `json:"value,omitempty"`
	Focused                  bool                               `json:"focused,omitempty"`
}

// DiscordApplicationCommandOptionChoice represents a command option choice
type DiscordApplicationCommandOptionChoice struct {
	Name              string            `json:"name"`
	NameLocalizations map[string]string `json:"name_localizations,omitempty"`
	Value             interface{}       `json:"value"`
}

// DiscordInteractionResponse represents a Discord interaction response
type DiscordInteractionResponse struct {
	Type int                                 `json:"type"`
	Data *DiscordInteractionResponseData    `json:"data,omitempty"`
}

// DiscordInteractionResponseData represents interaction response data
type DiscordInteractionResponseData struct {
	TTS             bool                    `json:"tts,omitempty"`
	Content         string                  `json:"content,omitempty"`
	Embeds          []DiscordEmbed          `json:"embeds,omitempty"`
	AllowedMentions *DiscordAllowedMentions `json:"allowed_mentions,omitempty"`
	Flags           int                     `json:"flags,omitempty"`
	Components      []DiscordComponent      `json:"components,omitempty"`
	Attachments     []DiscordAttachment     `json:"attachments,omitempty"`
}

// DiscordAllowedMentions represents allowed mentions in a message
type DiscordAllowedMentions struct {
	Parse       []string `json:"parse,omitempty"`
	Roles       []string `json:"roles,omitempty"`
	Users       []string `json:"users,omitempty"`
	RepliedUser bool     `json:"replied_user,omitempty"`
}

// DiscordComponent represents a Discord message component
type DiscordComponent struct {
	Type        int                 `json:"type"`
	CustomID    string              `json:"custom_id,omitempty"`
	Disabled    bool                `json:"disabled,omitempty"`
	Style       int                 `json:"style,omitempty"`
	Label       string              `json:"label,omitempty"`
	Emoji       *DiscordEmoji       `json:"emoji,omitempty"`
	URL         string              `json:"url,omitempty"`
	Options     []DiscordSelectOption `json:"options,omitempty"`
	Placeholder string              `json:"placeholder,omitempty"`
	MinValues   int                 `json:"min_values,omitempty"`
	MaxValues   int                 `json:"max_values,omitempty"`
	Components  []DiscordComponent  `json:"components,omitempty"`
}

// DiscordEmoji represents a Discord emoji
type DiscordEmoji struct {
	ID            string   `json:"id,omitempty"`
	Name          string   `json:"name,omitempty"`
	Roles         []string `json:"roles,omitempty"`
	User          *DiscordUser `json:"user,omitempty"`
	RequireColons bool     `json:"require_colons,omitempty"`
	Managed       bool     `json:"managed,omitempty"`
	Animated      bool     `json:"animated,omitempty"`
	Available     bool     `json:"available,omitempty"`
}

// DiscordSelectOption represents a select menu option
type DiscordSelectOption struct {
	Label       string        `json:"label"`
	Value       string        `json:"value"`
	Description string        `json:"description,omitempty"`
	Emoji       *DiscordEmoji `json:"emoji,omitempty"`
	Default     bool          `json:"default,omitempty"`
}

// DiscordAttachment represents a Discord attachment
type DiscordAttachment struct {
	ID          string `json:"id"`
	Filename    string `json:"filename"`
	Description string `json:"description,omitempty"`
	ContentType string `json:"content_type,omitempty"`
	Size        int    `json:"size"`
	URL         string `json:"url"`
	ProxyURL    string `json:"proxy_url"`
	Height      int    `json:"height,omitempty"`
	Width       int    `json:"width,omitempty"`
	Ephemeral   bool   `json:"ephemeral,omitempty"`
}

// DiscordMessage represents a Discord webhook message payload
type DiscordMessage struct {
	Content   string           `json:"content,omitempty"`
	Username  string           `json:"username,omitempty"`
	AvatarURL string           `json:"avatar_url,omitempty"`
	TTS       bool             `json:"tts,omitempty"`
	Embeds    []DiscordEmbed   `json:"embeds,omitempty"`
}

// DiscordEmbed represents a Discord embed
type DiscordEmbed struct {
	Title       string                 `json:"title,omitempty"`
	Description string                 `json:"description,omitempty"`
	URL         string                 `json:"url,omitempty"`
	Color       int                    `json:"color,omitempty"`
	Timestamp   string                 `json:"timestamp,omitempty"`
	Footer      *DiscordEmbedFooter    `json:"footer,omitempty"`
	Image       *DiscordEmbedImage     `json:"image,omitempty"`
	Thumbnail   *DiscordEmbedThumbnail `json:"thumbnail,omitempty"`
	Author      *DiscordEmbedAuthor    `json:"author,omitempty"`
	Fields      []DiscordEmbedField    `json:"fields,omitempty"`
}

// DiscordEmbedFooter represents a Discord embed footer
type DiscordEmbedFooter struct {
	Text    string `json:"text"`
	IconURL string `json:"icon_url,omitempty"`
}

// DiscordEmbedImage represents a Discord embed image
type DiscordEmbedImage struct {
	URL    string `json:"url"`
	Height int    `json:"height,omitempty"`
	Width  int    `json:"width,omitempty"`
}

// DiscordEmbedThumbnail represents a Discord embed thumbnail
type DiscordEmbedThumbnail struct {
	URL    string `json:"url"`
	Height int    `json:"height,omitempty"`
	Width  int    `json:"width,omitempty"`
}

// DiscordEmbedAuthor represents a Discord embed author
type DiscordEmbedAuthor struct {
	Name    string `json:"name"`
	URL     string `json:"url,omitempty"`
	IconURL string `json:"icon_url,omitempty"`
}

// DiscordEmbedField represents a Discord embed field
type DiscordEmbedField struct {
	Name   string `json:"name"`
	Value  string `json:"value"`
	Inline bool   `json:"inline,omitempty"`
}

// NewDiscordChannel creates a new Discord notification channel
func NewDiscordChannel(config *config.DiscordConfig, logger *zap.Logger, metrics *metrics.Metrics) (*DiscordChannel, error) {
	if config.WebhookURL == "" {
		return nil, fmt.Errorf("Discord webhook URL is required")
	}

	client := &http.Client{
		Timeout: config.Timeout,
	}

	return &DiscordChannel{
		config:  config,
		logger:  logger,
		metrics: metrics.NewChannelMetrics("discord"),
		client:  client,
	}, nil
}

// Send sends a notification message to Discord
func (dc *DiscordChannel) Send(ctx context.Context, message *Message) error {
	timer := metrics.NewTimer()
	
	discordMsg, err := dc.formatMessage(message)
	if err != nil {
		dc.metrics.RecordFailure("format_error", time.Since(timer.start))
		return fmt.Errorf("failed to format Discord message: %w", err)
	}

	payload, err := json.Marshal(discordMsg)
	if err != nil {
		dc.metrics.RecordFailure("marshal_error", time.Since(timer.start))
		return fmt.Errorf("failed to marshal Discord message: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", dc.config.WebhookURL, bytes.NewBuffer(payload))
	if err != nil {
		dc.metrics.RecordFailure("request_error", time.Since(timer.start))
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Prowzi-Notifier/1.0")

	dc.logger.Debug("Sending Discord notification",
		zap.String("message_id", message.ID),
		zap.String("type", message.Type),
		zap.Int("payload_size", len(payload)),
	)

	resp, err := dc.client.Do(req)
	if err != nil {
		dc.metrics.RecordFailure("network_error", time.Since(timer.start))
		return fmt.Errorf("failed to send Discord webhook: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		dc.metrics.RecordFailure("http_error", time.Since(timer.start))
		return fmt.Errorf("Discord webhook returned status %d", resp.StatusCode)
	}

	duration := time.Since(timer.start)
	dc.metrics.RecordSuccess(duration)
	
	dc.logger.Info("Discord notification sent successfully",
		zap.String("message_id", message.ID),
		zap.String("type", message.Type),
		zap.Duration("duration", duration),
	)

	return nil
}

// formatMessage converts a generic message to a Discord-specific format
func (dc *DiscordChannel) formatMessage(message *Message) (*DiscordMessage, error) {
	discordMsg := &DiscordMessage{
		Username:  dc.config.Username,
		AvatarURL: dc.config.AvatarURL,
	}

	// Handle different message types
	switch message.Type {
	case "text", "plain":
		discordMsg.Content = message.Content
		
	case "rich", "embed":
		embed, err := dc.createEmbed(message)
		if err != nil {
			return nil, fmt.Errorf("failed to create embed: %w", err)
		}
		discordMsg.Embeds = []DiscordEmbed{*embed}
		
	case "alert", "warning":
		embed := dc.createAlertEmbed(message)
		discordMsg.Embeds = []DiscordEmbed{*embed}
		
	case "error", "critical":
		embed := dc.createErrorEmbed(message)
		discordMsg.Embeds = []DiscordEmbed{*embed}
		
	case "success", "info":
		embed := dc.createInfoEmbed(message)
		discordMsg.Embeds = []DiscordEmbed{*embed}
		
	default:
		// Default to simple text message
		discordMsg.Content = message.Content
	}

	return discordMsg, nil
}

// createEmbed creates a generic embed from message metadata
func (dc *DiscordChannel) createEmbed(message *Message) (*DiscordEmbed, error) {
	embed := &DiscordEmbed{
		Description: message.Content,
		Timestamp:   message.Timestamp.Format(time.RFC3339),
		Footer: &DiscordEmbedFooter{
			Text: "Prowzi Notification System",
		},
	}

	// Extract embed data from metadata
	if message.Metadata != nil {
		if title, ok := message.Metadata["title"].(string); ok {
			embed.Title = title
		}
		if color, ok := message.Metadata["color"].(int); ok {
			embed.Color = color
		}
		if url, ok := message.Metadata["url"].(string); ok {
			embed.URL = url
		}
		if imageURL, ok := message.Metadata["image_url"].(string); ok {
			embed.Image = &DiscordEmbedImage{URL: imageURL}
		}
		if thumbnailURL, ok := message.Metadata["thumbnail_url"].(string); ok {
			embed.Thumbnail = &DiscordEmbedThumbnail{URL: thumbnailURL}
		}
		
		// Handle fields
		if fields, ok := message.Metadata["fields"].([]interface{}); ok {
			for _, field := range fields {
				if fieldMap, ok := field.(map[string]interface{}); ok {
					embedField := DiscordEmbedField{}
					if name, ok := fieldMap["name"].(string); ok {
						embedField.Name = name
					}
					if value, ok := fieldMap["value"].(string); ok {
						embedField.Value = value
					}
					if inline, ok := fieldMap["inline"].(bool); ok {
						embedField.Inline = inline
					}
					embed.Fields = append(embed.Fields, embedField)
				}
			}
		}
	}

	return embed, nil
}

// createAlertEmbed creates an alert-style embed
func (dc *DiscordChannel) createAlertEmbed(message *Message) *DiscordEmbed {
	return &DiscordEmbed{
		Title:       "⚠️ Alert",
		Description: message.Content,
		Color:       0xFFA500, // Orange
		Timestamp:   message.Timestamp.Format(time.RFC3339),
		Footer: &DiscordEmbedFooter{
			Text: "Prowzi Alert System",
		},
	}
}

// createErrorEmbed creates an error-style embed
func (dc *DiscordChannel) createErrorEmbed(message *Message) *DiscordEmbed {
	return &DiscordEmbed{
		Title:       "🚨 Error",
		Description: message.Content,
		Color:       0xFF0000, // Red
		Timestamp:   message.Timestamp.Format(time.RFC3339),
		Footer: &DiscordEmbedFooter{
			Text: "Prowzi Error System",
		},
	}
}

// createInfoEmbed creates an info-style embed
func (dc *DiscordChannel) createInfoEmbed(message *Message) *DiscordEmbed {
	return &DiscordEmbed{
		Title:       "ℹ️ Information",
		Description: message.Content,
		Color:       0x00FF00, // Green
		Timestamp:   message.Timestamp.Format(time.RFC3339),
		Footer: &DiscordEmbedFooter{
			Text: "Prowzi Information System",
		},
	}
}

// Health checks the health of the Discord channel
func (dc *DiscordChannel) Health(ctx context.Context) error {
	// Create a test message
	testMsg := &DiscordMessage{
		Content:  "Health check",
		Username: dc.config.Username,
	}

	payload, err := json.Marshal(testMsg)
	if err != nil {
		return fmt.Errorf("failed to marshal test message: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", dc.config.WebhookURL, bytes.NewBuffer(payload))
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Prowzi-Notifier/1.0")

	resp, err := dc.client.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("Discord health check returned status %d", resp.StatusCode)
	}

	return nil
}

// Close closes the Discord channel (cleanup if needed)
func (dc *DiscordChannel) Close() error {
	dc.logger.Debug("Closing Discord channel")
	// No specific cleanup needed for Discord webhook
	return nil
}

// GetName returns the channel name
func (dc *DiscordChannel) GetName() string {
	return "discord"
}

// GetConfig returns the channel configuration
func (dc *DiscordChannel) GetConfig() interface{} {
	return dc.config
}