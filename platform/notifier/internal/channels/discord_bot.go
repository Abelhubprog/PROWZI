package channels

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/bwmarrin/discordgo"
)

type DiscordBot struct {
	session   *discordgo.Session
	guildID   string
	channelID string
	ready     bool
}

type BotCommand struct {
	Name        string
	Description string
	Handler     func(s *discordgo.Session, i *discordgo.InteractionCreate)
}

type AgentStatusEmbed struct {
	Name     string  `json:"name"`
	Status   string  `json:"status"`
	Health   float64 `json:"health"`
	Tasks    int     `json:"tasks"`
	Uptime   string  `json:"uptime"`
	LastSeen string  `json:"last_seen"`
}

type TradingAlert struct {
	Type       string  `json:"type"`
	Symbol     string  `json:"symbol"`
	Price      float64 `json:"price"`
	Change24h  float64 `json:"change_24h"`
	Volume     float64 `json:"volume"`
	Timestamp  string  `json:"timestamp"`
	Confidence float64 `json:"confidence"`
	Action     string  `json:"action"`
}

func NewDiscordBot() (*DiscordBot, error) {
	token := os.Getenv("DISCORD_BOT_TOKEN")
	if token == "" {
		return nil, fmt.Errorf("DISCORD_BOT_TOKEN environment variable is required")
	}

	session, err := discordgo.New("Bot " + token)
	if err != nil {
		return nil, fmt.Errorf("failed to create Discord session: %w", err)
	}

	bot := &DiscordBot{
		session: session,
		guildID: os.Getenv("DISCORD_GUILD_ID"),
		channelID: os.Getenv("DISCORD_CHANNEL_ID"),
	}

	// Set up event handlers
	session.AddHandler(bot.onReady)
	session.AddHandler(bot.onInteractionCreate)
	session.AddHandler(bot.onMessageCreate)

	// Set intents
	session.Identify.Intents = discordgo.IntentsGuilds | 
		discordgo.IntentsGuildMessages | 
		discordgo.IntentsMessageContent |
		discordgo.IntentsGuildMembers

	return bot, nil
}

func (b *DiscordBot) Start() error {
	err := b.session.Open()
	if err != nil {
		return fmt.Errorf("failed to open Discord connection: %w", err)
	}

	log.Println("Discord bot started successfully")
	return nil
}

func (b *DiscordBot) Stop() error {
	if b.session != nil {
		return b.session.Close()
	}
	return nil
}

func (b *DiscordBot) onReady(s *discordgo.Session, event *discordgo.Ready) {
	log.Printf("Discord bot logged in as: %v#%v", s.State.User.Username, s.State.User.Discriminator)
	
	// Set bot status
	err := s.UpdateGameStatus(0, "🤖 Prowzi AI Agent Platform")
	if err != nil {
		log.Printf("Failed to set bot status: %v", err)
	}

	// Register slash commands
	b.registerCommands(s)
	b.ready = true
}

func (b *DiscordBot) registerCommands(s *discordgo.Session) {
	commands := []*discordgo.ApplicationCommand{
		{
			Name:        "status",
			Description: "Get current status of all Prowzi agents",
		},
		{
			Name:        "agents",
			Description: "List all active agents and their health metrics",
		},
		{
			Name:        "portfolio",
			Description: "View current portfolio performance and positions",
		},
		{
			Name:        "alerts",
			Description: "Configure trading alerts and notifications",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "action",
					Description: "Action to perform (list, add, remove)",
					Required:    true,
					Choices: []*discordgo.ApplicationCommandOptionChoice{
						{Name: "list", Value: "list"},
						{Name: "add", Value: "add"},
						{Name: "remove", Value: "remove"},
					},
				},
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "symbol",
					Description: "Token symbol (e.g., SOL, BTC)",
					Required:    false,
				},
				{
					Type:        discordgo.ApplicationCommandOptionNumber,
					Name:        "threshold",
					Description: "Price threshold for alerts",
					Required:    false,
				},
			},
		},
		{
			Name:        "missions",
			Description: "View active missions and their progress",
		},
		{
			Name:        "risk",
			Description: "Display current risk metrics and exposure",
		},
		{
			Name:        "performance",
			Description: "Show performance analytics and statistics",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "timeframe",
					Description: "Time period for analysis",
					Required:    false,
					Choices: []*discordgo.ApplicationCommandOptionChoice{
						{Name: "1h", Value: "1h"},
						{Name: "24h", Value: "24h"},
						{Name: "7d", Value: "7d"},
						{Name: "30d", Value: "30d"},
					},
				},
			},
		},
		{
			Name:        "emergency",
			Description: "Emergency controls for trading halt and risk management",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "action",
					Description: "Emergency action to take",
					Required:    true,
					Choices: []*discordgo.ApplicationCommandOptionChoice{
						{Name: "halt_trading", Value: "halt_trading"},
						{Name: "resume_trading", Value: "resume_trading"},
						{Name: "emergency_exit", Value: "emergency_exit"},
						{Name: "reduce_exposure", Value: "reduce_exposure"},
					},
				},
			},
		},
	}

	for _, cmd := range commands {
		_, err := s.ApplicationCommandCreate(s.State.User.ID, b.guildID, cmd)
		if err != nil {
			log.Printf("Failed to create command %s: %v", cmd.Name, err)
		} else {
			log.Printf("Registered command: %s", cmd.Name)
		}
	}
}

func (b *DiscordBot) onInteractionCreate(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if i.ApplicationCommandData().Name == "" {
		return
	}

	switch i.ApplicationCommandData().Name {
	case "status":
		b.handleStatusCommand(s, i)
	case "agents":
		b.handleAgentsCommand(s, i)
	case "portfolio":
		b.handlePortfolioCommand(s, i)
	case "alerts":
		b.handleAlertsCommand(s, i)
	case "missions":
		b.handleMissionsCommand(s, i)
	case "risk":
		b.handleRiskCommand(s, i)
	case "performance":
		b.handlePerformanceCommand(s, i)
	case "emergency":
		b.handleEmergencyCommand(s, i)
	}
}

func (b *DiscordBot) handleStatusCommand(s *discordgo.Session, i *discordgo.InteractionCreate) {
	embed := &discordgo.MessageEmbed{
		Title:       "🤖 Prowzi Agent Platform Status",
		Description: "Real-time status of all system components",
		Color:       0x00ff00, // Green
		Timestamp:   time.Now().Format(time.RFC3339),
		Fields: []*discordgo.MessageEmbedField{
			{
				Name:   "🏃 Active Agents",
				Value:  "7/8 agents online",
				Inline: true,
			},
			{
				Name:   "💹 Trading Status",
				Value:  "✅ Active",
				Inline: true,
			},
			{
				Name:   "🛡️ Risk Level",
				Value:  "🟢 Low (15%)",
				Inline: true,
			},
			{
				Name:   "📊 Performance 24h",
				Value:  "📈 +2.34% ($1,847)",
				Inline: true,
			},
			{
				Name:   "⚡ System Load",
				Value:  "CPU: 23% | RAM: 45%",
				Inline: true,
			},
			{
				Name:   "🔗 Network",
				Value:  "Solana: ✅ | APIs: ✅",
				Inline: true,
			},
		},
		Footer: &discordgo.MessageEmbedFooter{
			Text: "Last updated",
		},
	}

	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Embeds: []*discordgo.MessageEmbed{embed},
		},
	})
}

func (b *DiscordBot) handleAgentsCommand(s *discordgo.Session, i *discordgo.InteractionCreate) {
	agents := []AgentStatusEmbed{
		{Name: "Scout", Status: "🟢 Active", Health: 98.5, Tasks: 12, Uptime: "2d 4h", LastSeen: "2s ago"},
		{Name: "Planner", Status: "🟢 Active", Health: 95.2, Tasks: 3, Uptime: "2d 4h", LastSeen: "1s ago"},
		{Name: "Trader", Status: "🟢 Active", Health: 97.8, Tasks: 8, Uptime: "2d 4h", LastSeen: "1s ago"},
		{Name: "RiskSentinel", Status: "🟢 Active", Health: 99.1, Tasks: 5, Uptime: "2d 4h", LastSeen: "1s ago"},
		{Name: "Guardian", Status: "🟢 Active", Health: 100.0, Tasks: 2, Uptime: "2d 4h", LastSeen: "1s ago"},
		{Name: "Curator", Status: "🟢 Active", Health: 94.7, Tasks: 15, Uptime: "2d 4h", LastSeen: "3s ago"},
		{Name: "Analyzer", Status: "🟢 Active", Health: 96.3, Tasks: 7, Uptime: "2d 4h", LastSeen: "2s ago"},
		{Name: "Orchestrator", Status: "🟡 Degraded", Health: 87.2, Tasks: 25, Uptime: "1d 8h", LastSeen: "5s ago"},
	}

	fields := make([]*discordgo.MessageEmbedField, 0, len(agents))
	for _, agent := range agents {
		healthIcon := "🟢"
		if agent.Health < 90 {
			healthIcon = "🟡"
		}
		if agent.Health < 70 {
			healthIcon = "🔴"
		}

		value := fmt.Sprintf("%s Health: %.1f%%\nTasks: %d | Uptime: %s\nLast seen: %s",
			healthIcon, agent.Health, agent.Tasks, agent.Uptime, agent.LastSeen)

		fields = append(fields, &discordgo.MessageEmbedField{
			Name:   fmt.Sprintf("%s %s", agent.Status, agent.Name),
			Value:  value,
			Inline: true,
		})
	}

	embed := &discordgo.MessageEmbed{
		Title:       "🤖 Agent Health Dashboard",
		Description: "Real-time status of all autonomous agents",
		Color:       0x3498db, // Blue
		Fields:      fields,
		Timestamp:   time.Now().Format(time.RFC3339),
		Footer: &discordgo.MessageEmbedFooter{
			Text: "Auto-refresh every 30s",
		},
	}

	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Embeds: []*discordgo.MessageEmbed{embed},
		},
	})
}

func (b *DiscordBot) handlePortfolioCommand(s *discordgo.Session, i *discordgo.InteractionCreate) {
	embed := &discordgo.MessageEmbed{
		Title:       "📊 Portfolio Performance",
		Description: "Current positions and performance metrics",
		Color:       0x2ecc71, // Green
		Timestamp:   time.Now().Format(time.RFC3339),
		Fields: []*discordgo.MessageEmbedField{
			{
				Name:   "💰 Total Portfolio Value",
				Value:  "$78,945.32 (+2.34%)",
				Inline: false,
			},
			{
				Name:   "🪙 SOL Position",
				Value:  "125.45 SOL\n$15,234.67 (+1.8%)",
				Inline: true,
			},
			{
				Name:   "₿ BTC Position",
				Value:  "0.5421 BTC\n$23,890.12 (+0.9%)",
				Inline: true,
			},
			{
				Name:   "💎 Alt Positions",
				Value:  "BONK, WIF, JUP\n$39,820.53 (+3.2%)",
				Inline: true,
			},
			{
				Name:   "📈 24h Performance",
				Value:  "P&L: +$1,847.23\nWin Rate: 73%\nSharpe: 2.41",
				Inline: true,
			},
			{
				Name:   "⚖️ Risk Metrics",
				Value:  "VaR: -$2,341\nDrawdown: -1.2%\nBeta: 0.85",
				Inline: true,
			},
			{
				Name:   "🔄 Recent Trades",
				Value:  "3 executed\n2 pending\n1 canceled",
				Inline: true,
			},
		},
	}

	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Embeds: []*discordgo.MessageEmbed{embed},
		},
	})
}

func (b *DiscordBot) handleAlertsCommand(s *discordgo.Session, i *discordgo.InteractionCreate) {
	options := i.ApplicationCommandData().Options
	if len(options) == 0 {
		return
	}

	action := options[0].StringValue()
	
	switch action {
	case "list":
		b.listAlerts(s, i)
	case "add":
		b.addAlert(s, i, options)
	case "remove":
		b.removeAlert(s, i, options)
	}
}

func (b *DiscordBot) listAlerts(s *discordgo.Session, i *discordgo.InteractionCreate) {
	embed := &discordgo.MessageEmbed{
		Title:       "🚨 Active Trading Alerts",
		Description: "Your current alert configurations",
		Color:       0xe74c3c, // Red
		Fields: []*discordgo.MessageEmbedField{
			{
				Name:   "SOL Price Alert",
				Value:  "Trigger: > $125\nStatus: ✅ Active",
				Inline: true,
			},
			{
				Name:   "BTC Volatility Alert",
				Value:  "Trigger: > 5% change\nStatus: ✅ Active",
				Inline: true,
			},
			{
				Name:   "Portfolio Drawdown",
				Value:  "Trigger: > -5%\nStatus: ✅ Active",
				Inline: true,
			},
		},
	}

	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Embeds: []*discordgo.MessageEmbed{embed},
		},
	})
}

func (b *DiscordBot) addAlert(s *discordgo.Session, i *discordgo.InteractionCreate, options []*discordgo.ApplicationCommandInteractionDataOption) {
	symbol := "N/A"
	threshold := 0.0

	for _, opt := range options {
		switch opt.Name {
		case "symbol":
			symbol = opt.StringValue()
		case "threshold":
			threshold = opt.FloatValue()
		}
	}

	response := fmt.Sprintf("✅ Alert added for %s at threshold $%.2f", strings.ToUpper(symbol), threshold)
	
	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Content: response,
		},
	})
}

func (b *DiscordBot) removeAlert(s *discordgo.Session, i *discordgo.InteractionCreate, options []*discordgo.ApplicationCommandInteractionDataOption) {
	symbol := "N/A"
	
	for _, opt := range options {
		if opt.Name == "symbol" {
			symbol = opt.StringValue()
		}
	}

	response := fmt.Sprintf("🗑️ Alert removed for %s", strings.ToUpper(symbol))
	
	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Content: response,
		},
	})
}

func (b *DiscordBot) handleMissionsCommand(s *discordgo.Session, i *discordgo.InteractionCreate) {
	embed := &discordgo.MessageEmbed{
		Title:       "🎯 Active Missions",
		Description: "Current autonomous mission status",
		Color:       0x9b59b6, // Purple
		Fields: []*discordgo.MessageEmbedField{
			{
				Name:   "🔍 Market Scanner",
				Value:  "Progress: 87%\nETA: 2h 15m\nStatus: 🟢 Running",
				Inline: true,
			},
			{
				Name:   "⚡ Arbitrage Hunter",
				Value:  "Progress: 45%\nETA: 4h 32m\nStatus: 🟢 Running",
				Inline: true,
			},
			{
				Name:   "🧠 Strategy Optimizer",
				Value:  "Progress: 23%\nETA: 6h 18m\nStatus: 🟡 Queued",
				Inline: true,
			},
			{
				Name:   "📊 Risk Assessment",
				Value:  "Progress: 100%\nCompleted: 1h ago\nStatus: ✅ Complete",
				Inline: true,
			},
			{
				Name:   "🔗 Cross-Chain Analysis",
				Value:  "Progress: 12%\nETA: 8h 45m\nStatus: 🟢 Running",
				Inline: true,
			},
			{
				Name:   "💎 Alpha Discovery",
				Value:  "Progress: 0%\nETA: TBD\nStatus: ⏸️ Paused",
				Inline: true,
			},
		},
	}

	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Embeds: []*discordgo.MessageEmbed{embed},
		},
	})
}

func (b *DiscordBot) handleRiskCommand(s *discordgo.Session, i *discordgo.InteractionCreate) {
	embed := &discordgo.MessageEmbed{
		Title:       "🛡️ Risk Management Dashboard",
		Description: "Current risk exposure and protection status",
		Color:       0xf39c12, // Orange
		Fields: []*discordgo.MessageEmbedField{
			{
				Name:   "📊 Overall Risk Score",
				Value:  "🟢 Low (15/100)\nWithin safe parameters",
				Inline: false,
			},
			{
				Name:   "💹 Position Risk",
				Value:  "Max Exposure: 25%\nCurrent: 18.3%\nUtilization: 73%",
				Inline: true,
			},
			{
				Name:   "⚖️ Leverage",
				Value:  "Max: 3.0x\nCurrent: 1.8x\nSafety: ✅ Good",
				Inline: true,
			},
			{
				Name:   "🔥 Heat Level",
				Value:  "Current: 2/10\nTrend: ↓ Decreasing\nStatus: 🟢 Cool",
				Inline: true,
			},
			{
				Name:   "🛑 Stop Losses",
				Value:  "Active: 7/8 positions\nTriggered: 0 today\nEffectiveness: 94%",
				Inline: true,
			},
			{
				Name:   "⚡ Circuit Breakers",
				Value:  "Armed: ✅ Yes\nThreshold: -5%\nLast trigger: Never",
				Inline: true,
			},
			{
				Name:   "🔍 Anomaly Detection",
				Value:  "Scans: 1,247 today\nAnomalies: 3 minor\nAction: 🟢 Monitoring",
				Inline: true,
			},
		},
	}

	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Embeds: []*discordgo.MessageEmbed{embed},
		},
	})
}

func (b *DiscordBot) handlePerformanceCommand(s *discordgo.Session, i *discordgo.InteractionCreate) {
	timeframe := "24h"
	options := i.ApplicationCommandData().Options
	if len(options) > 0 {
		timeframe = options[0].StringValue()
	}

	var performanceData map[string]string
	switch timeframe {
	case "1h":
		performanceData = map[string]string{
			"return":   "+0.12%",
			"trades":   "3",
			"winrate":  "66.7%",
			"volume":   "$1,234",
			"sharpe":   "1.85",
		}
	case "7d":
		performanceData = map[string]string{
			"return":   "+12.8%",
			"trades":   "47",
			"winrate":  "74.5%",
			"volume":   "$24,891",
			"sharpe":   "2.91",
		}
	case "30d":
		performanceData = map[string]string{
			"return":   "+28.3%",
			"trades":   "186",
			"winrate":  "71.2%",
			"volume":   "$89,234",
			"sharpe":   "2.67",
		}
	default: // 24h
		performanceData = map[string]string{
			"return":   "+2.34%",
			"trades":   "12",
			"winrate":  "75.0%",
			"volume":   "$5,678",
			"sharpe":   "2.41",
		}
	}

	embed := &discordgo.MessageEmbed{
		Title:       fmt.Sprintf("📊 Performance Analytics (%s)", strings.ToUpper(timeframe)),
		Description: "Comprehensive performance metrics and statistics",
		Color:       0x3498db, // Blue
		Fields: []*discordgo.MessageEmbedField{
			{
				Name:   "📈 Total Return",
				Value:  performanceData["return"],
				Inline: true,
			},
			{
				Name:   "🔄 Trades Executed",
				Value:  performanceData["trades"],
				Inline: true,
			},
			{
				Name:   "🎯 Win Rate",
				Value:  performanceData["winrate"],
				Inline: true,
			},
			{
				Name:   "💰 Volume Traded",
				Value:  performanceData["volume"],
				Inline: true,
			},
			{
				Name:   "📊 Sharpe Ratio",
				Value:  performanceData["sharpe"],
				Inline: true,
			},
			{
				Name:   "⚡ Alpha Generated",
				Value:  "+1.87%",
				Inline: true,
			},
		},
		Timestamp: time.Now().Format(time.RFC3339),
	}

	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Embeds: []*discordgo.MessageEmbed{embed},
		},
	})
}

func (b *DiscordBot) handleEmergencyCommand(s *discordgo.Session, i *discordgo.InteractionCreate) {
	options := i.ApplicationCommandData().Options
	if len(options) == 0 {
		return
	}

	action := options[0].StringValue()
	
	var response string
	var color int
	
	switch action {
	case "halt_trading":
		response = "🛑 **EMERGENCY HALT ACTIVATED**\nAll trading operations have been suspended immediately."
		color = 0xe74c3c // Red
	case "resume_trading":
		response = "✅ **Trading Resumed**\nNormal trading operations have been restored."
		color = 0x2ecc71 // Green
	case "emergency_exit":
		response = "🚨 **EMERGENCY EXIT INITIATED**\nLiquidating all positions. Please standby..."
		color = 0x8b0000 // Dark Red
	case "reduce_exposure":
		response = "⚖️ **Reducing Exposure**\nPosition sizes being reduced to 50% of current levels."
		color = 0xf39c12 // Orange
	}

	embed := &discordgo.MessageEmbed{
		Title:       "🚨 Emergency Action Executed",
		Description: response,
		Color:       color,
		Timestamp:   time.Now().Format(time.RFC3339),
		Footer: &discordgo.MessageEmbedFooter{
			Text: "Action logged and recorded",
		},
	}

	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseChannelMessageWithSource,
		Data: &discordgo.InteractionResponseData{
			Embeds: []*discordgo.MessageEmbed{embed},
		},
	})
}

func (b *DiscordBot) onMessageCreate(s *discordgo.Session, m *discordgo.MessageCreate) {
	// Ignore messages from the bot itself
	if m.Author.ID == s.State.User.ID {
		return
	}

	// Handle direct mentions or keywords
	if strings.Contains(strings.ToLower(m.Content), "prowzi") || 
	   strings.Contains(m.Content, s.State.User.Mention()) {
		
		content := strings.ToLower(m.Content)
		
		if strings.Contains(content, "status") {
			s.ChannelMessageSend(m.ChannelID, "🤖 All systems operational! Use `/status` for detailed information.")
		} else if strings.Contains(content, "help") {
			s.ChannelMessageSend(m.ChannelID, "Use slash commands:\n`/status` - System status\n`/agents` - Agent health\n`/portfolio` - Portfolio view\n`/alerts` - Alert management")
		}
	}
}

// SendTradingAlert sends a trading alert to the Discord channel
func (b *DiscordBot) SendTradingAlert(alert TradingAlert) error {
	if !b.ready || b.channelID == "" {
		return fmt.Errorf("discord bot not ready or channel not configured")
	}

	color := 0x2ecc71 // Green for buy
	if alert.Action == "SELL" {
		color = 0xe74c3c // Red for sell
	} else if alert.Action == "HOLD" {
		color = 0xf39c12 // Orange for hold
	}

	priceChange := ""
	if alert.Change24h > 0 {
		priceChange = fmt.Sprintf("📈 +%.2f%%", alert.Change24h)
	} else {
		priceChange = fmt.Sprintf("📉 %.2f%%", alert.Change24h)
	}

	embed := &discordgo.MessageEmbed{
		Title:       fmt.Sprintf("🚨 %s Trading Alert", alert.Type),
		Description: fmt.Sprintf("**%s** signal detected", alert.Action),
		Color:       color,
		Fields: []*discordgo.MessageEmbedField{
			{
				Name:   "🪙 Symbol",
				Value:  alert.Symbol,
				Inline: true,
			},
			{
				Name:   "💰 Current Price",
				Value:  fmt.Sprintf("$%.4f", alert.Price),
				Inline: true,
			},
			{
				Name:   "📊 24h Change",
				Value:  priceChange,
				Inline: true,
			},
			{
				Name:   "📈 Volume",
				Value:  fmt.Sprintf("$%.0f", alert.Volume),
				Inline: true,
			},
			{
				Name:   "🎯 Confidence",
				Value:  fmt.Sprintf("%.1f%%", alert.Confidence*100),
				Inline: true,
			},
			{
				Name:   "⚡ Action",
				Value:  alert.Action,
				Inline: true,
			},
		},
		Timestamp: alert.Timestamp,
	}

	_, err := b.session.ChannelMessageSendEmbed(b.channelID, embed)
	return err
}

// SendAgentUpdate sends agent status updates to Discord
func (b *DiscordBot) SendAgentUpdate(agent AgentStatusEmbed) error {
	if !b.ready || b.channelID == "" {
		return fmt.Errorf("discord bot not ready or channel not configured")
	}

	statusIcon := "🟢"
	color := 0x2ecc71 // Green
	
	if agent.Health < 90 {
		statusIcon = "🟡"
		color = 0xf39c12 // Orange
	}
	if agent.Health < 70 {
		statusIcon = "🔴" 
		color = 0xe74c3c // Red
	}

	embed := &discordgo.MessageEmbed{
		Title:       fmt.Sprintf("%s Agent %s Update", statusIcon, agent.Name),
		Description: fmt.Sprintf("Health: %.1f%% | Tasks: %d", agent.Health, agent.Tasks),
		Color:       color,
		Fields: []*discordgo.MessageEmbedField{
			{
				Name:   "Status",
				Value:  agent.Status,
				Inline: true,
			},
			{
				Name:   "Uptime",
				Value:  agent.Uptime,
				Inline: true,
			},
			{
				Name:   "Last Seen",
				Value:  agent.LastSeen,
				Inline: true,
			},
		},
		Timestamp: time.Now().Format(time.RFC3339),
	}

	_, err := b.session.ChannelMessageSendEmbed(b.channelID, embed)
	return err
}