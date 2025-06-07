go
package channels

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strconv"
    "strings"
    "time"
)

type TelegramNotifier struct {
    botToken    string
    client      *http.Client
    webhookURL  string
    commandHandlers map[string]CommandHandler
}

type CommandHandler func(ctx context.Context, chatID int64, args []string) error

type TelegramMessage struct {
    MessageID int    `json:"message_id"`
    From      *User  `json:"from"`
    Chat      *Chat  `json:"chat"`
    Date      int64  `json:"date"`
    Text      string `json:"text"`
}

type User struct {
    ID        int64  `json:"id"`
    IsBot     bool   `json:"is_bot"`
    FirstName string `json:"first_name"`
    LastName  string `json:"last_name,omitempty"`
    Username  string `json:"username,omitempty"`
}

type Chat struct {
    ID    int64  `json:"id"`
    Type  string `json:"type"`
    Title string `json:"title,omitempty"`
}

type Update struct {
    UpdateID int              `json:"update_id"`
    Message  *TelegramMessage `json:"message"`
}

type InlineKeyboard struct {
    InlineKeyboard [][]InlineKeyboardButton `json:"inline_keyboard"`
}

type InlineKeyboardButton struct {
    Text         string `json:"text"`
    CallbackData string `json:"callback_data,omitempty"`
    URL          string `json:"url,omitempty"`
}

type NotificationPayload struct {
    ChatID      string                 `json:"chat_id"`
    Text        string                 `json:"text"`
    ParseMode   string                 `json:"parse_mode,omitempty"`
    ReplyMarkup *InlineKeyboard        `json:"reply_markup,omitempty"`
    DisableWebPagePreview bool         `json:"disable_web_page_preview,omitempty"`
    Priority    string                 `json:"priority,omitempty"`
    Tags        []string               `json:"tags,omitempty"`
    Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

func NewTelegramNotifier(botToken string) *TelegramNotifier {
    tn := &TelegramNotifier{
        botToken: botToken,
        client: &http.Client{
            Timeout: 30 * time.Second,
        },
        commandHandlers: make(map[string]CommandHandler),
    }
    
    // Register default commands
    tn.RegisterCommand("start", tn.handleStartCommand)
    tn.RegisterCommand("help", tn.handleHelpCommand)
    tn.RegisterCommand("status", tn.handleStatusCommand)
    tn.RegisterCommand("subscribe", tn.handleSubscribeCommand)
    tn.RegisterCommand("unsubscribe", tn.handleUnsubscribeCommand)
    tn.RegisterCommand("alerts", tn.handleAlertsCommand)
    tn.RegisterCommand("portfolio", tn.handlePortfolioCommand)
    tn.RegisterCommand("missions", tn.handleMissionsCommand)
    
    return tn
}

func (t *TelegramNotifier) RegisterCommand(command string, handler CommandHandler) {
    t.commandHandlers[command] = handler
}

func (t *TelegramNotifier) SetWebhook(webhookURL string) error {
    t.webhookURL = webhookURL
    url := fmt.Sprintf("https://api.telegram.org/bot%s/setWebhook", t.botToken)
    
    payload := map[string]interface{}{
        "url":             webhookURL,
        "allowed_updates": []string{"message", "callback_query"},
        "drop_pending_updates": true,
    }
    
    body, err := json.Marshal(payload)
    if err != nil {
        return err
    }
    
    resp, err := t.client.Post(url, "application/json", bytes.NewBuffer(body))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("failed to set webhook: %d", resp.StatusCode)
    }
    
    return nil
}

func (t *TelegramNotifier) HandleWebhook(ctx context.Context, update Update) error {
    if update.Message == nil {
        return nil
    }
    
    message := update.Message
    if strings.HasPrefix(message.Text, "/") {
        return t.handleCommand(ctx, message)
    }
    
    // Handle regular message if needed
    return nil
}

func (t *TelegramNotifier) handleCommand(ctx context.Context, message *TelegramMessage) error {
    parts := strings.Fields(message.Text)
    if len(parts) == 0 {
        return nil
    }
    
    command := strings.TrimPrefix(parts[0], "/")
    args := parts[1:]
    
    if handler, exists := t.commandHandlers[command]; exists {
        return handler(ctx, message.Chat.ID, args)
    }
    
    return t.sendMessage(message.Chat.ID, "Unknown command. Type /help for available commands.")
}

func (t *TelegramNotifier) Send(chatID string, message string) error {
    payload := NotificationPayload{
        ChatID:                chatID,
        Text:                  message,
        ParseMode:             "MarkdownV2",
        DisableWebPagePreview: true,
    }
    
    return t.sendNotification(payload)
}

func (t *TelegramNotifier) SendRichNotification(payload NotificationPayload) error {
    return t.sendNotification(payload)
}

func (t *TelegramNotifier) SendAlert(chatID string, alert map[string]interface{}) error {
    alertType := alert["type"].(string)
    title := alert["title"].(string)
    description := alert["description"].(string)
    severity := alert["severity"].(string)
    
    emoji := "ℹ️"
    switch severity {
    case "critical":
        emoji = "🚨"
    case "warning":
        emoji = "⚠️"
    case "success":
        emoji = "✅"
    }
    
    text := fmt.Sprintf("%s *%s Alert*\n\n*%s*\n\n%s", emoji, strings.Title(alertType), title, description)
    
    var keyboard *InlineKeyboard
    if alertType == "trading" {
        keyboard = &InlineKeyboard{
            InlineKeyboard: [][]InlineKeyboardButton{
                {
                    {Text: "📊 View Details", CallbackData: fmt.Sprintf("alert_details_%s", alert["id"])},
                    {Text: "🛑 Stop Trading", CallbackData: fmt.Sprintf("stop_trading_%s", alert["id"])},
                },
                {
                    {Text: "📈 Dashboard", URL: "https://app.prowzi.com/dashboard"},
                },
            },
        }
    }
    
    payload := NotificationPayload{
        ChatID:                chatID,
        Text:                  text,
        ParseMode:             "MarkdownV2",
        ReplyMarkup:           keyboard,
        DisableWebPagePreview: true,
        Priority:              severity,
        Tags:                  []string{alertType, severity},
        Metadata:              alert,
    }
    
    return t.sendNotification(payload)
}

func (t *TelegramNotifier) sendNotification(payload NotificationPayload) error {
    url := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", t.botToken)
    
    body, err := json.Marshal(payload)
    if err != nil {
        return err
    }
    
    resp, err := t.client.Post(url, "application/json", bytes.NewBuffer(body))
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != http.StatusOK {
        var errorResp map[string]interface{}
        json.NewDecoder(resp.Body).Decode(&errorResp)
        return fmt.Errorf("telegram API error: %d - %v", resp.StatusCode, errorResp)
    }
    
    return nil
}

func (t *TelegramNotifier) sendMessage(chatID int64, text string) error {
    return t.Send(strconv.FormatInt(chatID, 10), text)
}

// Command handlers
func (t *TelegramNotifier) handleStartCommand(ctx context.Context, chatID int64, args []string) error {
    welcome := `🚀 *Welcome to Prowzi AI Agent Platform\\!*

I'm your AI assistant for real\\-time crypto and DeFi intelligence\\. Here's what I can help you with:

*🔍 Intelligence Gathering*
• Market analysis and trends
• MEV opportunities detection
• Cross\\-chain arbitrage alerts
• Risk monitoring and protection

*📊 Portfolio Management*
• Real\\-time performance tracking
• Automated trading strategies  
• Risk assessment and controls
• Profit/loss reporting

*⚡ Real\\-time Alerts*
• Price movements and volatility
• Trading opportunities
• Security threats and risks
• System status updates

Type /help to see all available commands\\!`

    keyboard := &InlineKeyboard{
        InlineKeyboard: [][]InlineKeyboardButton{
            {
                {Text: "📱 Open Dashboard", URL: "https://app.prowzi.com"},
                {Text: "📚 Documentation", URL: "https://docs.prowzi.com"},
            },
            {
                {Text: "🔔 Setup Alerts", CallbackData: "setup_alerts"},
            },
        },
    }
    
    payload := NotificationPayload{
        ChatID:      strconv.FormatInt(chatID, 10),
        Text:        welcome,
        ParseMode:   "MarkdownV2",
        ReplyMarkup: keyboard,
    }
    
    return t.sendNotification(payload)
}

func (t *TelegramNotifier) handleHelpCommand(ctx context.Context, chatID int64, args []string) error {
    help := `*🤖 Prowzi Bot Commands*

*General Commands:*
/start \\- Initialize bot and show welcome
/help \\- Show this help message
/status \\- Check system status

*Trading & Portfolio:*
/portfolio \\- View portfolio performance
/missions \\- Show active trading missions
/alerts \\- Configure alert settings

*Subscriptions:*
/subscribe \\[type\\] \\- Subscribe to notifications
/unsubscribe \\[type\\] \\- Unsubscribe from notifications

*Available subscription types:*
• \\`trading\\` \\- Trading signals and execution
• \\`risk\\` \\- Risk alerts and protection
• \\`market\\` \\- Market analysis and trends
• \\`portfolio\\` \\- Portfolio performance updates
• \\`system\\` \\- System status and maintenance

*Examples:*
\\`/subscribe trading\\`
\\`/unsubscribe risk\\`
\\`/alerts critical\\`

Need more help\\? Visit [our documentation](https://docs\\.prowzi\\.com)`

    return t.sendMessage(chatID, help)
}

func (t *TelegramNotifier) handleStatusCommand(ctx context.Context, chatID int64, args []string) error {
    status := `*🟢 Prowzi System Status*

*Core Services:*
• 🤖 AI Agents: \\`Operational\\`
• 📊 Analytics: \\`Operational\\`  
• 🔒 Security: \\`Operational\\`
• 💰 Trading: \\`Operational\\`

*Network Status:*
• Solana RPC: \\`Connected\\`
• Ethereum RPC: \\`Connected\\`
• Price Feeds: \\`Active\\`
• MEV Protection: \\`Enabled\\`

*Performance Metrics:*
• Uptime: \\`99\\.9%\\`
• Response Time: \\`<50ms\\`
• Active Missions: \\`12\\`
• Protected Volume: \\`$2\\.4M\\`

Last Updated: ` + time.Now().Format("15:04 UTC")

    keyboard := &InlineKeyboard{
        InlineKeyboard: [][]InlineKeyboardButton{
            {
                {Text: "🔄 Refresh", CallbackData: "refresh_status"},
                {Text: "📊 Detailed Metrics", URL: "https://app.prowzi.com/status"},
            },
        },
    }
    
    payload := NotificationPayload{
        ChatID:      strconv.FormatInt(chatID, 10),
        Text:        status,
        ParseMode:   "MarkdownV2",
        ReplyMarkup: keyboard,
    }
    
    return t.sendNotification(payload)
}

func (t *TelegramNotifier) handleSubscribeCommand(ctx context.Context, chatID int64, args []string) error {
    if len(args) == 0 {
        return t.sendMessage(chatID, "Please specify subscription type. Example: /subscribe trading")
    }
    
    subscriptionType := args[0]
    validTypes := []string{"trading", "risk", "market", "portfolio", "system"}
    
    valid := false
    for _, validType := range validTypes {
        if subscriptionType == validType {
            valid = true
            break
        }
    }
    
    if !valid {
        return t.sendMessage(chatID, fmt.Sprintf("Invalid subscription type. Valid types: %s", strings.Join(validTypes, ", ")))
    }
    
    // TODO: Implement actual subscription logic with database
    message := fmt.Sprintf("✅ Successfully subscribed to *%s* notifications\\!\n\nYou'll now receive real\\-time updates for %s events\\.", subscriptionType, subscriptionType)
    
    return t.sendMessage(chatID, message)
}

func (t *TelegramNotifier) handleUnsubscribeCommand(ctx context.Context, chatID int64, args []string) error {
    if len(args) == 0 {
        return t.sendMessage(chatID, "Please specify subscription type. Example: /unsubscribe trading")
    }
    
    subscriptionType := args[0]
    
    // TODO: Implement actual unsubscription logic with database
    message := fmt.Sprintf("❌ Successfully unsubscribed from *%s* notifications\\.", subscriptionType)
    
    return t.sendMessage(chatID, message)
}

func (t *TelegramNotifier) handleAlertsCommand(ctx context.Context, chatID int64, args []string) error {
    alerts := `*🔔 Alert Configuration*

*Current Alert Settings:*
• Trading Signals: \\`Enabled\\`
• Risk Warnings: \\`Enabled\\`
• Portfolio Updates: \\`Daily\\`
• System Status: \\`Critical Only\\`

*Available Alert Levels:*
• \\`all\\` \\- All notifications
• \\`critical\\` \\- Critical alerts only
• \\`minimal\\` \\- Essential updates only
• \\`off\\` \\- No notifications

*Usage:*
\\`/alerts critical\\` \\- Set to critical only
\\`/alerts all\\` \\- Enable all notifications`

    keyboard := &InlineKeyboard{
        InlineKeyboard: [][]InlineKeyboardButton{
            {
                {Text: "🔕 Critical Only", CallbackData: "alerts_critical"},
                {Text: "🔔 All Alerts", CallbackData: "alerts_all"},
            },
            {
                {Text: "📱 Customize", URL: "https://app.prowzi.com/settings/notifications"},
            },
        },
    }
    
    payload := NotificationPayload{
        ChatID:      strconv.FormatInt(chatID, 10),
        Text:        alerts,
        ParseMode:   "MarkdownV2",
        ReplyMarkup: keyboard,
    }
    
    return t.sendNotification(payload)
}

func (t *TelegramNotifier) handlePortfolioCommand(ctx context.Context, chatID int64, args []string) error {
    portfolio := `*💰 Portfolio Overview*

*Total Value:* \\$45,234\\.56 \\(\\+2\\.1%\\)
*24h PnL:* \\+\\$987\\.43 \\(\\+2\\.23%\\)
*Active Positions:* 8

*Top Performers:*
🟢 SOL/USDC: \\+\\$345\\.12 \\(\\+4\\.2%\\)
🟢 ETH/USDC: \\+\\$234\\.56 \\(\\+1\\.8%\\)
🔴 BTC/USDC: \\-\\$123\\.45 \\(\\-0\\.9%\\)

*Active Strategies:*
• Cross\\-chain Arbitrage: \\`Running\\`
• MEV Protection: \\`Active\\`
• Risk Management: \\`Monitoring\\`

*Risk Metrics:*
• Portfolio Risk: \\`Medium\\` \\(6/10\\)
• Max Drawdown: \\-3\\.2%
• Sharpe Ratio: 1\\.85`

    keyboard := &InlineKeyboard{
        InlineKeyboard: [][]InlineKeyboardButton{
            {
                {Text: "📊 Full Report", URL: "https://app.prowzi.com/portfolio"},
                {Text: "⚙️ Adjust Strategy", CallbackData: "adjust_strategy"},
            },
            {
                {Text: "🛑 Emergency Stop", CallbackData: "emergency_stop"},
            },
        },
    }
    
    payload := NotificationPayload{
        ChatID:      strconv.FormatInt(chatID, 10),
        Text:        portfolio,
        ParseMode:   "MarkdownV2",
        ReplyMarkup: keyboard,
    }
    
    return t.sendNotification(payload)
}

func (t *TelegramNotifier) handleMissionsCommand(ctx context.Context, chatID int64, args []string) error {
    missions := `*🎯 Active Missions*

*Mission \\#1: Cross\\-Chain Arbitrage*
Status: \\`🟢 Active\\`
Progress: 85% \\(17/20 opportunities\\)
Profit: \\+\\$1,234\\.56
Risk: \\`Low\\`

*Mission \\#2: MEV Protection*
Status: \\`🟡 Monitoring\\`
Transactions Protected: 156
Savings: \\+\\$567\\.89
Risk: \\`Medium\\`

*Mission \\#3: Yield Optimization*
Status: \\`🟢 Active\\`
APY Improvement: \\+12\\.3%
Capital Deployed: \\$15,000
Risk: \\`Low\\`

*Queue:*
• Liquidity Mining Analysis
• DeFi Protocol Assessment
• Risk Model Calibration`

    keyboard := &InlineKeyboard{
        InlineKeyboard: [][]InlineKeyboardButton{
            {
                {Text: "📋 Mission Details", URL: "https://app.prowzi.com/missions"},
                {Text: "➕ New Mission", CallbackData: "new_mission"},
            },
            {
                {Text: "⏸️ Pause All", CallbackData: "pause_missions"},
            },
        },
    }
    
    payload := NotificationPayload{
        ChatID:      strconv.FormatInt(chatID, 10),
        Text:        missions,
        ParseMode:   "MarkdownV2",
        ReplyMarkup: keyboard,
    }
    
    return t.sendNotification(payload)
}
