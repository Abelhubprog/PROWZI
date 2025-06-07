use crate::{Actor, Budget, Message};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use anyhow::{Result, anyhow};
use regex::Regex;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCommand {
    pub command: String,
    pub parameters: HashMap<String, String>,
    pub user_id: String,
    pub channel_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub confidence: f64,
    pub intent: CommandIntent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandIntent {
    Trade {
        action: TradeAction,
        token: Option<String>,
        amount: Option<f64>,
        conditions: Vec<TradeCondition>,
    },
    Monitor {
        target: MonitorTarget,
        frequency: Option<u64>,
        alerts: Vec<AlertCondition>,
    },
    Query {
        subject: QuerySubject,
        filters: HashMap<String, String>,
        format: ResponseFormat,
    },
    Control {
        action: ControlAction,
        scope: ControlScope,
        parameters: HashMap<String, String>,
    },
    Learn {
        feedback_type: FeedbackType,
        content: String,
        rating: Option<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeAction {
    Buy,
    Sell,
    SetStopLoss,
    SetTakeProfit,
    Cancel,
    ModifyPosition,
    AnalyzeOpportunity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeCondition {
    PriceAbove(f64),
    PriceBelow(f64),
    VolumeSpike(f64),
    TimeFrame(String),
    TechnicalIndicator { name: String, value: f64 },
    MarketCondition(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitorTarget {
    Token(String),
    Portfolio,
    MarketSector(String),
    TechnicalIndicator(String),
    NewsKeyword(String),
    SocialSentiment(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    PriceMove { percentage: f64, direction: String },
    VolumeThreshold(f64),
    TimeBasedAlert(chrono::DateTime<chrono::Utc>),
    CustomMetric { name: String, threshold: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuerySubject {
    Portfolio,
    TradingHistory,
    MarketAnalysis,
    Performance,
    Agents,
    Risk,
    Opportunities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseFormat {
    Brief,
    Detailed,
    Chart,
    Table,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlAction {
    Start,
    Stop,
    Pause,
    Resume,
    Emergency,
    Configure,
    Reset,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlScope {
    AllAgents,
    SpecificAgent(String),
    TradingEngine,
    RiskManagement,
    Portfolio,
    Monitoring,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    TradeOutcome,
    StrategyPerformance,
    AlertRelevance,
    UserPreference,
    ErrorReport,
    Suggestion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandPattern {
    pub regex: String,
    pub intent_type: String,
    pub parameter_extractors: HashMap<String, String>,
    pub confidence_boost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningContext {
    pub user_preferences: HashMap<String, String>,
    pub command_history: Vec<ChatCommand>,
    pub feedback_patterns: HashMap<String, f64>,
    pub success_metrics: HashMap<String, f64>,
}

pub struct ChatCommandProcessor {
    patterns: Arc<RwLock<Vec<CommandPattern>>>,
    learning_context: Arc<RwLock<HashMap<String, LearningContext>>>,
    nlp_service: Arc<dyn NLPService + Send + Sync>,
    command_history: Arc<RwLock<Vec<ChatCommand>>>,
}

#[async_trait]
pub trait NLPService {
    async fn extract_intent(&self, text: &str) -> Result<CommandIntent>;
    async fn extract_entities(&self, text: &str) -> Result<HashMap<String, String>>;
    async fn calculate_confidence(&self, text: &str, intent: &CommandIntent) -> Result<f64>;
    async fn learn_from_feedback(&self, command: &ChatCommand, feedback: &str) -> Result<()>;
}

impl ChatCommandProcessor {
    pub fn new(nlp_service: Arc<dyn NLPService + Send + Sync>) -> Self {
        let mut patterns = Vec::new();
        
        // Trading patterns
        patterns.extend(vec![
            CommandPattern {
                regex: r"(?i)buy\s+(\d+(?:\.\d+)?)\s*(\w+)\s*(?:at\s+(\d+(?:\.\d+)?)|when\s+(.+))?".to_string(),
                intent_type: "trade".to_string(),
                parameter_extractors: HashMap::from([
                    ("action".to_string(), "buy".to_string()),
                    ("amount".to_string(), "$1".to_string()),
                    ("token".to_string(), "$2".to_string()),
                    ("price".to_string(), "$3".to_string()),
                    ("condition".to_string(), "$4".to_string()),
                ]),
                confidence_boost: 0.9,
            },
            CommandPattern {
                regex: r"(?i)sell\s+(?:(\d+(?:\.\d+)?)\s*)?(\w+)\s*(?:at\s+(\d+(?:\.\d+)?)|when\s+(.+))?".to_string(),
                intent_type: "trade".to_string(),
                parameter_extractors: HashMap::from([
                    ("action".to_string(), "sell".to_string()),
                    ("amount".to_string(), "$1".to_string()),
                    ("token".to_string(), "$2".to_string()),
                    ("price".to_string(), "$3".to_string()),
                    ("condition".to_string(), "$4".to_string()),
                ]),
                confidence_boost: 0.9,
            },
            CommandPattern {
                regex: r"(?i)set\s+stop\s*loss\s+(?:for\s+)?(\w+)\s+at\s+(\d+(?:\.\d+)?)".to_string(),
                intent_type: "trade".to_string(),
                parameter_extractors: HashMap::from([
                    ("action".to_string(), "stop_loss".to_string()),
                    ("token".to_string(), "$1".to_string()),
                    ("price".to_string(), "$2".to_string()),
                ]),
                confidence_boost: 0.85,
            },
        ]);

        // Monitoring patterns
        patterns.extend(vec![
            CommandPattern {
                regex: r"(?i)monitor\s+(\w+)\s*(?:for\s+(.+))?".to_string(),
                intent_type: "monitor".to_string(),
                parameter_extractors: HashMap::from([
                    ("target".to_string(), "$1".to_string()),
                    ("conditions".to_string(), "$2".to_string()),
                ]),
                confidence_boost: 0.8,
            },
            CommandPattern {
                regex: r"(?i)watch\s+(?:for\s+)?(.+)".to_string(),
                intent_type: "monitor".to_string(),
                parameter_extractors: HashMap::from([
                    ("target".to_string(), "$1".to_string()),
                ]),
                confidence_boost: 0.75,
            },
            CommandPattern {
                regex: r"(?i)alert\s+me\s+(?:when|if)\s+(.+)".to_string(),
                intent_type: "monitor".to_string(),
                parameter_extractors: HashMap::from([
                    ("condition".to_string(), "$1".to_string()),
                ]),
                confidence_boost: 0.8,
            },
        ]);

        // Query patterns
        patterns.extend(vec![
            CommandPattern {
                regex: r"(?i)(?:show|get|what's)\s+(?:my\s+)?(?:(portfolio|positions|balance|performance|history|agents|risk)(?:\s+(.+))?)".to_string(),
                intent_type: "query".to_string(),
                parameter_extractors: HashMap::from([
                    ("subject".to_string(), "$1".to_string()),
                    ("filters".to_string(), "$2".to_string()),
                ]),
                confidence_boost: 0.85,
            },
            CommandPattern {
                regex: r"(?i)analyze\s+(.+)".to_string(),
                intent_type: "query".to_string(),
                parameter_extractors: HashMap::from([
                    ("subject".to_string(), "analysis".to_string()),
                    ("target".to_string(), "$1".to_string()),
                ]),
                confidence_boost: 0.8,
            },
        ]);

        // Control patterns
        patterns.extend(vec![
            CommandPattern {
                regex: r"(?i)(start|stop|pause|resume)\s+(.+)".to_string(),
                intent_type: "control".to_string(),
                parameter_extractors: HashMap::from([
                    ("action".to_string(), "$1".to_string()),
                    ("scope".to_string(), "$2".to_string()),
                ]),
                confidence_boost: 0.9,
            },
            CommandPattern {
                regex: r"(?i)emergency\s+(?:(stop|halt|exit)|(.+))".to_string(),
                intent_type: "control".to_string(),
                parameter_extractors: HashMap::from([
                    ("action".to_string(), "emergency".to_string()),
                    ("type".to_string(), "$1".to_string()),
                    ("scope".to_string(), "$2".to_string()),
                ]),
                confidence_boost: 0.95,
            },
        ]);

        Self {
            patterns: Arc::new(RwLock::new(patterns)),
            learning_context: Arc::new(RwLock::new(HashMap::new())),
            nlp_service,
            command_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn process_command(&self, text: &str, user_id: &str, channel_id: &str) -> Result<ChatCommand> {
        // First try pattern matching
        let pattern_result = self.match_patterns(text).await?;
        
        // Then use NLP service for intent extraction
        let nlp_intent = self.nlp_service.extract_intent(text).await?;
        let entities = self.nlp_service.extract_entities(text).await?;
        
        // Combine pattern and NLP results
        let (final_intent, confidence) = self.combine_results(pattern_result, nlp_intent, text).await?;
        
        let command = ChatCommand {
            command: text.to_string(),
            parameters: entities,
            user_id: user_id.to_string(),
            channel_id: channel_id.to_string(),
            timestamp: chrono::Utc::now(),
            confidence,
            intent: final_intent,
        };

        // Learn from this command
        self.learn_from_command(&command).await?;
        
        // Store in history
        let mut history = self.command_history.write().await;
        history.push(command.clone());
        
        // Keep only last 1000 commands
        if history.len() > 1000 {
            history.drain(0..history.len() - 1000);
        }

        Ok(command)
    }

    async fn match_patterns(&self, text: &str) -> Result<Option<(CommandIntent, f64)>> {
        let patterns = self.patterns.read().await;
        
        for pattern in patterns.iter() {
            let regex = Regex::new(&pattern.regex)?;
            if let Some(captures) = regex.captures(text) {
                let mut parameters = HashMap::new();
                
                for (param_name, capture_group) in &pattern.parameter_extractors {
                    if let Some(captured) = self.extract_capture(&captures, capture_group) {
                        parameters.insert(param_name.clone(), captured);
                    }
                }
                
                let intent = self.build_intent_from_pattern(&pattern.intent_type, parameters).await?;
                return Ok(Some((intent, pattern.confidence_boost)));
            }
        }
        
        Ok(None)
    }

    fn extract_capture(&self, captures: &regex::Captures, capture_group: &str) -> Option<String> {
        if capture_group.starts_with('$') {
            let group_num: usize = capture_group[1..].parse().ok()?;
            captures.get(group_num)?.as_str().to_string().into()
        } else {
            Some(capture_group.to_string())
        }
    }

    async fn build_intent_from_pattern(&self, intent_type: &str, parameters: HashMap<String, String>) -> Result<CommandIntent> {
        match intent_type {
            "trade" => {
                let action = match parameters.get("action").map(|s| s.as_str()) {
                    Some("buy") => TradeAction::Buy,
                    Some("sell") => TradeAction::Sell,
                    Some("stop_loss") => TradeAction::SetStopLoss,
                    Some("take_profit") => TradeAction::SetTakeProfit,
                    _ => TradeAction::AnalyzeOpportunity,
                };
                
                let amount = parameters.get("amount")
                    .and_then(|s| s.parse::<f64>().ok());
                let token = parameters.get("token").cloned();
                
                let mut conditions = Vec::new();
                if let Some(price_str) = parameters.get("price") {
                    if let Ok(price) = price_str.parse::<f64>() {
                        conditions.push(match action {
                            TradeAction::Buy => TradeCondition::PriceBelow(price),
                            TradeAction::Sell => TradeCondition::PriceAbove(price),
                            _ => TradeCondition::PriceAbove(price),
                        });
                    }
                }
                
                Ok(CommandIntent::Trade { action, token, amount, conditions })
            },
            "monitor" => {
                let target = parameters.get("target")
                    .map(|s| MonitorTarget::Token(s.clone()))
                    .unwrap_or(MonitorTarget::Portfolio);
                
                let alerts = self.parse_alert_conditions(&parameters).await?;
                
                Ok(CommandIntent::Monitor { target, frequency: None, alerts })
            },
            "query" => {
                let subject = match parameters.get("subject").map(|s| s.as_str()) {
                    Some("portfolio") => QuerySubject::Portfolio,
                    Some("positions") => QuerySubject::Portfolio,
                    Some("performance") => QuerySubject::Performance,
                    Some("history") => QuerySubject::TradingHistory,
                    Some("agents") => QuerySubject::Agents,
                    Some("risk") => QuerySubject::Risk,
                    Some("analysis") => QuerySubject::MarketAnalysis,
                    _ => QuerySubject::Portfolio,
                };
                
                let filters = self.parse_query_filters(&parameters).await?;
                
                Ok(CommandIntent::Query { subject, filters, format: ResponseFormat::Brief })
            },
            "control" => {
                let action = match parameters.get("action").map(|s| s.as_str()) {
                    Some("start") => ControlAction::Start,
                    Some("stop") => ControlAction::Stop,
                    Some("pause") => ControlAction::Pause,
                    Some("resume") => ControlAction::Resume,
                    Some("emergency") => ControlAction::Emergency,
                    _ => ControlAction::Configure,
                };
                
                let scope = match parameters.get("scope").map(|s| s.as_str()) {
                    Some("trading") => ControlScope::TradingEngine,
                    Some("agents") => ControlScope::AllAgents,
                    Some("risk") => ControlScope::RiskManagement,
                    Some("portfolio") => ControlScope::Portfolio,
                    Some(agent_name) => ControlScope::SpecificAgent(agent_name.to_string()),
                    _ => ControlScope::AllAgents,
                };
                
                Ok(CommandIntent::Control { action, scope, parameters })
            },
            _ => Err(anyhow!("Unknown intent type: {}", intent_type)),
        }
    }

    async fn parse_alert_conditions(&self, parameters: &HashMap<String, String>) -> Result<Vec<AlertCondition>> {
        let mut conditions = Vec::new();
        
        if let Some(condition_str) = parameters.get("conditions").or_else(|| parameters.get("condition")) {
            // Parse common alert patterns
            if condition_str.contains("price") && condition_str.contains("%") {
                let regex = Regex::new(r"(\d+(?:\.\d+)?)%")?;
                if let Some(captures) = regex.captures(condition_str) {
                    if let Ok(percentage) = captures[1].parse::<f64>() {
                        let direction = if condition_str.contains("up") || condition_str.contains("above") {
                            "up".to_string()
                        } else {
                            "down".to_string()
                        };
                        conditions.push(AlertCondition::PriceMove { percentage, direction });
                    }
                }
            }
            
            if condition_str.contains("volume") {
                let regex = Regex::new(r"volume[^\d]*(\d+(?:\.\d+)?)")?;
                if let Some(captures) = regex.captures(condition_str) {
                    if let Ok(threshold) = captures[1].parse::<f64>() {
                        conditions.push(AlertCondition::VolumeThreshold(threshold));
                    }
                }
            }
        }
        
        Ok(conditions)
    }

    async fn parse_query_filters(&self, parameters: &HashMap<String, String>) -> Result<HashMap<String, String>> {
        let mut filters = HashMap::new();
        
        if let Some(filter_str) = parameters.get("filters") {
            // Parse time filters
            if filter_str.contains("day") || filter_str.contains("24h") {
                filters.insert("timeframe".to_string(), "1d".to_string());
            } else if filter_str.contains("week") || filter_str.contains("7d") {
                filters.insert("timeframe".to_string(), "7d".to_string());
            } else if filter_str.contains("month") || filter_str.contains("30d") {
                filters.insert("timeframe".to_string(), "30d".to_string());
            }
            
            // Parse token filters
            let token_regex = Regex::new(r"\b([A-Z]{2,10})\b").unwrap();
            if let Some(captures) = token_regex.captures(filter_str) {
                filters.insert("token".to_string(), captures[1].to_string());
            }
        }
        
        Ok(filters)
    }

    async fn combine_results(&self, pattern_result: Option<(CommandIntent, f64)>, nlp_intent: CommandIntent, text: &str) -> Result<(CommandIntent, f64)> {
        let nlp_confidence = self.nlp_service.calculate_confidence(text, &nlp_intent).await?;
        
        match pattern_result {
            Some((pattern_intent, pattern_confidence)) => {
                // If pattern confidence is high, prefer pattern result
                if pattern_confidence > 0.8 {
                    Ok((pattern_intent, pattern_confidence))
                } else if nlp_confidence > pattern_confidence {
                    Ok((nlp_intent, nlp_confidence))
                } else {
                    Ok((pattern_intent, pattern_confidence))
                }
            },
            None => Ok((nlp_intent, nlp_confidence)),
        }
    }

    async fn learn_from_command(&self, command: &ChatCommand) -> Result<()> {
        let mut context_map = self.learning_context.write().await;
        let context = context_map.entry(command.user_id.clone())
            .or_insert_with(|| LearningContext {
                user_preferences: HashMap::new(),
                command_history: Vec::new(),
                feedback_patterns: HashMap::new(),
                success_metrics: HashMap::new(),
            });
        
        context.command_history.push(command.clone());
        
        // Keep only last 100 commands per user
        if context.command_history.len() > 100 {
            context.command_history.drain(0..context.command_history.len() - 100);
        }
        
        // Learn patterns from command history
        self.update_user_preferences(&command.user_id, command).await?;
        
        Ok(())
    }

    async fn update_user_preferences(&self, user_id: &str, command: &ChatCommand) -> Result<()> {
        let mut context_map = self.learning_context.write().await;
        if let Some(context) = context_map.get_mut(user_id) {
            // Learn preferred response format
            match &command.intent {
                CommandIntent::Query { format, .. } => {
                    let format_key = format!("preferred_format_{:?}", format);
                    let count = context.user_preferences.get(&format_key)
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(0);
                    context.user_preferences.insert(format_key, (count + 1).to_string());
                },
                CommandIntent::Trade { action, .. } => {
                    let action_key = format!("preferred_trade_action_{:?}", action);
                    let count = context.user_preferences.get(&action_key)
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(0);
                    context.user_preferences.insert(action_key, (count + 1).to_string());
                },
                _ => {},
            }
        }
        
        Ok(())
    }

    pub async fn provide_feedback(&self, command_id: &str, feedback: &str, rating: Option<f64>) -> Result<()> {
        // Find the command in history
        let history = self.command_history.read().await;
        if let Some(command) = history.iter().find(|cmd| {
            // Simple ID matching based on timestamp and user
            format!("{}_{}", cmd.timestamp.timestamp(), cmd.user_id) == command_id
        }) {
            // Learn from feedback
            self.nlp_service.learn_from_feedback(command, feedback).await?;
            
            // Update learning context
            let mut context_map = self.learning_context.write().await;
            if let Some(context) = context_map.get_mut(&command.user_id) {
                let feedback_key = format!("{:?}", command.intent);
                if let Some(rating) = rating {
                    context.feedback_patterns.insert(feedback_key, rating);
                }
            }
        }
        
        Ok(())
    }

    pub async fn get_command_suggestions(&self, partial_text: &str, user_id: &str) -> Result<Vec<String>> {
        let context_map = self.learning_context.read().await;
        let mut suggestions = Vec::new();
        
        // Get user's command history for personalized suggestions
        if let Some(context) = context_map.get(user_id) {
            for cmd in context.command_history.iter().rev().take(10) {
                if cmd.command.to_lowercase().starts_with(&partial_text.to_lowercase()) {
                    suggestions.push(cmd.command.clone());
                }
            }
        }
        
        // Add common command suggestions based on partial text
        let common_suggestions = self.get_common_suggestions(partial_text).await?;
        suggestions.extend(common_suggestions);
        
        // Remove duplicates and limit to 5 suggestions
        suggestions.sort_by(|a, b| a.len().cmp(&b.len()));
        suggestions.dedup();
        suggestions.truncate(5);
        
        Ok(suggestions)
    }

    async fn get_common_suggestions(&self, partial_text: &str) -> Result<Vec<String>> {
        let text_lower = partial_text.to_lowercase();
        let mut suggestions = Vec::new();
        
        if text_lower.starts_with("buy") {
            suggestions.extend(vec![
                "buy 10 SOL when price drops 5%".to_string(),
                "buy 100 USDC worth of BTC".to_string(),
                "buy the dip on ETH".to_string(),
            ]);
        } else if text_lower.starts_with("sell") {
            suggestions.extend(vec![
                "sell half my SOL position".to_string(),
                "sell all BTC at 5% profit".to_string(),
                "sell when RSI > 70".to_string(),
            ]);
        } else if text_lower.starts_with("monitor") || text_lower.starts_with("watch") {
            suggestions.extend(vec![
                "monitor SOL for price breakout".to_string(),
                "watch portfolio for 10% gains".to_string(),
                "monitor BTC volume spikes".to_string(),
            ]);
        } else if text_lower.starts_with("show") || text_lower.starts_with("get") {
            suggestions.extend(vec![
                "show my portfolio performance".to_string(),
                "get trading history for this week".to_string(),
                "show agent status".to_string(),
            ]);
        }
        
        Ok(suggestions)
    }
}

#[async_trait]
impl Actor for ChatCommandProcessor {
    async fn init(&mut self, _budget: Budget) -> Result<()> {
        log::info!("Chat Command Processor initialized");
        Ok(())
    }

    async fn handle(&mut self, message: Message, _budget: Budget) -> Result<()> {
        match message.message_type.as_str() {
            "chat_command" => {
                if let Ok(text) = serde_json::from_value::<String>(message.payload.clone()) {
                    let command = self.process_command(
                        &text,
                        &message.sender,
                        message.headers.get("channel_id").unwrap_or(&"default".to_string())
                    ).await?;
                    
                    log::info!("Processed command: {:?} with confidence: {:.2}", 
                              command.intent, command.confidence);
                }
            },
            "feedback" => {
                if let Ok(feedback_data) = serde_json::from_value::<serde_json::Value>(message.payload.clone()) {
                    if let (Some(command_id), Some(feedback), rating) = (
                        feedback_data.get("command_id").and_then(|v| v.as_str()),
                        feedback_data.get("feedback").and_then(|v| v.as_str()),
                        feedback_data.get("rating").and_then(|v| v.as_f64()),
                    ) {
                        self.provide_feedback(command_id, feedback, rating).await?;
                    }
                }
            },
            _ => {},
        }
        Ok(())
    }

    async fn tick(&mut self, _budget: Budget) -> Result<()> {
        // Periodic learning and pattern optimization
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        log::info!("Chat Command Processor shutting down");
        Ok(())
    }
}

// Mock NLP Service for development
pub struct MockNLPService;

#[async_trait]
impl NLPService for MockNLPService {
    async fn extract_intent(&self, text: &str) -> Result<CommandIntent> {
        let text_lower = text.to_lowercase();
        
        if text_lower.contains("buy") || text_lower.contains("purchase") {
            Ok(CommandIntent::Trade {
                action: TradeAction::Buy,
                token: None,
                amount: None,
                conditions: vec![],
            })
        } else if text_lower.contains("sell") {
            Ok(CommandIntent::Trade {
                action: TradeAction::Sell,
                token: None,
                amount: None,
                conditions: vec![],
            })
        } else if text_lower.contains("monitor") || text_lower.contains("watch") {
            Ok(CommandIntent::Monitor {
                target: MonitorTarget::Portfolio,
                frequency: None,
                alerts: vec![],
            })
        } else if text_lower.contains("show") || text_lower.contains("get") || text_lower.contains("status") {
            Ok(CommandIntent::Query {
                subject: QuerySubject::Portfolio,
                filters: HashMap::new(),
                format: ResponseFormat::Brief,
            })
        } else if text_lower.contains("stop") || text_lower.contains("emergency") {
            Ok(CommandIntent::Control {
                action: ControlAction::Emergency,
                scope: ControlScope::AllAgents,
                parameters: HashMap::new(),
            })
        } else {
            Ok(CommandIntent::Query {
                subject: QuerySubject::Portfolio,
                filters: HashMap::new(),
                format: ResponseFormat::Brief,
            })
        }
    }

    async fn extract_entities(&self, text: &str) -> Result<HashMap<String, String>> {
        let mut entities = HashMap::new();
        
        // Extract numbers (amounts, prices)
        let number_regex = Regex::new(r"\b(\d+(?:\.\d+)?)\b")?;
        if let Some(captures) = number_regex.captures(text) {
            entities.insert("amount".to_string(), captures[1].to_string());
        }
        
        // Extract token symbols
        let token_regex = Regex::new(r"\b([A-Z]{2,10})\b")?;
        if let Some(captures) = token_regex.captures(text) {
            entities.insert("token".to_string(), captures[1].to_string());
        }
        
        // Extract percentages
        let percentage_regex = Regex::new(r"(\d+(?:\.\d+)?)%")?;
        if let Some(captures) = percentage_regex.captures(text) {
            entities.insert("percentage".to_string(), captures[1].to_string());
        }
        
        Ok(entities)
    }

    async fn calculate_confidence(&self, _text: &str, _intent: &CommandIntent) -> Result<f64> {
        Ok(0.75) // Mock confidence score
    }

    async fn learn_from_feedback(&self, _command: &ChatCommand, _feedback: &str) -> Result<()> {
        // Mock learning implementation
        Ok(())
    }
}