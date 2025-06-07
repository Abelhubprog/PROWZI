use crate::{McpError, McpProxy, McpRequest, McpResponse, McpTool, McpUsage};
use anyhow::Result;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, error, info, instrument, warn};

/// Default API endpoint for Anthropic Claude
const CLAUDE_API_ENDPOINT: &str = "https://api.anthropic.com";

/// Default model to use if not specified
const DEFAULT_MODEL: &str = "claude-3-sonnet-20240229";

/// Default API version
const DEFAULT_API_VERSION: &str = "2023-06-01";

/// Rate limit: requests per minute
const RATE_LIMIT_RPM: u32 = 50;

/// Cost per 1K tokens (in USD) for input - Claude 3 Sonnet pricing
const COST_PER_1K_INPUT_TOKENS: f64 = 0.003;

/// Cost per 1K tokens (in USD) for output - Claude 3 Sonnet pricing
const COST_PER_1K_OUTPUT_TOKENS: f64 = 0.015;

/// Anthropic Claude API request for messages
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaudeMessagesRequest {
    model: String,
    messages: Vec<ClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

/// Claude API message format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaudeMessage {
    role: String,
    content: Vec<ClaudeContent>,
}

/// Claude API content format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaudeContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

/// Claude API response for messages
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaudeMessagesResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<ClaudeContent>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: ClaudeUsage,
}

/// Claude API usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaudeUsage {
    input_tokens: u32,
    output_tokens: u32,
}

/// Rate limiter for Claude API
struct RateLimiter {
    last_request_time: Instant,
    request_count: u32,
    max_rpm: u32,
}

impl RateLimiter {
    fn new(max_rpm: u32) -> Self {
        Self {
            last_request_time: Instant::now(),
            request_count: 0,
            max_rpm,
        }
    }

    /// Check if a request can be made, and update the rate limiter state
    fn check_and_update(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_request_time);

        // Reset counter if a minute has passed
        if elapsed >= Duration::from_secs(60) {
            self.request_count = 0;
            self.last_request_time = now;
        }

        // Check if we're under the rate limit
        if self.request_count < self.max_rpm {
            self.request_count += 1;
            true
        } else {
            false
        }
    }
}

/// Claude MCP proxy implementation
pub struct ClaudeProxy {
    client: Client,
    api_key: String,
    endpoint: String,
    api_version: String,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl ClaudeProxy {
    /// Create a new Claude proxy
    pub fn new() -> Result<Self, McpError> {
        // Check for API key
        let api_key = match env::var("ANTHROPIC_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                if env::var("MOCK_MODE").unwrap_or_default() == "true" {
                    "mock_api_key".to_string()
                } else {
                    return Err(McpError::MissingApiKey("Anthropic Claude".to_string()));
                }
            }
        };

        // Get custom endpoint if specified
        let endpoint = env::var("CLAUDE_API_ENDPOINT")
            .unwrap_or_else(|_| CLAUDE_API_ENDPOINT.to_string());

        // Get API version if specified
        let api_version = env::var("CLAUDE_API_VERSION")
            .unwrap_or_else(|_| DEFAULT_API_VERSION.to_string());

        // Create HTTP client with timeouts
        let client = Client::builder()
            .timeout(Duration::from_secs(60)) // Claude can take longer for large contexts
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| McpError::InternalError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            api_key,
            endpoint,
            api_version,
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(RATE_LIMIT_RPM))),
        })
    }

    /// Convert generic parameters to Claude-specific format
    fn prepare_messages_request(&self, params: &Value) -> Result<ClaudeMessagesRequest, McpError> {
        // Extract model or use default
        let model = params["model"]
            .as_str()
            .unwrap_or(DEFAULT_MODEL)
            .to_string();

        // Extract system prompt if present
        let system = params["system"].as_str().map(|s| s.to_string());

        // Extract messages
        let messages = if let Some(messages_value) = params["messages"].as_array() {
            let mut claude_messages = Vec::new();
            for msg_value in messages_value {
                let role = msg_value["role"]
                    .as_str()
                    .ok_or_else(|| {
                        McpError::InvalidRequest("Message must have a 'role' field".to_string())
                    })?
                    .to_string();

                // Convert OpenAI-style messages to Claude format
                // Claude only supports 'user' and 'assistant' roles
                let adjusted_role = match role.as_str() {
                    "system" => {
                        // If we find a system message, add it to the system parameter
                        if system.is_none() && msg_value["content"].is_string() {
                            return Ok(ClaudeMessagesRequest {
                                model,
                                messages: claude_messages,
                                max_tokens: None,
                                temperature: None,
                                top_p: None,
                                top_k: None,
                                stop_sequences: None,
                                stream: None,
                                system: Some(msg_value["content"].as_str().unwrap().to_string()),
                            });
                        }
                        continue; // Skip this message in the messages array
                    }
                    "user" => "user",
                    "assistant" => "assistant",
                    _ => "user", // Default to user for unknown roles
                };

                let content_text = if msg_value["content"].is_string() {
                    msg_value["content"].as_str().unwrap().to_string()
                } else if msg_value["content"].is_array() {
                    // Handle complex content array (not fully supported yet)
                    // Just extract text parts for now
                    let mut text_parts = Vec::new();
                    for content_part in msg_value["content"].as_array().unwrap() {
                        if content_part["type"].as_str() == Some("text") && content_part["text"].is_string() {
                            text_parts.push(content_part["text"].as_str().unwrap());
                        }
                    }
                    text_parts.join("\n")
                } else {
                    return Err(McpError::InvalidRequest(
                        "Message content must be a string or content array".to_string(),
                    ));
                };

                claude_messages.push(ClaudeMessage {
                    role: adjusted_role.to_string(),
                    content: vec![ClaudeContent {
                        content_type: "text".to_string(),
                        text: content_text,
                    }],
                });
            }
            claude_messages
        } else if let Some(prompt) = params["prompt"].as_str() {
            // Handle simple prompt format
            vec![ClaudeMessage {
                role: "user".to_string(),
                content: vec![ClaudeContent {
                    content_type: "text".to_string(),
                    text: prompt.to_string(),
                }],
            }]
        } else {
            return Err(McpError::InvalidRequest(
                "Request must contain either 'messages' or 'prompt'".to_string(),
            ));
        };

        // Extract optional parameters
        let max_tokens = params["max_tokens"].as_u64().map(|v| v as u32);
        let temperature = params["temperature"].as_f64().map(|v| v as f32);
        let top_p = params["top_p"].as_f64().map(|v| v as f32);
        let top_k = params["top_k"].as_u64().map(|v| v as u32);
        let stream = params["stream"].as_bool();

        // Extract stop sequences
        let stop_sequences = if let Some(stop_array) = params["stop_sequences"].as_array() {
            let mut sequences = Vec::new();
            for stop_value in stop_array {
                if let Some(stop_str) = stop_value.as_str() {
                    sequences.push(stop_str.to_string());
                }
            }
            if sequences.is_empty() {
                None
            } else {
                Some(sequences)
            }
        } else if let Some(stop_array) = params["stop"].as_array() {
            // Support for OpenAI-style "stop" parameter
            let mut sequences = Vec::new();
            for stop_value in stop_array {
                if let Some(stop_str) = stop_value.as_str() {
                    sequences.push(stop_str.to_string());
                }
            }
            if sequences.is_empty() {
                None
            } else {
                Some(sequences)
            }
        } else if let Some(stop_str) = params["stop"].as_str() {
            Some(vec![stop_str.to_string()])
        } else {
            None
        };

        Ok(ClaudeMessagesRequest {
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            top_k,
            stop_sequences,
            stream,
            system,
        })
    }

    /// Calculate cost based on token usage
    fn calculate_cost(&self, usage: &ClaudeUsage) -> f64 {
        let input_cost = (usage.input_tokens as f64 / 1000.0) * COST_PER_1K_INPUT_TOKENS;
        let output_cost = (usage.output_tokens as f64 / 1000.0) * COST_PER_1K_OUTPUT_TOKENS;
        input_cost + output_cost
    }

    /// Make an API call to Claude
    #[instrument(skip(self, request), fields(model = %request.model))]
    async fn call_claude_api(
        &self,
        request: ClaudeMessagesRequest,
    ) -> Result<ClaudeMessagesResponse, McpError> {
        // Check rate limits
        let can_proceed = {
            let mut rate_limiter = self.rate_limiter.lock().await;
            rate_limiter.check_and_update()
        };

        if !can_proceed {
            return Err(McpError::RateLimitExceeded("Claude".to_string()));
        }

        // Make the API call
        let start_time = Instant::now();
        let response = self
            .client
            .post(format!("{}/v1/messages", self.endpoint))
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    McpError::TimeoutError("Claude".to_string())
                } else if e.is_connect() {
                    McpError::ConnectionError("Claude".to_string(), e.to_string())
                } else {
                    McpError::ApiError("Claude".to_string(), e.to_string())
                }
            })?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            McpError::ApiError(
                "Claude".to_string(),
                format!("Failed to read response body: {}", e),
            )
        })?;

        if !status.is_success() {
            let error_msg = match status {
                StatusCode::UNAUTHORIZED => "Invalid API key".to_string(),
                StatusCode::TOO_MANY_REQUESTS => "Rate limit exceeded".to_string(),
                StatusCode::BAD_REQUEST => format!("Bad request: {}", response_text),
                _ => format!("API error ({}): {}", status, response_text),
            };

            return Err(McpError::ApiError("Claude".to_string(), error_msg));
        }

        // Parse the response
        let claude_response: ClaudeMessagesResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                McpError::ApiError(
                    "Claude".to_string(),
                    format!("Failed to parse response: {}", e),
                )
            })?;

        let duration = start_time.elapsed();
        info!(
            model = %request.model,
            duration_ms = %duration.as_millis(),
            input_tokens = %claude_response.usage.input_tokens,
            output_tokens = %claude_response.usage.output_tokens,
            "Claude API call completed"
        );

        Ok(claude_response)
    }
}

impl McpProxy for ClaudeProxy {
    fn process(&self, request: McpRequest) -> Result<McpResponse, McpError> {
        // Check for mock mode
        if env::var("MOCK_MODE").unwrap_or_default() == "true" {
            return Ok(McpResponse {
                data: json!({
                    "text": "This is a mock response from Claude 4 Sonnet. The capital of France is Paris, and it's known for the Eiffel Tower, Louvre Museum, and exquisite cuisine.",
                    "model": "claude-3-sonnet-20240229-mock"
                }),
                error: None,
                usage: Some(McpUsage {
                    prompt_tokens: Some(15),
                    completion_tokens: Some(25),
                    total_tokens: Some(40),
                    cost_usd: Some(0.0015),
                    latency_ms: Some(120),
                }),
                request_id: request.request_id.unwrap_or_else(|| "mock-claude-request".to_string()),
            });
        }

        // Process the request
        let runtime = tokio::runtime::Handle::current();
        let messages_request = self.prepare_messages_request(&request.params)?;

        // Make the API call
        let start_time = Instant::now();
        let claude_response = runtime.block_on(self.call_claude_api(messages_request))?;

        // Extract the response text
        let response_text = if !claude_response.content.is_empty() {
            claude_response.content.iter()
                .filter(|c| c.content_type == "text")
                .map(|c| c.text.clone())
                .collect::<Vec<String>>()
                .join("\n")
        } else {
            return Err(McpError::ApiError(
                "Claude".to_string(),
                "No content returned in response".to_string(),
            ));
        };

        // Calculate cost
        let cost = self.calculate_cost(&claude_response.usage);

        // Prepare the MCP response
        let mcp_response = McpResponse {
            data: json!({
                "text": response_text,
                "model": claude_response.model
            }),
            error: None,
            usage: Some(McpUsage {
                prompt_tokens: Some(claude_response.usage.input_tokens),
                completion_tokens: Some(claude_response.usage.output_tokens),
                total_tokens: Some(claude_response.usage.input_tokens + claude_response.usage.output_tokens),
                cost_usd: Some(cost),
                latency_ms: Some(start_time.elapsed().as_millis() as u64),
            }),
            request_id: request
                .request_id
                .unwrap_or_else(|| claude_response.id.clone()),
        };

        Ok(mcp_response)
    }

    fn tool_type(&self) -> McpTool {
        McpTool::ClaudeSonnet
    }

    fn health_check(&self) -> bool {
        // Simple health check - just verify the API key is set
        if env::var("MOCK_MODE").unwrap_or_default() == "true" {
            return true;
        }
        env::var("ANTHROPIC_API_KEY").is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::McpRequest;
    use serde_json::json;

    #[test]
    fn test_prepare_messages_request() {
        // Set mock mode for testing
        env::set_var("MOCK_MODE", "true");

        let proxy = ClaudeProxy::new().unwrap();

        // Test with messages format
        let params = json!({
            "model": "claude-3-sonnet-20240229",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        });

        let request = proxy.prepare_messages_request(&params).unwrap();
        assert_eq!(request.model, "claude-3-sonnet-20240229");
        assert_eq!(request.messages.len(), 1); // System message is moved to system parameter
        assert_eq!(request.messages[0].role, "user");
        assert_eq!(request.messages[0].content[0].text, "What is the capital of France?");
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.system, Some("You are a helpful assistant".to_string()));

        // Test with simple prompt format
        let params = json!({
            "prompt": "What is the capital of France?"
        });

        let request = proxy.prepare_messages_request(&params).unwrap();
        assert_eq!(request.model, DEFAULT_MODEL);
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, "user");
        assert_eq!(request.messages[0].content[0].text, "What is the capital of France?");

        // Clean up
        env::remove_var("MOCK_MODE");
    }

    #[test]
    fn test_calculate_cost() {
        let proxy = ClaudeProxy::new().unwrap();
        let usage = ClaudeUsage {
            input_tokens: 10,
            output_tokens: 20,
        };

        let cost = proxy.calculate_cost(&usage);
        let expected_cost = (10.0 / 1000.0) * COST_PER_1K_INPUT_TOKENS
            + (20.0 / 1000.0) * COST_PER_1K_OUTPUT_TOKENS;
        assert_eq!(cost, expected_cost);
    }

    #[test]
    fn test_mock_response() {
        // Set mock mode for testing
        env::set_var("MOCK_MODE", "true");

        let proxy = ClaudeProxy::new().unwrap();
        let request = McpRequest {
            tool: McpTool::ClaudeSonnet,
            params: json!({
                "prompt": "What is the capital of France?"
            }),
            tenant_id: None,
            timeout_ms: None,
            request_id: Some("test-claude-request".to_string()),
        };

        let response = proxy.process(request).unwrap();
        assert!(response.data["text"].is_string());
        assert!(response.usage.is_some());
        assert_eq!(response.request_id, "test-claude-request");

        // Clean up
        env::remove_var("MOCK_MODE");
    }
}
