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

/// Default API endpoint for Perplexity
const PERPLEXITY_API_ENDPOINT: &str = "https://api.perplexity.ai";

/// Default model to use if not specified
const DEFAULT_MODEL: &str = "pplx-7b-online";

/// Rate limit: requests per minute
const RATE_LIMIT_RPM: u32 = 60;

/// Cost per 1K tokens (in USD) for input
const COST_PER_1K_INPUT_TOKENS: f64 = 0.0001;

/// Cost per 1K tokens (in USD) for output
const COST_PER_1K_OUTPUT_TOKENS: f64 = 0.0002;

/// Perplexity API request for chat completions
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerplexityChatRequest {
    model: String,
    messages: Vec<PerplexityMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

/// Perplexity API message format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerplexityMessage {
    role: String,
    content: String,
}

/// Perplexity API response for chat completions
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerplexityChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<PerplexityChoice>,
    usage: PerplexityUsage,
}

/// Perplexity API choice in response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerplexityChoice {
    index: u32,
    message: PerplexityMessage,
    finish_reason: String,
}

/// Perplexity API usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerplexityUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Rate limiter for Perplexity API
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

/// Perplexity MCP proxy implementation
pub struct PerplexityProxy {
    client: Client,
    api_key: String,
    endpoint: String,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl PerplexityProxy {
    /// Create a new Perplexity proxy
    pub fn new() -> Result<Self, McpError> {
        // Check for API key
        let api_key = match env::var("PERPLEXITY_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                if env::var("MOCK_MODE").unwrap_or_default() == "true" {
                    "mock_api_key".to_string()
                } else {
                    return Err(McpError::MissingApiKey("Perplexity".to_string()));
                }
            }
        };

        // Get custom endpoint if specified
        let endpoint = env::var("PERPLEXITY_API_ENDPOINT")
            .unwrap_or_else(|_| PERPLEXITY_API_ENDPOINT.to_string());

        // Create HTTP client with timeouts
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| McpError::InternalError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            api_key,
            endpoint,
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new(RATE_LIMIT_RPM))),
        })
    }

    /// Convert generic parameters to Perplexity-specific format
    fn prepare_chat_request(&self, params: &Value) -> Result<PerplexityChatRequest, McpError> {
        // Extract model or use default
        let model = params["model"]
            .as_str()
            .unwrap_or(DEFAULT_MODEL)
            .to_string();

        // Extract messages
        let messages = if let Some(messages_value) = params["messages"].as_array() {
            let mut perplexity_messages = Vec::new();
            for msg_value in messages_value {
                let role = msg_value["role"]
                    .as_str()
                    .ok_or_else(|| {
                        McpError::InvalidRequest("Message must have a 'role' field".to_string())
                    })?
                    .to_string();

                let content = msg_value["content"]
                    .as_str()
                    .ok_or_else(|| {
                        McpError::InvalidRequest("Message must have a 'content' field".to_string())
                    })?
                    .to_string();

                perplexity_messages.push(PerplexityMessage { role, content });
            }
            perplexity_messages
        } else if let Some(query) = params["query"].as_str() {
            // Handle simple query format
            vec![
                PerplexityMessage {
                    role: "system".to_string(),
                    content: "You are a helpful assistant that provides accurate information with references.".to_string(),
                },
                PerplexityMessage {
                    role: "user".to_string(),
                    content: query.to_string(),
                },
            ]
        } else if let Some(prompt) = params["prompt"].as_str() {
            // Handle simple prompt format
            vec![
                PerplexityMessage {
                    role: "system".to_string(),
                    content: "You are a helpful assistant that provides accurate information with references.".to_string(),
                },
                PerplexityMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ]
        } else {
            return Err(McpError::InvalidRequest(
                "Request must contain either 'messages', 'query', or 'prompt'".to_string(),
            ));
        };

        // Extract optional parameters
        let max_tokens = params["max_tokens"].as_u64().map(|v| v as u32);
        let temperature = params["temperature"].as_f64().map(|v| v as f32);
        let top_p = params["top_p"].as_f64().map(|v| v as f32);
        let presence_penalty = params["presence_penalty"].as_f64().map(|v| v as f32);
        let frequency_penalty = params["frequency_penalty"].as_f64().map(|v| v as f32);
        let stream = params["stream"].as_bool();

        // Extract stop sequences
        let stop = if let Some(stop_array) = params["stop"].as_array() {
            let mut stop_sequences = Vec::new();
            for stop_value in stop_array {
                if let Some(stop_str) = stop_value.as_str() {
                    stop_sequences.push(stop_str.to_string());
                }
            }
            if stop_sequences.is_empty() {
                None
            } else {
                Some(stop_sequences)
            }
        } else if let Some(stop_str) = params["stop"].as_str() {
            Some(vec![stop_str.to_string()])
        } else {
            None
        };

        Ok(PerplexityChatRequest {
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            stream,
            stop,
        })
    }

    /// Calculate cost based on token usage
    fn calculate_cost(&self, usage: &PerplexityUsage) -> f64 {
        let input_cost = (usage.prompt_tokens as f64 / 1000.0) * COST_PER_1K_INPUT_TOKENS;
        let output_cost = (usage.completion_tokens as f64 / 1000.0) * COST_PER_1K_OUTPUT_TOKENS;
        input_cost + output_cost
    }

    /// Make an API call to Perplexity
    #[instrument(skip(self, request), fields(model = %request.model))]
    async fn call_perplexity_api(
        &self,
        request: PerplexityChatRequest,
    ) -> Result<PerplexityChatResponse, McpError> {
        // Check rate limits
        let can_proceed = {
            let mut rate_limiter = self.rate_limiter.lock().await;
            rate_limiter.check_and_update()
        };

        if !can_proceed {
            return Err(McpError::RateLimitExceeded("Perplexity".to_string()));
        }

        // Make the API call
        let start_time = Instant::now();
        let response = self
            .client
            .post(format!("{}/chat/completions", self.endpoint))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    McpError::TimeoutError("Perplexity".to_string())
                } else if e.is_connect() {
                    McpError::ConnectionError("Perplexity".to_string(), e.to_string())
                } else {
                    McpError::ApiError("Perplexity".to_string(), e.to_string())
                }
            })?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            McpError::ApiError(
                "Perplexity".to_string(),
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

            return Err(McpError::ApiError("Perplexity".to_string(), error_msg));
        }

        // Parse the response
        let perplexity_response: PerplexityChatResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                McpError::ApiError(
                    "Perplexity".to_string(),
                    format!("Failed to parse response: {}", e),
                )
            })?;

        let duration = start_time.elapsed();
        info!(
            model = %request.model,
            duration_ms = %duration.as_millis(),
            tokens = %perplexity_response.usage.total_tokens,
            "Perplexity API call completed"
        );

        Ok(perplexity_response)
    }
}

impl McpProxy for PerplexityProxy {
    fn process(&self, request: McpRequest) -> Result<McpResponse, McpError> {
        // Check for mock mode
        if env::var("MOCK_MODE").unwrap_or_default() == "true" {
            return Ok(McpResponse {
                data: json!({
                    "text": "This is a mock response from the Perplexity API",
                    "references": [
                        {"title": "Mock Reference 1", "url": "https://example.com/1"},
                        {"title": "Mock Reference 2", "url": "https://example.com/2"}
                    ]
                }),
                error: None,
                usage: Some(McpUsage {
                    prompt_tokens: Some(10),
                    completion_tokens: Some(20),
                    total_tokens: Some(30),
                    cost_usd: Some(0.001),
                    latency_ms: Some(50),
                }),
                request_id: request.request_id.unwrap_or_else(|| "mock-request".to_string()),
            });
        }

        // Process the request
        let runtime = tokio::runtime::Handle::current();
        let chat_request = self.prepare_chat_request(&request.params)?;

        // Make the API call
        let start_time = Instant::now();
        let perplexity_response = runtime.block_on(self.call_perplexity_api(chat_request))?;

        // Extract the response text
        let response_text = if !perplexity_response.choices.is_empty() {
            perplexity_response.choices[0].message.content.clone()
        } else {
            return Err(McpError::ApiError(
                "Perplexity".to_string(),
                "No choices returned in response".to_string(),
            ));
        };

        // Calculate cost
        let cost = self.calculate_cost(&perplexity_response.usage);

        // Prepare the MCP response
        let mcp_response = McpResponse {
            data: json!({
                "text": response_text,
                "model": perplexity_response.model
            }),
            error: None,
            usage: Some(McpUsage {
                prompt_tokens: Some(perplexity_response.usage.prompt_tokens),
                completion_tokens: Some(perplexity_response.usage.completion_tokens),
                total_tokens: Some(perplexity_response.usage.total_tokens),
                cost_usd: Some(cost),
                latency_ms: Some(start_time.elapsed().as_millis() as u64),
            }),
            request_id: request
                .request_id
                .unwrap_or_else(|| perplexity_response.id.clone()),
        };

        Ok(mcp_response)
    }

    fn tool_type(&self) -> McpTool {
        McpTool::Perplexity
    }

    fn health_check(&self) -> bool {
        // Simple health check - just verify the API key is set
        if env::var("MOCK_MODE").unwrap_or_default() == "true" {
            return true;
        }
        env::var("PERPLEXITY_API_KEY").is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::McpRequest;
    use serde_json::json;

    #[test]
    fn test_prepare_chat_request() {
        // Set mock mode for testing
        env::set_var("MOCK_MODE", "true");

        let proxy = PerplexityProxy::new().unwrap();

        // Test with messages format
        let params = json!({
            "model": "pplx-7b-online",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        });

        let request = proxy.prepare_chat_request(&params).unwrap();
        assert_eq!(request.model, "pplx-7b-online");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, "system");
        assert_eq!(request.messages[1].content, "What is the capital of France?");
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));

        // Test with simple query format
        let params = json!({
            "query": "What is the capital of France?"
        });

        let request = proxy.prepare_chat_request(&params).unwrap();
        assert_eq!(request.model, DEFAULT_MODEL);
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, "system");
        assert_eq!(request.messages[1].content, "What is the capital of France?");

        // Test with simple prompt format
        let params = json!({
            "prompt": "What is the capital of France?"
        });

        let request = proxy.prepare_chat_request(&params).unwrap();
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[1].content, "What is the capital of France?");

        // Clean up
        env::remove_var("MOCK_MODE");
    }

    #[test]
    fn test_calculate_cost() {
        let proxy = PerplexityProxy::new().unwrap();
        let usage = PerplexityUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
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

        let proxy = PerplexityProxy::new().unwrap();
        let request = McpRequest {
            tool: McpTool::Perplexity,
            params: json!({
                "query": "What is the capital of France?"
            }),
            tenant_id: None,
            timeout_ms: None,
            request_id: Some("test-request".to_string()),
        };

        let response = proxy.process(request).unwrap();
        assert!(response.data["text"].is_string());
        assert!(response.usage.is_some());
        assert_eq!(response.request_id, "test-request");

        // Clean up
        env::remove_var("MOCK_MODE");
    }
}
