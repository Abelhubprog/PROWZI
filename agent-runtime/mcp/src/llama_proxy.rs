use crate::{McpError, McpProxy, McpRequest, McpResponse, McpTool, McpUsage};
use anyhow::Result;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, instrument, warn};

/// Default endpoint for local Llama inference server
const DEFAULT_LLAMA_ENDPOINT: &str = "http://localhost:8080";

/// Default model to use if not specified
const DEFAULT_MODEL: &str = "llama-3-8b-instruct";

/// Llama API request for completions
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlamaRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Llama API response for completions
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlamaResponse {
    text: String,
    #[serde(default)]
    usage: Option<LlamaUsage>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// Llama API usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlamaUsage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
}

/// Llama MCP proxy implementation
pub struct LlamaProxy {
    client: Client,
    endpoint: String,
}

impl LlamaProxy {
    /// Create a new Llama proxy
    pub fn new() -> Result<Self, McpError> {
        // Get endpoint from environment variable or use default
        let endpoint = env::var("LLAMA_MCP_ENDPOINT")
            .unwrap_or_else(|_| DEFAULT_LLAMA_ENDPOINT.to_string());

        // Create HTTP client with timeouts
        let client = Client::builder()
            .timeout(Duration::from_secs(120)) // Local inference can take longer
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| McpError::InternalError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            endpoint,
        })
    }

    /// Convert generic parameters to Llama-specific format
    fn prepare_request(&self, params: &Value) -> Result<LlamaRequest, McpError> {
        // Extract model or use default
        let model = params["model"]
            .as_str()
            .unwrap_or(DEFAULT_MODEL)
            .to_string();

        // Extract prompt from different possible formats
        let prompt = if let Some(prompt_str) = params["prompt"].as_str() {
            prompt_str.to_string()
        } else if let Some(messages) = params["messages"].as_array() {
            // Convert chat messages to a single prompt
            let mut prompt_parts = Vec::new();
            
            for msg in messages {
                let role = msg["role"].as_str().unwrap_or("user");
                let content = match msg["content"].as_str() {
                    Some(content_str) => content_str,
                    None => return Err(McpError::InvalidRequest(
                        "Message content must be a string".to_string(),
                    )),
                };
                
                match role {
                    "system" => prompt_parts.push(format!("System: {}\n", content)),
                    "user" => prompt_parts.push(format!("User: {}\n", content)),
                    "assistant" => prompt_parts.push(format!("Assistant: {}\n", content)),
                    _ => prompt_parts.push(format!("{}: {}\n", role, content)),
                }
            }
            
            // Add final prompt for the assistant to continue
            prompt_parts.push("Assistant: ".to_string());
            prompt_parts.join("")
        } else if let Some(query) = params["query"].as_str() {
            // Simple query format
            format!("User: {}\nAssistant: ", query)
        } else {
            return Err(McpError::InvalidRequest(
                "Request must contain 'prompt', 'messages', or 'query'".to_string(),
            ));
        };

        // Extract optional parameters
        let max_tokens = params["max_tokens"].as_u64().map(|v| v as u32);
        let temperature = params["temperature"].as_f64().map(|v| v as f32);
        let top_p = params["top_p"].as_f64().map(|v| v as f32);
        let top_k = params["top_k"].as_u64().map(|v| v as u32);
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

        Ok(LlamaRequest {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            stop,
            stream,
        })
    }

    /// Make an API call to local Llama inference server
    #[instrument(skip(self, request), fields(model = %request.model))]
    async fn call_llama_api(
        &self,
        request: LlamaRequest,
    ) -> Result<LlamaResponse, McpError> {
        // Make the API call
        let start_time = Instant::now();
        
        debug!(
            endpoint = %self.endpoint,
            model = %request.model,
            "Calling local Llama inference server"
        );
        
        let response = self
            .client
            .post(format!("{}/generate", self.endpoint))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    McpError::TimeoutError("Llama".to_string())
                } else if e.is_connect() {
                    McpError::ConnectionError("Llama".to_string(), 
                        format!("Failed to connect to local Llama server at {}: {}", self.endpoint, e))
                } else {
                    McpError::ApiError("Llama".to_string(), e.to_string())
                }
            })?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            McpError::ApiError(
                "Llama".to_string(),
                format!("Failed to read response body: {}", e),
            )
        })?;

        if !status.is_success() {
            let error_msg = match status {
                StatusCode::BAD_REQUEST => format!("Bad request: {}", response_text),
                StatusCode::SERVICE_UNAVAILABLE => "Llama inference server is unavailable".to_string(),
                _ => format!("API error ({}): {}", status, response_text),
            };

            return Err(McpError::ApiError("Llama".to_string(), error_msg));
        }

        // Parse the response
        let llama_response: LlamaResponse = serde_json::from_str(&response_text).map_err(|e| {
            McpError::ApiError(
                "Llama".to_string(),
                format!("Failed to parse response: {}", e),
            )
        })?;

        let duration = start_time.elapsed();
        info!(
            model = %request.model,
            duration_ms = %duration.as_millis(),
            "Llama inference completed"
        );

        Ok(llama_response)
    }
}

impl McpProxy for LlamaProxy {
    fn process(&self, request: McpRequest) -> Result<McpResponse, McpError> {
        // Check for mock mode
        if env::var("MOCK_MODE").unwrap_or_default() == "true" {
            return Ok(McpResponse {
                data: json!({
                    "text": "This is a mock response from the local Llama model. The capital of France is Paris, which is known for the Eiffel Tower and the Louvre Museum.",
                    "model": "llama-3-8b-instruct-mock"
                }),
                error: None,
                usage: Some(McpUsage {
                    prompt_tokens: Some(20),
                    completion_tokens: Some(30),
                    total_tokens: Some(50),
                    cost_usd: Some(0.0), // Local inference has no direct cost
                    latency_ms: Some(200),
                }),
                request_id: request.request_id.unwrap_or_else(|| "mock-llama-request".to_string()),
            });
        }

        // Process the request
        let runtime = tokio::runtime::Handle::current();
        let llama_request = self.prepare_request(&request.params)?;

        // Make the API call
        let start_time = Instant::now();
        let llama_response = match runtime.block_on(self.call_llama_api(llama_request.clone())) {
            Ok(response) => response,
            Err(e) => {
                // If the Llama server is down, return a minimal response
                if let McpError::ConnectionError(_, _) = e {
                    warn!(
                        error = %e,
                        "Llama server connection failed, returning minimal response"
                    );
                    
                    // Generate a minimal response based on the prompt
                    let minimal_text = if llama_request.prompt.len() > 100 {
                        format!(
                            "Unable to process request due to Llama server unavailability. Your query was about: {}...",
                            &llama_request.prompt[..100]
                        )
                    } else {
                        format!(
                            "Unable to process request due to Llama server unavailability. Your query was: {}",
                            llama_request.prompt
                        )
                    };
                    
                    return Ok(McpResponse {
                        data: json!({
                            "text": minimal_text,
                            "model": llama_request.model,
                            "fallback": true
                        }),
                        error: Some(e.to_string()),
                        usage: Some(McpUsage {
                            prompt_tokens: Some(llama_request.prompt.split_whitespace().count() as u32),
                            completion_tokens: Some(0),
                            total_tokens: Some(llama_request.prompt.split_whitespace().count() as u32),
                            cost_usd: Some(0.0),
                            latency_ms: Some(start_time.elapsed().as_millis() as u64),
                        }),
                        request_id: request.request_id.unwrap_or_else(|| "fallback-llama-request".to_string()),
                    });
                }
                
                return Err(e);
            }
        };

        // Estimate token counts if not provided by the server
        let prompt_tokens = llama_response.usage
            .as_ref()
            .and_then(|u| u.prompt_tokens)
            .unwrap_or_else(|| (llama_request.prompt.split_whitespace().count() as u32));
            
        let completion_tokens = llama_response.usage
            .as_ref()
            .and_then(|u| u.completion_tokens)
            .unwrap_or_else(|| (llama_response.text.split_whitespace().count() as u32));
            
        let total_tokens = llama_response.usage
            .as_ref()
            .and_then(|u| u.total_tokens)
            .unwrap_or_else(|| prompt_tokens + completion_tokens);

        // Prepare the MCP response
        let mcp_response = McpResponse {
            data: json!({
                "text": llama_response.text,
                "model": llama_response.model.unwrap_or_else(|| llama_request.model.clone())
            }),
            error: None,
            usage: Some(McpUsage {
                prompt_tokens: Some(prompt_tokens),
                completion_tokens: Some(completion_tokens),
                total_tokens: Some(total_tokens),
                cost_usd: Some(0.0), // Local inference has no direct cost
                latency_ms: Some(start_time.elapsed().as_millis() as u64),
            }),
            request_id: request.request_id.unwrap_or_else(|| {
                format!("llama-{}", chrono::Utc::now().timestamp_millis())
            }),
        };

        Ok(mcp_response)
    }

    fn tool_type(&self) -> McpTool {
        McpTool::LlamaLocal
    }

    fn health_check(&self) -> bool {
        // For mock mode, always return healthy
        if env::var("MOCK_MODE").unwrap_or_default() == "true" {
            return true;
        }
        
        // Try to connect to the Llama server
        let runtime = tokio::runtime::Handle::current();
        let result = runtime.block_on(async {
            match self.client.get(format!("{}/health", self.endpoint)).send().await {
                Ok(response) => response.status().is_success(),
                Err(_) => false,
            }
        });
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::McpRequest;
    use serde_json::json;

    #[test]
    fn test_prepare_request() {
        // Set mock mode for testing
        env::set_var("MOCK_MODE", "true");

        let proxy = LlamaProxy::new().unwrap();

        // Test with simple prompt
        let params = json!({
            "model": "llama-3-8b-instruct",
            "prompt": "What is the capital of France?",
            "max_tokens": 100,
            "temperature": 0.7
        });

        let request = proxy.prepare_request(&params).unwrap();
        assert_eq!(request.model, "llama-3-8b-instruct");
        assert_eq!(request.prompt, "What is the capital of France?");
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));

        // Test with chat messages
        let params = json!({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        });

        let request = proxy.prepare_request(&params).unwrap();
        assert_eq!(request.model, DEFAULT_MODEL);
        assert!(request.prompt.contains("System: You are a helpful assistant"));
        assert!(request.prompt.contains("User: What is the capital of France?"));
        assert!(request.prompt.ends_with("Assistant: "));

        // Test with query format
        let params = json!({
            "query": "What is the capital of France?"
        });

        let request = proxy.prepare_request(&params).unwrap();
        assert_eq!(request.prompt, "User: What is the capital of France?\nAssistant: ");

        // Clean up
        env::remove_var("MOCK_MODE");
    }

    #[test]
    fn test_mock_response() {
        // Set mock mode for testing
        env::set_var("MOCK_MODE", "true");

        let proxy = LlamaProxy::new().unwrap();
        let request = McpRequest {
            tool: McpTool::LlamaLocal,
            params: json!({
                "prompt": "What is the capital of France?"
            }),
            tenant_id: None,
            timeout_ms: None,
            request_id: Some("test-llama-request".to_string()),
        };

        let response = proxy.process(request).unwrap();
        assert!(response.data["text"].is_string());
        assert!(response.usage.is_some());
        assert_eq!(response.usage.as_ref().unwrap().cost_usd, Some(0.0)); // Local inference has no cost
        assert_eq!(response.request_id, "test-llama-request");

        // Clean up
        env::remove_var("MOCK_MODE");
    }
}
