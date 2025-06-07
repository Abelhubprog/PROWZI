use std::fmt;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use metrics::{counter, histogram};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{debug, error, info, instrument, warn};

/// Enum representing all available MCP tools that can be invoked by agents
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum McpTool {
    /// Perplexity API for search and RAG
    Perplexity,
    /// OpenAI GPT-4.1 for complex reasoning
    OpenAI_GPT41,
    /// Anthropic Claude 4 Sonnet for high-impact summarization
    ClaudeSonnet,
    /// Deepseek R1 for code and document retrieval
    DeepseekR1,
    /// Qwen 2.5 for code synthesis and trading agent
    Qwen25,
    /// Google Gemini Flash for ultra-fast reasoning
    GeminiFlash,
    /// Local Llama model for low-cost inference
    LlamaLocal,
    /// GitHub API for repository analysis
    GitHub,
    /// arXiv API for research paper analysis
    ArXiv,
}

impl fmt::Display for McpTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpTool::Perplexity => write!(f, "perplexity"),
            McpTool::OpenAI_GPT41 => write!(f, "openai-gpt41"),
            McpTool::ClaudeSonnet => write!(f, "claude-sonnet"),
            McpTool::DeepseekR1 => write!(f, "deepseek-r1"),
            McpTool::Qwen25 => write!(f, "qwen25"),
            McpTool::GeminiFlash => write!(f, "gemini-flash"),
            McpTool::LlamaLocal => write!(f, "llama-local"),
            McpTool::GitHub => write!(f, "github"),
            McpTool::ArXiv => write!(f, "arxiv"),
        }
    }
}

/// MCP specific error types
#[derive(Error, Debug)]
pub enum McpError {
    #[error("Invalid request format: {0}")]
    InvalidRequest(String),
    
    #[error("API key missing for {0}")]
    MissingApiKey(String),
    
    #[error("Rate limit exceeded for {0}")]
    RateLimitExceeded(String),
    
    #[error("API error from {0}: {1}")]
    ApiError(String, String),
    
    #[error("Connection error for {0}: {1}")]
    ConnectionError(String, String),
    
    #[error("Timeout error for {0}")]
    TimeoutError(String),
    
    #[error("Unsupported operation for {0}: {1}")]
    UnsupportedOperation(String, String),
    
    #[error("Internal MCP error: {0}")]
    InternalError(String),
    
    #[error("Mock mode enabled, returning mock response")]
    MockResponse,
}

/// MCP request structure
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpRequest {
    /// The tool to invoke
    pub tool: McpTool,
    
    /// Request parameters as JSON
    pub params: Value,
    
    /// Optional tenant ID for multi-tenancy
    #[serde(default)]
    pub tenant_id: Option<String>,
    
    /// Optional request timeout in milliseconds
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    
    /// Optional request ID for tracing
    #[serde(default)]
    pub request_id: Option<String>,
}

/// MCP response structure
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpResponse {
    /// Response data as JSON
    pub data: Value,
    
    /// Optional error information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    
    /// Optional usage metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<McpUsage>,
    
    /// Request ID for tracing (echoed from request or generated)
    pub request_id: String,
}

/// Usage metrics for MCP calls
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpUsage {
    /// Number of tokens in the prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    
    /// Number of tokens in the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    
    /// Total tokens used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    
    /// Cost in USD
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    
    /// Latency in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
}

/// MCP proxy trait that must be implemented by all tool proxies
pub trait McpProxy: Send + Sync {
    /// Process an MCP request and return a response
    fn process(&self, request: McpRequest) -> Result<McpResponse, McpError>;
    
    /// Get the tool type this proxy handles
    fn tool_type(&self) -> McpTool;
    
    /// Check if the proxy is healthy
    fn health_check(&self) -> bool {
        true
    }
}

/// MCP server state
pub struct McpState {
    /// Map of proxies by tool type
    proxies: Arc<Mutex<Vec<Box<dyn McpProxy>>>>,
}

impl McpState {
    /// Create a new MCP state
    pub fn new() -> Self {
        Self {
            proxies: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Register a proxy for a specific tool
    pub async fn register_proxy(&self, proxy: Box<dyn McpProxy>) {
        let tool = proxy.tool_type();
        let mut proxies = self.proxies.lock().await;
        
        // Remove any existing proxy for this tool
        proxies.retain(|p| p.tool_type() != tool);
        
        // Add the new proxy
        info!("Registering MCP proxy for tool: {}", tool);
        proxies.push(proxy);
    }
    
    /// Get a proxy for a specific tool
    pub async fn get_proxy(&self, tool: &McpTool) -> Option<Box<dyn McpProxy>> {
        let proxies = self.proxies.lock().await;
        for proxy in proxies.iter() {
            if &proxy.tool_type() == tool {
                // Clone is not available for Box<dyn Trait>, so we can't return a reference here
                // In a real implementation, we might use Arc<dyn McpProxy> instead
                return Some(proxy.clone());
            }
        }
        None
    }
}

/// Global MCP endpoint mapping
static ENDPOINT_MAP: Lazy<std::collections::HashMap<McpTool, &'static str>> = Lazy::new(|| {
    let mut map = std::collections::HashMap::new();
    map.insert(McpTool::Perplexity, "/mcp/perplexity");
    map.insert(McpTool::OpenAI_GPT41, "/mcp/openai");
    map.insert(McpTool::ClaudeSonnet, "/mcp/claude");
    map.insert(McpTool::DeepseekR1, "/mcp/deepseek");
    map.insert(McpTool::Qwen25, "/mcp/qwen");
    map.insert(McpTool::GeminiFlash, "/mcp/gemini");
    map.insert(McpTool::LlamaLocal, "/mcp/llama");
    map.insert(McpTool::GitHub, "/mcp/github");
    map.insert(McpTool::ArXiv, "/mcp/arxiv");
    map
});

/// Get the endpoint for a specific MCP tool
pub fn get_endpoint(tool: &McpTool) -> &'static str {
    ENDPOINT_MAP.get(tool).unwrap_or("/mcp/unknown")
}

/// Process an MCP request
#[instrument(skip(state), fields(tool = ?request.tool, request_id = ?request.request_id))]
async fn process_request(
    State(state): State<Arc<McpState>>,
    Json(request): Json<McpRequest>,
) -> Json<McpResponse> {
    let start_time = std::time::Instant::now();
    let tool_name = request.tool.to_string();
    let request_id = request.request_id.clone().unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        format!("req-{}-{}", timestamp, rand::random::<u16>())
    });
    
    info!(
        request_id = %request_id,
        tool = %tool_name,
        tenant_id = ?request.tenant_id,
        "Processing MCP request"
    );
    
    // Increment request counter
    counter!("mcp_requests_total", "tool" => tool_name.clone()).increment(1);
    
    // Check if mock mode is enabled
    if std::env::var("MOCK_MODE").unwrap_or_default() == "true" {
        debug!("Mock mode enabled, returning mock response");
        return Json(McpResponse {
            data: json!({ "mock": true, "tool": tool_name }),
            error: None,
            usage: Some(McpUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                total_tokens: Some(30),
                cost_usd: Some(0.0),
                latency_ms: Some(start_time.elapsed().as_millis() as u64),
            }),
            request_id,
        });
    }
    
    // Get the proxy for this tool
    let proxy_result = state.get_proxy(&request.tool).await;
    
    let response = match proxy_result {
        Some(proxy) => {
            match proxy.process(request) {
                Ok(response) => response,
                Err(err) => {
                    error!(error = %err, "Error processing MCP request");
                    counter!("mcp_errors_total", "tool" => tool_name.clone(), "type" => err.to_string()).increment(1);
                    
                    McpResponse {
                        data: json!(null),
                        error: Some(err.to_string()),
                        usage: None,
                        request_id,
                    }
                }
            }
        }
        None => {
            let error_msg = format!("No proxy registered for tool: {}", tool_name);
            error!(error = %error_msg, "MCP proxy not found");
            counter!("mcp_errors_total", "tool" => tool_name, "type" => "proxy_not_found").increment(1);
            
            McpResponse {
                data: json!(null),
                error: Some(error_msg),
                usage: None,
                request_id,
            }
        }
    };
    
    // Record request duration
    let duration_ms = start_time.elapsed().as_millis() as f64;
    histogram!("mcp_request_duration_ms", "tool" => tool_name).record(duration_ms);
    
    info!(
        request_id = %request_id,
        tool = %tool_name,
        duration_ms = %duration_ms,
        error = ?response.error,
        "MCP request completed"
    );
    
    Json(response)
}

/// Create an MCP router with all endpoints
pub fn create_mcp_router(state: Arc<McpState>) -> Router {
    Router::new()
        .route("/mcp/process", post(process_request))
        .with_state(state)
}

/// Helper function to check if an environment variable exists
pub fn has_env_var(name: &str) -> bool {
    std::env::var(name).is_ok()
}

/// Helper function to get an environment variable with a default value
pub fn get_env_var(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

/// Helper function to create a mock response for testing
#[cfg(feature = "mock")]
pub fn create_mock_response(tool: McpTool, request_id: Option<String>) -> McpResponse {
    let request_id = request_id.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        format!("mock-{}-{}", timestamp, rand::random::<u16>())
    });
    
    McpResponse {
        data: json!({
            "mock": true,
            "tool": tool.to_string(),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }),
        error: None,
        usage: Some(McpUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(20),
            total_tokens: Some(30),
            cost_usd: Some(0.0),
            latency_ms: Some(50),
        }),
        request_id,
    }
}

// Re-export modules for specific proxies
#[cfg(feature = "perplexity")]
pub mod perplexity_proxy;

#[cfg(feature = "claude")]
pub mod claude_proxy;

#[cfg(feature = "deepseek")]
pub mod deepseek_proxy;

#[cfg(feature = "qwen")]
pub mod qwen_proxy;

#[cfg(feature = "gemini")]
pub mod gemini_proxy;

#[cfg(feature = "llama")]
pub mod llama_proxy;
