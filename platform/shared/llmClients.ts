import axios, { AxiosError, AxiosRequestConfig } from 'axios';
import { ModelSpec } from './modelRouter';

/**
 * Standardized response format for all LLM clients
 */
export interface LLMResponse {
  text: string;
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
  };
  modelUsed?: string;
}

/**
 * Base request parameters for all LLM calls
 */
export interface LLMRequestParams {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stopSequences?: string[];
}

/**
 * Error thrown when an API key is missing
 */
export class MissingAPIKeyError extends Error {
  constructor(provider: string, envVar: string) {
    super(`Missing API key for ${provider}. Please set the ${envVar} environment variable.`);
    this.name = 'MissingAPIKeyError';
  }
}

/**
 * Error thrown when an LLM API call fails
 */
export class LLMAPIError extends Error {
  status?: number;
  
  constructor(message: string, status?: number) {
    super(message);
    this.name = 'LLMAPIError';
    this.status = status;
  }
}

/**
 * Handles errors from axios requests to LLM APIs
 */
function handleAxiosError(error: unknown, provider: string): never {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;
    const status = axiosError.response?.status;
    const message = axiosError.response?.data 
      ? JSON.stringify(axiosError.response.data)
      : axiosError.message;
    
    throw new LLMAPIError(`${provider} API error: ${message}`, status);
  }
  
  if (error instanceof Error) {
    throw new LLMAPIError(`${provider} error: ${error.message}`);
  }
  
  throw new LLMAPIError(`Unknown error with ${provider} API`);
}

/**
 * Calls the OpenAI API with the provided parameters
 * 
 * @param params Request parameters including prompt and optional settings
 * @param modelOverride Optional model override that replaces the default GPT-4.1
 * @returns A promise resolving to the standardized LLM response
 * @throws {MissingAPIKeyError} If the OPENAI_API_KEY environment variable is not set
 * @throws {LLMAPIError} If the API call fails
 */
export async function callOpenAI(
  params: LLMRequestParams,
  modelOverride?: ModelSpec
): Promise<LLMResponse> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new MissingAPIKeyError('OpenAI', 'OPENAI_API_KEY');
  }

  const model = modelOverride?.model || 'gpt-4-1106-preview';
  const apiVersion = modelOverride?.apiVersion || '2023-05-15';
  
  try {
    const response = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model,
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: params.prompt }
        ],
        max_tokens: params.maxTokens || 2048,
        temperature: params.temperature || 0.7,
        top_p: params.topP || 1,
        stop: params.stopSequences
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
          'OpenAI-Beta': `api-version=${apiVersion}`
        }
      }
    );

    return {
      text: response.data.choices[0].message.content,
      usage: {
        promptTokens: response.data.usage?.prompt_tokens,
        completionTokens: response.data.usage?.completion_tokens,
        totalTokens: response.data.usage?.total_tokens
      },
      modelUsed: model
    };
  } catch (error) {
    handleAxiosError(error, 'OpenAI');
  }
}

/**
 * Calls the Anthropic Claude API with the provided parameters
 * 
 * @param params Request parameters including prompt and optional settings
 * @param modelOverride Optional model override that replaces the default Claude 3 Sonnet
 * @returns A promise resolving to the standardized LLM response
 * @throws {MissingAPIKeyError} If the ANTHROPIC_API_KEY environment variable is not set
 * @throws {LLMAPIError} If the API call fails
 */
export async function callClaudeSonnet(
  params: LLMRequestParams,
  modelOverride?: ModelSpec
): Promise<LLMResponse> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new MissingAPIKeyError('Anthropic Claude', 'ANTHROPIC_API_KEY');
  }

  const model = modelOverride?.model || 'claude-3-sonnet-20240229';
  
  try {
    const response = await axios.post(
      'https://api.anthropic.com/v1/messages',
      {
        model,
        messages: [
          { role: 'user', content: params.prompt }
        ],
        max_tokens: params.maxTokens || 4096,
        temperature: params.temperature || 0.7,
        top_p: params.topP || 1,
        stop_sequences: params.stopSequences
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01'
        }
      }
    );

    return {
      text: response.data.content[0].text,
      usage: {
        promptTokens: response.data.usage?.input_tokens,
        completionTokens: response.data.usage?.output_tokens,
        totalTokens: response.data.usage?.input_tokens + response.data.usage?.output_tokens
      },
      modelUsed: model
    };
  } catch (error) {
    handleAxiosError(error, 'Claude');
  }
}

/**
 * Calls the Google Vertex AI Gemini API with the provided parameters
 * 
 * @param params Request parameters including prompt and optional settings
 * @param modelOverride Optional model override that replaces the default Gemini Flash
 * @returns A promise resolving to the standardized LLM response
 * @throws {MissingAPIKeyError} If the GOOGLE_VERTEX_API_KEY environment variable is not set
 * @throws {LLMAPIError} If the API call fails
 */
export async function callGeminiFlash(
  params: LLMRequestParams,
  modelOverride?: ModelSpec
): Promise<LLMResponse> {
  const apiKey = process.env.GOOGLE_VERTEX_API_KEY;
  if (!apiKey) {
    throw new MissingAPIKeyError('Google Vertex AI', 'GOOGLE_VERTEX_API_KEY');
  }

  const model = modelOverride?.model || 'gemini-1.5-flash';
  
  try {
    const response = await axios.post(
      `https://generativelanguage.googleapis.com/v1/models/${model}:generateContent`,
      {
        contents: [
          { role: 'user', parts: [{ text: params.prompt }] }
        ],
        generationConfig: {
          maxOutputTokens: params.maxTokens || 2048,
          temperature: params.temperature || 0.7,
          topP: params.topP || 1,
          stopSequences: params.stopSequences
        }
      },
      {
        headers: {
          'Content-Type': 'application/json'
        },
        params: {
          key: apiKey
        }
      }
    );

    return {
      text: response.data.candidates[0].content.parts[0].text,
      usage: response.data.usageMetadata ? {
        promptTokens: response.data.usageMetadata.promptTokenCount,
        completionTokens: response.data.usageMetadata.candidatesTokenCount,
        totalTokens: response.data.usageMetadata.totalTokenCount
      } : undefined,
      modelUsed: model
    };
  } catch (error) {
    handleAxiosError(error, 'Gemini');
  }
}

/**
 * Calls the Perplexity API with the provided parameters
 * 
 * @param params Request parameters including prompt/query and optional settings
 * @param modelOverride Optional model override that replaces the default model
 * @returns A promise resolving to the standardized LLM response
 * @throws {MissingAPIKeyError} If the PERPLEXITY_API_KEY environment variable is not set
 * @throws {LLMAPIError} If the API call fails
 */
export async function callPerplexity(
  params: LLMRequestParams,
  modelOverride?: ModelSpec
): Promise<LLMResponse> {
  const apiKey = process.env.PERPLEXITY_API_KEY;
  if (!apiKey) {
    throw new MissingAPIKeyError('Perplexity', 'PERPLEXITY_API_KEY');
  }

  const model = modelOverride?.model || 'pplx-7b-online';
  
  try {
    const response = await axios.post(
      'https://api.perplexity.ai/chat/completions',
      {
        model,
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: params.prompt }
        ],
        max_tokens: params.maxTokens || 1024,
        temperature: params.temperature || 0.7,
        top_p: params.topP || 1,
        stop: params.stopSequences
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        }
      }
    );

    return {
      text: response.data.choices[0].message.content,
      usage: {
        promptTokens: response.data.usage?.prompt_tokens,
        completionTokens: response.data.usage?.completion_tokens,
        totalTokens: response.data.usage?.total_tokens
      },
      modelUsed: model
    };
  } catch (error) {
    handleAxiosError(error, 'Perplexity');
  }
}

/**
 * Calls the Deepseek API with the provided parameters
 * 
 * @param params Request parameters including prompt and optional settings
 * @param modelOverride Optional model override that replaces the default Deepseek Coder
 * @returns A promise resolving to the standardized LLM response
 * @throws {MissingAPIKeyError} If the DEEPSEEK_API_KEY environment variable is not set
 * @throws {LLMAPIError} If the API call fails
 */
export async function callDeepseekR1(
  params: LLMRequestParams,
  modelOverride?: ModelSpec
): Promise<LLMResponse> {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new MissingAPIKeyError('Deepseek', 'DEEPSEEK_API_KEY');
  }

  const model = modelOverride?.model || 'deepseek-coder-v1';
  
  try {
    const response = await axios.post(
      'https://api.deepseek.com/v1/chat/completions',
      {
        model,
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: params.prompt }
        ],
        max_tokens: params.maxTokens || 2048,
        temperature: params.temperature || 0.7,
        top_p: params.topP || 1,
        stop: params.stopSequences
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        }
      }
    );

    return {
      text: response.data.choices[0].message.content,
      usage: {
        promptTokens: response.data.usage?.prompt_tokens,
        completionTokens: response.data.usage?.completion_tokens,
        totalTokens: response.data.usage?.total_tokens
      },
      modelUsed: model
    };
  } catch (error) {
    handleAxiosError(error, 'Deepseek');
  }
}

/**
 * Calls the Qwen API with the provided parameters
 * 
 * @param params Request parameters including prompt and optional settings
 * @param modelOverride Optional model override that replaces the default Qwen 2.5
 * @returns A promise resolving to the standardized LLM response
 * @throws {MissingAPIKeyError} If the QWEN_API_KEY environment variable is not set
 * @throws {LLMAPIError} If the API call fails
 */
export async function callQwen25(
  params: LLMRequestParams,
  modelOverride?: ModelSpec
): Promise<LLMResponse> {
  const apiKey = process.env.QWEN_API_KEY;
  if (!apiKey) {
    throw new MissingAPIKeyError('Qwen', 'QWEN_API_KEY');
  }

  const model = modelOverride?.model || 'qwen2.5-72b-instruct';
  
  try {
    const response = await axios.post(
      'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
      {
        model,
        input: {
          messages: [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: params.prompt }
          ]
        },
        parameters: {
          max_tokens: params.maxTokens || 1024,
          temperature: params.temperature || 0.7,
          top_p: params.topP || 1,
          stop: params.stopSequences
        }
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        }
      }
    );

    return {
      text: response.data.output.choices[0].message.content,
      usage: {
        promptTokens: response.data.usage?.input_tokens,
        completionTokens: response.data.usage?.output_tokens,
        totalTokens: response.data.usage?.total_tokens
      },
      modelUsed: model
    };
  } catch (error) {
    handleAxiosError(error, 'Qwen');
  }
}

/**
 * Calls the local Llama model via MCP proxy with the provided parameters
 * No API key is required as this is an internal service
 * 
 * @param params Request parameters including prompt and optional settings
 * @param modelOverride Optional model override that replaces the default Llama 3 8B
 * @returns A promise resolving to the standardized LLM response
 * @throws {LLMAPIError} If the API call fails or the Llama service is unavailable
 */
export async function callLlama(
  params: LLMRequestParams,
  modelOverride?: ModelSpec
): Promise<LLMResponse> {
  const model = modelOverride?.model || 'llama-3-8b-instruct';
  const mcpEndpoint = process.env.LLAMA_MCP_ENDPOINT || 'http://mcp-llama.prowzi-system.svc.cluster.local:8080';
  
  try {
    const response = await axios.post(
      `${mcpEndpoint}/generate`,
      {
        model,
        prompt: params.prompt,
        max_tokens: params.maxTokens || 2048,
        temperature: params.temperature || 0.7,
        top_p: params.topP || 1,
        stop: params.stopSequences
      },
      {
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );

    return {
      text: response.data.text,
      usage: response.data.usage,
      modelUsed: model
    };
  } catch (error) {
    handleAxiosError(error, 'Llama');
  }
}

/**
 * Generic function to call any model based on ModelSpec
 * 
 * @param model The model specification from modelRouter
 * @param params Request parameters including prompt and optional settings
 * @returns A promise resolving to the standardized LLM response
 * @throws {Error} If the model provider is not supported
 */
export async function call<T extends ModelSpec>(
  model: T,
  params: LLMRequestParams
): Promise<LLMResponse> {
  switch (model.provider) {
    case 'openai':
      return callOpenAI(params, model);
    case 'anthropic':
      return callClaudeSonnet(params, model);
    case 'google':
      return callGeminiFlash(params, model);
    case 'perplexity':
      return callPerplexity(params, model);
    case 'deepseek':
      return callDeepseekR1(params, model);
    case 'qwen':
      return callQwen25(params, model);
    case 'llama-local':
      return callLlama(params, model);
    default:
      throw new Error(`Unsupported model provider: ${model.provider}`);
  }
}
