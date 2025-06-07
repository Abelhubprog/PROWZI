import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

/**
 * Defines the types of model operations
 */
export type ModelKind = 'search' | 'reasoning' | 'summarise';

/**
 * Defines the urgency bands for events
 */
export type Band = 'instant' | 'same-day' | 'weekly' | 'archive';

/**
 * Represents a model specification with all necessary information
 * to make an API call to the model provider
 */
export interface ModelSpec {
  provider: string;
  model: string;
  apiVersion?: string;
  contextWindow?: number;
  maxTokens?: number;
}

/**
 * Tenant-specific model overrides
 */
export interface TenantModelOverrides {
  search?: string;
  reasoning?: string;
  summarise?: string;
}

/**
 * Structure of the models.yaml configuration file
 */
interface ModelsConfig {
  default: {
    search: string;
    reasoning: string;
    summarise: string;
  };
  bands: {
    instant?: {
      search?: string;
      reasoning?: string;
      summarise?: string;
    };
    'same-day'?: {
      search?: string;
      reasoning?: string;
      summarise?: string;
    };
    weekly?: {
      search?: string;
      reasoning?: string;
      summarise?: string;
    };
    archive?: {
      search?: string;
      reasoning?: string;
      summarise?: string;
    };
  };
}

// Model provider mappings to full specifications
const MODEL_SPECS: Record<string, ModelSpec> = {
  'perplexity': {
    provider: 'perplexity',
    model: 'pplx-7b-online',
    contextWindow: 4096,
    maxTokens: 1024
  },
  'gpt-4.1': {
    provider: 'openai',
    model: 'gpt-4-1106-preview',
    apiVersion: '2023-05-15',
    contextWindow: 8192,
    maxTokens: 2048
  },
  'claude-4-sonnet': {
    provider: 'anthropic',
    model: 'claude-3-sonnet-20240229',
    contextWindow: 200000,
    maxTokens: 4096
  },
  'deepseek-r1': {
    provider: 'deepseek',
    model: 'deepseek-coder-v1',
    contextWindow: 16384,
    maxTokens: 2048
  },
  'qwen-2.5': {
    provider: 'qwen',
    model: 'qwen2.5-72b-instruct',
    contextWindow: 32768,
    maxTokens: 1024
  },
  'llama-3-8b': {
    provider: 'llama-local',
    model: 'llama-3-8b-instruct',
    contextWindow: 8192,
    maxTokens: 2048
  },
  'gemini-flash': {
    provider: 'google',
    model: 'gemini-1.5-flash',
    contextWindow: 16384,
    maxTokens: 2048
  }
};

// Cache for the loaded config to avoid repeated file reads
let configCache: ModelsConfig | null = null;

/**
 * Loads the models configuration from the YAML file
 * @returns The parsed configuration object
 */
function loadConfig(): ModelsConfig {
  if (configCache) {
    return configCache;
  }

  try {
    const configPath = path.resolve(process.cwd(), 'config', 'models.yaml');
    const fileContents = fs.readFileSync(configPath, 'utf8');
    configCache = yaml.load(fileContents) as ModelsConfig;
    return configCache;
  } catch (error) {
    console.error('Failed to load models configuration:', error);
    // Provide sensible defaults if config can't be loaded
    return {
      default: {
        search: 'perplexity',
        reasoning: 'gpt-4.1',
        summarise: 'llama-3-8b'
      },
      bands: {
        instant: {
          summarise: 'claude-4-sonnet',
          search: 'deepseek-r1'
        },
        'same-day': {
          summarise: 'qwen-2.5'
        },
        weekly: {
          summarise: 'llama-3-8b'
        }
      }
    };
  }
}

/**
 * Chooses the appropriate model based on the kind, band, impact, and tenant overrides
 * @param kind The type of model operation (search, reasoning, summarise)
 * @param band The urgency band (instant, same-day, weekly, archive)
 * @param impact Optional impact score (0-1) for fine-grained decision making
 * @param tenantOverrides Optional tenant-specific model overrides
 * @returns A ModelSpec object with the selected model information
 */
export function chooseModel(
  kind: ModelKind,
  band: Band,
  impact?: number,
  tenantOverrides?: TenantModelOverrides
): ModelSpec {
  // First check tenant overrides if provided
  if (tenantOverrides && tenantOverrides[kind]) {
    const modelName = tenantOverrides[kind];
    if (modelName && MODEL_SPECS[modelName]) {
      return MODEL_SPECS[modelName];
    }
  }

  const config = loadConfig();
  
  // Check for band-specific model
  const bandConfig = config.bands[band];
  if (bandConfig && bandConfig[kind]) {
    const modelName = bandConfig[kind];
    if (modelName && MODEL_SPECS[modelName]) {
      return MODEL_SPECS[modelName];
    }
  }
  
  // Impact-based model selection for certain scenarios
  if (impact !== undefined) {
    // For high-impact instant events, prefer Claude for summarization
    if (band === 'instant' && impact > 0.8 && kind === 'summarise') {
      return MODEL_SPECS['claude-4-sonnet'];
    }
    
    // For medium-impact same-day events, prefer Qwen
    if (band === 'same-day' && impact > 0.5 && kind === 'summarise') {
      return MODEL_SPECS['qwen-2.5'];
    }
  }
  
  // Fall back to default model for the kind
  const defaultModelName = config.default[kind];
  if (defaultModelName && MODEL_SPECS[defaultModelName]) {
    return MODEL_SPECS[defaultModelName];
  }
  
  // Ultimate fallback if everything else fails
  const fallbacks: Record<ModelKind, string> = {
    search: 'perplexity',
    reasoning: 'gpt-4.1',
    summarise: 'llama-3-8b'
  };
  
  return MODEL_SPECS[fallbacks[kind]];
}

/**
 * Resets the configuration cache, forcing a reload on the next call
 * Primarily used for testing or when configuration has been updated
 */
export function resetConfigCache(): void {
  configCache = null;
}
