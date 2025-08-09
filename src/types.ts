import {
  type LanguageModel,
  type CallSettings,
  type Prompt,
  ToolSet,
  ToolChoice,
  ProviderRegistryProvider,
} from "ai";
import type { SharedV2ProviderOptions as ProviderOptions } from "@ai-sdk/provider";

export type Bindings = {
  DEBUG?: string;

  ANTHROPIC_API_KEY?: string;
  ANTHROPIC_BASE_URL?: string;

  OPENAI_API_KEY?: string;
  OPENAI_BASE_URL?: string;

  GOOGLE_GENERATIVE_AI_API_KEY?: string;
  GOOGLE_GENERATIVE_AI_BASE_URL?: string;

  XAI_API_KEY?: string;
  XAI_BASE_URL?: string;

  HAIKU_MODEL_ID?: string;
  SONNET_MODEL_ID?: string;
  OPUS_MODEL_ID?: string;
};

export type TextOptions = CallSettings &
  Prompt & {
    model: LanguageModel;
    tools?: ToolSet;
    toolChoice?: ToolChoice<ToolSet>;
    providerOptions?: ProviderOptions;
  };

export type MiddlewareOptions = {
  modelId: string;
  overrideModelId: string;
  providers: ProviderRegistryProvider;
  env: Bindings;
};
