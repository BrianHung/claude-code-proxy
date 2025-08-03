import { Hono } from "hono";
import { cors } from "hono/cors";
import { v4 as uuid } from "uuid";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createXai } from "@ai-sdk/xai";
import {
  streamText,
  generateText,
  createProviderRegistry,
  type LanguageModel,
  ModelMessage,
  Tool,
} from "ai";
import { AnthropicApiRequestSchema, AnthropicResponseSchema } from "./schemas";
import {
  convertAnthropicToCoreMessages,
  convertCoreToAnthropicResponse,
  convertAnthropicTool,
  resolveModelId,
  toAnthropicStream,
} from "./convert";
import type {
  AnthropicToolChoice,
  AnthropicModelId,
  AnthropicTool,
} from "./types";

const TOOL_CHOICE_MAP = {
  auto: "auto",
  any: "required",
  tool: (name: string) => ({ type: "tool", toolName: name }),
} as const;

const CORS_CONFIG = {
  origin: "*",
  allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowHeaders: [
    "Content-Type",
    "Authorization",
    "X-API-Key",
    "Anthropic-Version",
  ],
};

const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  Connection: "keep-alive",
} as const;

type Bindings = {
  DEBUG?: string;
  ANTHROPIC_API_KEY: string;
  OPENAI_API_KEY?: string;
  GOOGLE_GENERATIVE_AI_API_KEY?: string;
  XAI_API_KEY?: string;
  HAIKU_MODEL_ID?: string;
  SONNET_MODEL_ID?: string;
  OPUS_MODEL_ID?: string;
};

/**
 * Convert Anthropic tool choice to AI SDK format using standardized mappings
 */
function convertAnthropicToolToCoreTool(toolChoice?: AnthropicToolChoice): any {
  if (!toolChoice) return undefined;
  switch (toolChoice.type) {
    case "auto":
      return TOOL_CHOICE_MAP.auto;
    case "any":
      return TOOL_CHOICE_MAP.any;
    case "tool":
      return toolChoice.name
        ? TOOL_CHOICE_MAP.tool(toolChoice.name)
        : undefined;
    default:
      return undefined;
  }
}

// Interface for AI SDK options using the types they DO export
interface AiSdkOptions {
  model: LanguageModel;
  messages: ModelMessage[];
  system?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
  tools?: Record<string, Tool>;
  toolChoice?: any;
}

// ========== UTILITY FUNCTIONS ==========

/**
 * Debug logging function
 */
function debug(env: Bindings, ...args: any[]) {
  if (env.DEBUG === "1" || env.DEBUG === "true") {
    console.log("[DEBUG]", ...args);
  }
}

const app = new Hono<{ Bindings: Bindings }>();

// CORS middleware
app.use("*", cors(CORS_CONFIG));

// Health check endpoint
app.get("/", (c) => {
  return c.json({
    name: "Claude Code Proxy",
    version: "1.0.0",
  });
});

// Create provider registry for clean model resolution
function createRegistry(env: Bindings) {
  return createProviderRegistry({
    // Create providers properly with API keys
    anthropic: createAnthropic({
      apiKey: env.ANTHROPIC_API_KEY || "placeholder",
    }),
    openai: createOpenAI({
      apiKey: env.OPENAI_API_KEY || "placeholder",
      fetch: (...args) => {
        console.log("🚀 OPENAI FETCH", args);
        return fetch(...args);
      },
    }),
    google: createGoogleGenerativeAI({
      apiKey: env.GOOGLE_GENERATIVE_AI_API_KEY || "placeholder",
    }),
    xai: createXai({
      apiKey: env.XAI_API_KEY || "placeholder",
    }),
  });
}





interface ErrorResponse {
  type: "error";
  error: {
    type: string;
    message: string;
    details?: string | any[];
  };
}

/**
 * Create Anthropic-compatible error response
 */
function createErrorResponse(
  type: string,
  message: string,
  details?: string | any[]
): ErrorResponse {
  return {
    type: "error",
    error: {
      type,
      message,
      ...(details && { details }),
    },
  };
}

// Claude-compatible messages endpoint
app.post("/v1/messages", async (c) => {
  try {
    const json = await c.req.json();

    const validation = AnthropicApiRequestSchema.safeParse(json);
    if (!validation.success) {
      return c.json(
        createErrorResponse(
          "invalid_request_error",
          "Invalid request format. Ensure your request matches the Anthropic Claude API specification",
          validation.error.issues
        ),
        400
      );
    }

    const request = validation.data;
    const registry = createRegistry(c.env);
    const modelId = resolveModelId(request.model, c.env);
    const model = registry.languageModel(modelId as any);

    // Determine which tools are available for this provider
    const isAnthropicProvider = modelId.startsWith("anthropic:");
    let availableTools: Set<string> | undefined;
    if (
      request.tools &&
      Array.isArray(request.tools) &&
      request.tools.length > 0
    ) {
      availableTools = new Set(
        request.tools
          .filter((tool) => {
            // Include function tools for all providers
            if ("input_schema" in tool) return true;
            // Include system tools only for Anthropic providers
            return isAnthropicProvider;
          })
          .map((tool) => tool.name)
      );
    }

    // Convert messages with available tool information
    const coreMessages = convertAnthropicToCoreMessages(
      request,
      availableTools
    );
    debug(c.env, "Core messages:", coreMessages);

    // Build options object for AI SDK with proper typing
    const aiSdkOptions: AiSdkOptions = {
      model,
      messages: coreMessages,
      maxTokens: request.max_tokens,
      temperature: request.temperature,
      topP: request.top_p,
      topK: request.top_k,
      stopSequences: request.stop_sequences,
      tools: convertAnthropicTool(request.tools as AnthropicTool[]),
      toolChoice: convertAnthropicToolToCoreTool(request.tool_choice),
    };

    // Handle system prompt
    if (request.system && typeof request.system === "string") {
      aiSdkOptions.system = request.system;
    }

    if (request.stream) {
      const result = streamText(aiSdkOptions);
      const stream = result.fullStream.pipeThrough(
        toAnthropicStream(request.model)
      );
      return new Response(stream, {
        headers: SSE_HEADERS,
      });
    } else {
      const result = await generateText(aiSdkOptions);
      const response = convertCoreToAnthropicResponse(
        {
          text: result.text,
          toolCalls: result.toolCalls,
          usage: result.usage,
          finishReason: result.finishReason,
        },
        request.model,
        uuid()
      );

      const validation = AnthropicResponseSchema.safeParse(response);
      if (!validation.success) {
        return c.json(
          createErrorResponse(
            "api_error",
            "Generated response does not match Anthropic specification"
          ),
          500
        );
      }

      return c.json(validation.data);
    }
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    return c.json(
      createErrorResponse(
        "api_error",
        `Internal server error: ${errorMessage}. Check your request format and try again. Enable DEBUG=1 for more details.`
      ),
      500
    );
  }
});

export default app;
