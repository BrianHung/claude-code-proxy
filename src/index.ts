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
  type CallSettings,
  type Prompt,
  ToolSet,
  ToolChoice,
} from "ai";
import { AnthropicApiRequestSchema, AnthropicResponseSchema } from "./schemas";
import {
  toModelMessages,
  toAnthropicResponse,
  toTools,
  resolveModelId,
  toAnthropicStream,
  toToolChoice,
} from "./convert";
import type { AnthropicTool } from "./types";

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

/**
 * List of all supported providers and their configuration options:
 * https://ai-sdk.dev/providers/ai-sdk-providers
 */
const modelProviders = (env: Bindings) =>
  createProviderRegistry({
    anthropic: createAnthropic({
      apiKey: env.ANTHROPIC_API_KEY,
      baseURL: env.ANTHROPIC_BASE_URL,
    }),
    openai: createOpenAI({
      apiKey: env.OPENAI_API_KEY,
      baseURL: env.OPENAI_BASE_URL,
    }),
    google: createGoogleGenerativeAI({
      apiKey: env.GOOGLE_GENERATIVE_AI_API_KEY,
      baseURL: env.GOOGLE_GENERATIVE_AI_BASE_URL,
    }),
    xai: createXai({
      apiKey: env.XAI_API_KEY,
      baseURL: env.XAI_BASE_URL,
    }),
  });

type TextOptions = CallSettings &
  Prompt & {
    model: LanguageModel;
    tools?: ToolSet;
    toolChoice?: ToolChoice<ToolSet>;
  };

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

/**
 * Create Anthropic-compatible error response
 */
const createErrorResponse = (
  type: string,
  message: string,
  details?: string | any[]
) => ({
  type: "error",
  error: {
    type,
    message,
    ...(details && { details }),
  },
});

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
    const providers = modelProviders(c.env);

    const modelId = resolveModelId(c.env, request.model);

    const availableTools = new Set<string>(
      (request.tools ?? []).map((tool) => tool.name)
    );
    const messages = toModelMessages(request, availableTools);

    const tools = toTools(request.tools as AnthropicTool[]);
    const hasServerTool = Object.keys(tools.server).length > 0;
    const overrideModelId =
      hasServerTool && Boolean(c.env.ANTHROPIC_API_KEY)
        ? `anthropic:${request.model}`
        : modelId;

    if (hasServerTool) {
      if (overrideModelId !== modelId) {
        console.warn(
          `${modelId} does not support server tools. Routing request to Anthropic model ${overrideModelId} instead for: ${Object.keys(
            tools.server
          ).join(", ")}`
        );
      } else {
        console.warn(
          "Anthropic API key not configured. Server tools will not natively work.",
          Object.keys(tools.server).join(", ")
        );
      }
    }

    const model = providers.languageModel(overrideModelId as any);
    const options: TextOptions = {
      model,
      messages,
      maxOutputTokens: request.max_tokens,
      temperature: request.temperature,
      topP: request.top_p,
      topK: request.top_k,
      stopSequences: request.stop_sequences,
      tools: tools.all,
      toolChoice: toToolChoice(request.tool_choice),
    };

    // Handle system prompt
    if (request.system && typeof request.system === "string") {
      options.system = request.system;
    }

    if (request.stream) {
      const result = streamText(options);
      const stream = result.fullStream.pipeThrough(
        toAnthropicStream(request.model)
      );
      return new Response(stream, {
        headers: SSE_HEADERS,
      });
    } else {
      const result = await generateText(options);
      const response = toAnthropicResponse(
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
        `Internal Server Error: ${errorMessage}. Check browser or devtools for more details.`
      ),
      500
    );
  }
});

export default app;
