import { Hono } from "hono";
import { cors } from "hono/cors";
import { v4 as uuid } from "uuid";
import { anthropic } from "@ai-sdk/anthropic";
import { openai } from "@ai-sdk/openai";
import { google } from "@ai-sdk/google";
import { xai } from "@ai-sdk/xai";
import {
  streamText,
  generateText,
  createProviderRegistry,
  type LanguageModel,
  type CallSettings,
  type Prompt,
  ToolSet,
  ToolChoice,
  wrapLanguageModel,
  ModelMessage,
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

import type {
  LanguageModelV2Middleware,
  LanguageModelV2Prompt,
} from "@ai-sdk/provider";

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

const anthropicSystemToolCalls = new Set(["WebSearch"]);
const promptHasSystemToolCall = (prompt: LanguageModelV2Prompt) =>
  prompt.some(
    (message) =>
      message.role === "assistant" &&
      message.content.some(
        (part) =>
          part.type === "tool-call" &&
          anthropicSystemToolCalls.has(part.toolName)
      )
  );

/**
 * Check if the last message contains an anthropic system tool call.
 */
function lastMessageSystemTool(messages: ModelMessage[]) {
  const last = messages[messages.length - 1];
  console.log("messages", messages);
  return false;
}

/**
 * Routes requests to Anthropic if the prompt contains a system tool call.
 * @param env
 * @returns
 */
const anthropicSystemToolCall = (env: Bindings): LanguageModelV2Middleware => ({
  wrapStream: async function (props) {
    // TODO: If messages contain an anthropic system tool call, and model is not anthropic
    // route request to an anthropic model instead.
    const ANTHROPIC_API_KEY = env.ANTHROPIC_API_KEY;
    const prompt = props.params.prompt;

    if (promptHasSystemToolCall(prompt) && ANTHROPIC_API_KEY) {
      // Call doStream with original model id.
    }

    return props.doStream();
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
    const registry = createProviderRegistry({
      anthropic,
      openai,
      google,
      xai,
    });

    const modelId = resolveModelId(request.model, c.env);

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

    const messages = toModelMessages(request, availableTools);
    const model = wrapLanguageModel({
      model: registry.languageModel(modelId as any),
      middleware: [anthropicSystemToolCall(c.env)],
      // Override model id if last message contains a system tool call.
      modelId: lastMessageSystemTool(messages)
        ? `anthropic:${request.model}`
        : modelId,
    });

    // Build options object for AI SDK with proper typing
    const options: TextOptions = {
      model,
      messages,
      maxOutputTokens: request.max_tokens,
      temperature: request.temperature,
      topP: request.top_p,
      topK: request.top_k,
      stopSequences: request.stop_sequences,
      tools: toTools(request.tools as AnthropicTool[]),
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
        `Internal server error: ${errorMessage}. Check your request format and try again. Enable DEBUG=1 for more details.`
      ),
      500
    );
  }
});

export default app;
