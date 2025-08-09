import type {
  ModelMessage,
  TextPart,
  ImagePart,
  ToolCallPart,
  ToolResultPart,
  Tool,
} from "ai";
import { tool, jsonSchema } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { v4 as uuid } from "uuid";

import type {
  AnthropicResponse,
  AnthropicMessage,
  AnthropicContentBlock,
  AnthropicApiRequest,
} from "./schemas";

import type {
  AnthropicTool,
  AnthropicModelId,
  AnthropicToolChoice,
} from "./anthropic-types";
import { Bindings } from "./types";

export {
  type AnthropicResponse,
  type AnthropicMessage,
  type AnthropicContentBlock,
} from "./schemas";

/**
 * Generate Anthropic-style message ID
 * Format: msg_{timestamp}_{8 random lowercase alphanumeric chars}
 */
function generateAnthropicId(): string {
  const timestamp = Date.now();
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  const randomSuffix = Array.from({ length: 8 }, () =>
    chars.charAt(Math.floor(Math.random() * chars.length))
  ).join("");
  return `msg_${timestamp}_${randomSuffix}`;
}

const VALID_ANTHROPIC_ROLES = ["user", "assistant"] as const;

const FINISH_REASON_MAP = {
  stop: "end_turn",
  length: "max_tokens",
  "tool-calls": "tool_use",
  "content-filter": "end_turn",
  "stop-sequence": "stop_sequence",
} as const;

const DEFAULT_FINISH_REASON = "end_turn" as const;

/**
 * Map finish reasons from AI SDK to official Anthropic format
 */
function mapFinishReason(reason?: string): AnthropicResponse["stop_reason"] {
  if (!reason) return DEFAULT_FINISH_REASON;
  return (FINISH_REASON_MAP as any)[reason] || DEFAULT_FINISH_REASON;
}

/**
 * Normalize content to array format
 */
function normalizeContentToArray<T>(content: string | T[]): T[] {
  return typeof content === "string"
    ? [{ type: "text" as const, text: content } as T]
    : content;
}

/**
 * Extract system content from Anthropic system field
 */
function toSystemContent(system: AnthropicApiRequest["system"]): string {
  if (typeof system === "string") {
    return system;
  }
  return (
    system
      ?.map((block) => (block.type === "text" ? block.text : ""))
      .join("\n") || ""
  );
}

/**
 * Process user content blocks and return both user content and any tool messages
 */
function processUserContentBlocks(
  content: AnthropicContentBlock[],
  currentToolContext: Map<string, string>,
  availableTools?: Set<string>
): {
  userContent: (TextPart | ImagePart)[];
  toolMessages: ModelMessage[];
} {
  const userContent: (TextPart | ImagePart)[] = [];
  const toolMessages: ModelMessage[] = [];

  for (const block of content) {
    if (block.type === "text") {
      userContent.push({ type: "text", text: block.text });
    } else if (block.type === "image" && block.source.type === "base64") {
      // For AI SDK v5, use ImagePart with proper format
      userContent.push({
        type: "image",
        image: `data:${block.source.media_type};base64,${block.source.data}`,
      });
    } else if (block.type === "tool_result") {
      // Look up the tool name from the current assistant's tool context
      const toolName =
        currentToolContext.get(block.tool_use_id) || "unknown_tool";

      // Skip tool results for tools that aren't available in current provider
      if (availableTools && !availableTools.has(toolName)) {
        console.warn(
          `[CONVERT] Skipping tool result for unavailable tool: ${toolName} (tool_use_id: ${block.tool_use_id})`
        );
        continue;
      }

      // Extract tool results to separate tool message
      const toolContent =
        typeof block.content === "string"
          ? block.content
          : JSON.stringify(block.content);

      toolMessages.push({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: block.tool_use_id,
            toolName: toolName,
            result: toolContent, // Keep for backward compatibility
            output: block.is_error
              ? { type: "error-text", value: toolContent }
              : { type: "text", value: toolContent },
            ...(block.is_error && { isError: true }),
          } as ToolResultPart,
        ],
      });
    }
  }

  return { userContent, toolMessages };
}

/**
 * Process assistant content blocks
 */
function processAssistantContentBlocks(
  content: AnthropicContentBlock[],
  availableTools?: Set<string>
): (TextPart | ToolCallPart)[] {
  const assistantContent: (TextPart | ToolCallPart)[] = [];

  for (const block of content) {
    if (block.type === "text") {
      assistantContent.push({ type: "text", text: block.text });
    } else if (block.type === "tool_use") {
      // Skip tool calls for tools that aren't available in current provider
      if (availableTools && !availableTools.has(block.name)) {
        console.warn(
          `[CONVERT] Skipping tool call for unavailable tool: ${block.name}`
        );
        continue;
      }

      const toolCallPart: ToolCallPart = {
        type: "tool-call",
        toolCallId: block.id,
        toolName: block.name,
        input: block.input || {}, // Ensure input is never undefined
      };

      assistantContent.push(toolCallPart);
    }
  }

  return assistantContent;
}

const TOOL_CHOICE_MAP = {
  auto: "auto",
  any: "required",
  tool: (name: string) => ({ type: "tool", toolName: name }),
} as const;

/**
 * Convert Anthropic tool choice to AI SDK format using standardized mappings
 */
export function toToolChoice(toolChoice?: AnthropicToolChoice): any {
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

/**
 * Extract tool use context from assistant content blocks
 */
function extractToolContext(
  content: AnthropicContentBlock[]
): Map<string, string> {
  const toolContext = new Map<string, string>();
  for (const block of content) {
    if (block.type === "tool_use") {
      toolContext.set(block.id, block.name);
    }
  }
  return toolContext;
}

/**
 * Build usage object with optional cache fields
 */
function buildUsageObject(usage?: {
  inputTokens?: number;
  outputTokens?: number;
  cacheCreationInputTokens?: number;
  cacheReadInputTokens?: number;
}): AnthropicResponse["usage"] {
  const result: AnthropicResponse["usage"] = {
    input_tokens: usage?.inputTokens || 0,
    output_tokens: usage?.outputTokens || 0,
  };

  // Add cache fields if provided by the AI SDK
  if (usage?.cacheCreationInputTokens !== undefined) {
    result.cache_creation_input_tokens = usage.cacheCreationInputTokens;
  }
  if (usage?.cacheReadInputTokens !== undefined) {
    result.cache_read_input_tokens = usage.cacheReadInputTokens;
  }

  return result;
}

// Main conversion functions

/**
 * Convert Anthropic request messages to AI SDK ModelMessage format (v5)
 * Handles system prompts, text, images, tool uses, and tool results
 * Uses single-pass approach leveraging the alternating user-assistant pattern
 */
export function toModelMessages(
  request: AnthropicApiRequest,
  availableTools?: Set<string>
): ModelMessage[] {
  const modelMessages: ModelMessage[] = [];

  // Track the current assistant's tool context for resolving tool_result names
  let currentToolContext = new Map<string, string>();

  // Add system message if present
  if (request.system) {
    const systemContent = toSystemContent(request.system);
    modelMessages.push({
      role: "system",
      content: systemContent,
    });
  }

  // Single pass: process messages in order, maintaining tool context
  for (const msg of request.messages.values()) {
    if (msg.role === "user") {
      const content = normalizeContentToArray<AnthropicContentBlock>(
        msg.content
      );

      const { userContent, toolMessages } = processUserContentBlocks(
        content,
        currentToolContext,
        availableTools
      );

      // Add any tool messages first
      modelMessages.push(...toolMessages);

      // Only add user message if it has content
      if (userContent.length > 0) {
        modelMessages.push({ role: "user", content: userContent });
      }
    } else if (msg.role === "assistant") {
      const content = normalizeContentToArray<AnthropicContentBlock>(
        msg.content
      );

      // Update tool context for the next user message's tool_result blocks
      currentToolContext = extractToolContext(content);

      const assistantContent = processAssistantContentBlocks(
        content,
        availableTools
      );

      // Only add assistant message if it has content after filtering
      if (assistantContent.length > 0) {
        modelMessages.push({ role: "assistant", content: assistantContent });
      } else {
        console.warn(
          `[CONVERT] Skipping empty assistant message after tool filtering`
        );
      }
    }
  }

  return modelMessages;
}

/**
 * Convert AI SDK output back to Anthropic response format
 * Handles text, tool calls, usage, and stop reasons
 */
export function toAnthropicResponse(
  coreOutput: {
    text?: string;
    toolCalls?: Array<ToolCallPart>;
    usage?: {
      inputTokens?: number;
      outputTokens?: number;
      cacheCreationInputTokens?: number;
      cacheReadInputTokens?: number;
    };
    finishReason?: string;
  },
  model: string,
  requestId?: string
): AnthropicResponse {
  const content: AnthropicResponse["content"] = [];

  // Add text content if present
  if (coreOutput.text) {
    content.push({ type: "text", text: coreOutput.text });
  }

  // Add tool calls if present
  if (coreOutput.toolCalls && coreOutput.toolCalls.length > 0) {
    for (const toolCall of coreOutput.toolCalls) {
      content.push({
        type: "tool_use",
        id:
          (toolCall as any).toolCallId ||
          (toolCall as any).id ||
          "toolu_" + uuid(),
        name: (toolCall as any).toolName || (toolCall as any).name,
        input: ((toolCall as any).input || (toolCall as any).args) as Record<
          string,
          any
        >,
      });
    }
  }

  const usage = buildUsageObject(coreOutput.usage);

  return {
    id: requestId || generateAnthropicId(),
    type: "message",
    role: "assistant",
    content,
    model,
    stop_reason: mapFinishReason(coreOutput.finishReason),
    stop_sequence: null,
    usage,
  };
}

/**
 * Helper function to validate if a message array is in Anthropic format
 */
export function isAnthropicMessageFormat(
  messages: any[]
): messages is AnthropicMessage[] {
  return messages.every(
    (msg) =>
      msg.role &&
      VALID_ANTHROPIC_ROLES.includes(msg.role) &&
      (typeof msg.content === "string" || Array.isArray(msg.content))
  );
}

/**
 * Helper function to convert simple messages to Anthropic format
 */
export function normalizeToAnthropicMessages(
  messages: any[]
): AnthropicMessage[] {
  return messages.map((msg) => ({
    role: msg.role === "system" ? "user" : msg.role, // Convert system to user
    content:
      typeof msg.content === "string"
        ? msg.content
        : Array.isArray(msg.content)
        ? msg.content
        : JSON.stringify(msg.content),
  }));
}

const SERVER_TOOLS = new Set(["web_search"]);

const TYPE_TO_TOOL = {
  web_search_20250305: anthropic.tools.webSearch_20250305,
  bash_20241022: anthropic.tools.bash_20241022,
  bash_20250124: anthropic.tools.bash_20250124,
  computer_20241022: anthropic.tools.computer_20241022,
  computer_20250124: anthropic.tools.computer_20250124,
  text_editor_20241022: anthropic.tools.textEditor_20241022,
  text_editor_20250429: anthropic.tools.textEditor_20250429,
  text_editor_20250124: anthropic.tools.textEditor_20250124,
} as const;

/**
 * Convert Anthropic tools to AI SDK tools
 */
export function toTools(tools: AnthropicTool[] = []): {
  client: Record<string, Tool>;
  server: Record<string, Tool>;
  all: Record<string, Tool>;
} {
  const client: Record<string, Tool> = {};
  const server: Record<string, Tool> = {};
  tools.forEach((t) => {
    if ("input_schema" in t) {
      client[t.name] = tool({
        description: t.description || "",
        inputSchema: jsonSchema(t.input_schema),
      });
    } else {
      // Anthropic system tools
      const toolSet = SERVER_TOOLS.has(t.name) ? server : client;
      toolSet[t.name] = TYPE_TO_TOOL[t.type](t as any);
    }
  });
  return {
    client,
    server,
    get all() {
      return {
        ...client,
        ...server,
      };
    },
  };
}

/**
 * Resolve model shortcuts and redirect to configured providers
 * Allows routing Claude model requests to any AI provider (OpenAI, Google, etc.)
 * via environment variable configuration for cost optimization or availability
 */
export function resolveModelId(
  modelOverrides: Bindings,
  modelId: AnthropicModelId
): string {
  const overrides: Array<[string, string | undefined]> = [
    ["haiku", modelOverrides.HAIKU_MODEL_ID],
    ["sonnet", modelOverrides.SONNET_MODEL_ID],
    ["opus", modelOverrides.OPUS_MODEL_ID],
  ];

  for (const [keyword, override] of overrides) {
    if (modelId.includes(keyword) && override) {
      return override;
    }
  }

  // Fallback: use the original Anthropic model via the Vercel AI SDK
  return `anthropic:${modelId}`;
}

/**
 * Creates a stream transformer that converts AI SDK stream parts to Anthropic SSE events
 */
export function toAnthropicStream(model: string) {
  const encoder = new TextEncoder();
  const messageId = uuid();

  let contentIndex = 0;
  let isContentOpen = false;
  let currentToolId: string | null = null;
  let currentWebSearchToolId: string | null = null;
  let finishReason: string = "stop";
  let outputTokens = 0;
  let textBlockStarted = false;
  let textBlockClosed = false;
  let hasTextContent = false;

  return new TransformStream({
    start(controller: TransformStreamDefaultController<Uint8Array>) {
      // Send message_start with complete Anthropic-compatible usage
      controller.enqueue(
        encoder.encode(
          `event: message_start\ndata: ${JSON.stringify({
            type: "message_start",
            message: {
              id: messageId,
              type: "message",
              role: "assistant",
              content: [],
              model: model,
              stop_reason: null,
              stop_sequence: null,
              usage: {
                input_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: 0,
              },
            },
          })}\n\n`
        )
      );

      // Send initial content block start for text
      controller.enqueue(
        encoder.encode(
          `event: content_block_start\ndata: ${JSON.stringify({
            type: "content_block_start",
            index: 0,
            content_block: { type: "text", text: "" },
          })}\n\n`
        )
      );

      // Send ping event for connection keep-alive (like Anthropic)
      controller.enqueue(
        encoder.encode(
          `event: ping\ndata: ${JSON.stringify({
            type: "ping",
          })}\n\n`
        )
      );

      textBlockStarted = true;
    },

    transform(part, controller: TransformStreamDefaultController<Uint8Array>) {
      switch (part.type) {
        case "text-start":
          hasTextContent = true;
          break;

        case "text-delta":
          hasTextContent = true;

          // Ensure text block is open
          if (!textBlockStarted || textBlockClosed) {
            if (textBlockClosed) {
              // Need to start a new text block
              contentIndex++;
              controller.enqueue(
                encoder.encode(
                  `event: content_block_start\ndata: ${JSON.stringify({
                    type: "content_block_start",
                    index: contentIndex,
                    content_block: { type: "text", text: "" },
                  })}\n\n`
                )
              );
              textBlockStarted = true;
              textBlockClosed = false;
            }
          }

          controller.enqueue(
            encoder.encode(
              `event: content_block_delta\ndata: ${JSON.stringify({
                type: "content_block_delta",
                index: textBlockClosed ? contentIndex : 0,
                delta: { type: "text_delta", text: part.text },
              })}\n\n`
            )
          );
          break;

        case "text-end":
          break;

        case "reasoning-start":
          hasTextContent = true;
          break;

        case "reasoning-delta":
          hasTextContent = true;

          // Handle reasoning content (for models that support it)
          if (!textBlockStarted || textBlockClosed) {
            if (textBlockClosed) {
              contentIndex++;
              controller.enqueue(
                encoder.encode(
                  `event: content_block_start\ndata: ${JSON.stringify({
                    type: "content_block_start",
                    index: contentIndex,
                    content_block: { type: "text", text: "" },
                  })}\n\n`
                )
              );
              textBlockStarted = true;
              textBlockClosed = false;
            }
          }

          controller.enqueue(
            encoder.encode(
              `event: content_block_delta\ndata: ${JSON.stringify({
                type: "content_block_delta",
                index: textBlockClosed ? contentIndex : 0,
                delta: { type: "text_delta", text: part.text },
              })}\n\n`
            )
          );
          break;

        case "reasoning-end":
          break;

        case "tool-input-start":
          const toolInputId = part.id;

          // Close text block if open
          if (textBlockStarted && !textBlockClosed) {
            controller.enqueue(
              encoder.encode(
                `event: content_block_stop\ndata: ${JSON.stringify({
                  type: "content_block_stop",
                  index: hasTextContent
                    ? textBlockClosed
                      ? contentIndex
                      : 0
                    : 0,
                })}\n\n`
              )
            );
            textBlockClosed = true;
          }

          contentIndex++;
          controller.enqueue(
            encoder.encode(
              `event: content_block_start\ndata: ${JSON.stringify({
                type: "content_block_start",
                index: contentIndex,
                content_block: {
                  type: "tool_use",
                  id: toolInputId,
                  name: part.toolName,
                  input: {},
                },
              })}\n\n`
            )
          );
          isContentOpen = true;
          currentToolId = toolInputId;
          break;

        case "tool-input-delta":
          if (isContentOpen && part.id === currentToolId) {
            controller.enqueue(
              encoder.encode(
                `event: content_block_delta\ndata: ${JSON.stringify({
                  type: "content_block_delta",
                  index: contentIndex,
                  delta: {
                    type: "input_json_delta",
                    partial_json: part.delta,
                  },
                })}\n\n`
              )
            );
          }
          break;

        case "tool-input-end":
          // Tool input is complete - close the tool block
          if (isContentOpen) {
            controller.enqueue(
              encoder.encode(
                `event: content_block_stop\ndata: ${JSON.stringify({
                  type: "content_block_stop",
                  index: contentIndex,
                })}\n\n`
              )
            );
            isContentOpen = false;
          }
          break;

        case "tool-call":
          // Complete tool call (non-streaming)
          const toolCallId = part.toolCallId;

          // Close text block if open
          if (textBlockStarted && !textBlockClosed) {
            controller.enqueue(
              encoder.encode(
                `event: content_block_stop\ndata: ${JSON.stringify({
                  type: "content_block_stop",
                  index: hasTextContent
                    ? textBlockClosed
                      ? contentIndex
                      : 0
                    : 0,
                })}\n\n`
              )
            );
            textBlockClosed = true;
          }

          contentIndex++;

          // Track web search tool calls for source event association
          if (part.toolName === "web_search") {
            currentWebSearchToolId = toolCallId;
          }

          controller.enqueue(
            encoder.encode(
              `event: content_block_start\ndata: ${JSON.stringify({
                type: "content_block_start",
                index: contentIndex,
                content_block: {
                  type: "tool_use",
                  id: toolCallId,
                  name: part.toolName,
                  input: part.input,
                },
              })}\n\n`
            )
          );

          controller.enqueue(
            encoder.encode(
              `event: content_block_stop\ndata: ${JSON.stringify({
                type: "content_block_stop",
                index: contentIndex,
              })}\n\n`
            )
          );
          isContentOpen = false;
          break;

        case "tool-result":
          // Tool execution result - typically not streamed in Claude format
          break;

        case "tool-error":
          // Tool execution error
          console.error("Tool error:", part);
          break;

        case "start":
          // Stream is starting
          break;

        case "start-step":
          break;

        case "finish":
          finishReason = part.finishReason;
          outputTokens = part.totalUsage?.outputTokens ?? 0;
          break;

        case "finish-step":
          // Step is finished - capture usage
          if (part.usage) {
            outputTokens = part.usage.outputTokens ?? outputTokens;
          }
          break;

        case "error":
          console.error("Stream error:", part.error);
          controller.error(new Error(`Stream error: ${part.error}`));
          break;

        case "abort":
          console.warn("Stream aborted");
          break;

        case "source":
          // Handle web search source results from Vercel AI SDK
          if (part.sourceType === "url") {
            // Use current web search tool ID or generate one if not available
            const toolUseId = currentWebSearchToolId || `web_search_${uuid()}`;

            // Close text block if open to make room for web search result
            if (textBlockStarted && !textBlockClosed) {
              controller.enqueue(
                encoder.encode(
                  `event: content_block_stop\ndata: ${JSON.stringify({
                    type: "content_block_stop",
                    index: hasTextContent
                      ? textBlockClosed
                        ? contentIndex
                        : 0
                      : 0,
                  })}\n\n`
                )
              );
              textBlockClosed = true;
            }

            contentIndex++;

            // Extract encrypted content from provider metadata if available
            const anthropicMetadata = part.providerMetadata?.anthropic;
            const encryptedContent =
              anthropicMetadata?.encrypted_content ||
              anthropicMetadata?.content ||
              part.url; // Fallback to URL
            const pageAge = anthropicMetadata?.page_age || null;

            // Create web search tool result content block
            controller.enqueue(
              encoder.encode(
                `event: content_block_start\ndata: ${JSON.stringify({
                  type: "content_block_start",
                  index: contentIndex,
                  content_block: {
                    type: "web_search_tool_result",
                    tool_use_id: toolUseId,
                    content: [
                      {
                        type: "web_search_result",
                        url: part.url,
                        title: part.title,
                        page_age: pageAge,
                        encrypted_content: encryptedContent,
                      },
                    ],
                  },
                })}\n\n`
              )
            );

            controller.enqueue(
              encoder.encode(
                `event: content_block_stop\ndata: ${JSON.stringify({
                  type: "content_block_stop",
                  index: contentIndex,
                })}\n\n`
              )
            );
          } else {
            console.warn(
              "Received source event with unsupported sourceType:",
              part.sourceType,
              part
            );
          }
          break;

        default:
          // Log unknown event types for debugging
          console.warn("Unknown event type:", part.type, part);
          break;
      }
    },

    flush(controller: TransformStreamDefaultController<Uint8Array>) {
      // Close any remaining open blocks
      if (isContentOpen) {
        controller.enqueue(
          encoder.encode(
            `event: content_block_stop\ndata: ${JSON.stringify({
              type: "content_block_stop",
              index: contentIndex,
            })}\n\n`
          )
        );
      }

      // Close text block if still open
      if (textBlockStarted && !textBlockClosed) {
        controller.enqueue(
          encoder.encode(
            `event: content_block_stop\ndata: ${JSON.stringify({
              type: "content_block_stop",
              index: hasTextContent ? (textBlockClosed ? contentIndex : 0) : 0,
            })}\n\n`
          )
        );
      }

      // Complete stop reason mapping like Python version
      const stopReasonMap: Record<string, string> = {
        stop: "end_turn",
        length: "max_tokens",
        "content-filter": "content_filter",
        "tool-calls": "tool_use",
        error: "error",
        other: "stop_sequence",
        unknown: "end_turn",
      };

      // Send message_delta with complete usage
      controller.enqueue(
        encoder.encode(
          `event: message_delta\ndata: ${JSON.stringify({
            type: "message_delta",
            delta: {
              stop_reason: stopReasonMap[finishReason] || "end_turn",
              stop_sequence: null,
            },
            usage: { output_tokens: outputTokens },
          })}\n\n`
        )
      );

      controller.enqueue(
        encoder.encode(
          `event: message_stop\ndata: ${JSON.stringify({
            type: "message_stop",
          })}\n\n`
        )
      );

      controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
    },
  });
}
