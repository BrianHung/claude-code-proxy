import { z } from "zod/v4";

const ToolChoiceSchema = z.union([
  z.object({ type: z.literal("auto") }),
  z.object({ type: z.literal("any") }),
  z.object({
    type: z.literal("tool"),
    name: z.string(),
  }),
]);

const ToolSchema = z.union([
  // Function tools with input_schema
  z.object({
    name: z.string().min(1).max(128),
    description: z.string().optional(),
    input_schema: z.object({
      type: z.literal("object"),
      properties: z.record(z.string(), z.any()).optional(),
      required: z.array(z.string()).optional(),
    }),
    cache_control: z.object({ type: z.literal("ephemeral") }).optional(),
  }),
  // System tools (computer, bash, text_editor, web_search) without input_schema
  z.object({
    name: z.string().min(1).max(128),
    type: z.string().optional(), // e.g., "computer_20250124", "bash_20241022", etc.
    display_width_px: z.number().optional(),
    display_height_px: z.number().optional(),
    display_number: z.number().optional(),
    max_uses: z.number().optional(),
    allowed_domains: z.array(z.string()).optional(),
    blocked_domains: z.array(z.string()).optional(),
    user_location: z
      .object({
        type: z.literal("approximate"),
        city: z.string().optional(),
        region: z.string().optional(),
        country: z.string().optional(),
        timezone: z.string().optional(),
      })
      .optional(),
  }),
]);

const TextContentBlockSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
  cache_control: z.object({ type: z.literal("ephemeral") }).optional(),
});

const ImageContentBlockSchema = z.object({
  type: z.literal("image"),
  source: z.object({
    type: z.literal("base64"),
    media_type: z.enum(["image/jpeg", "image/png", "image/gif", "image/webp"]),
    data: z.string(), // base64 encoded image data
  }),
  cache_control: z.object({ type: z.literal("ephemeral") }).optional(),
});

const ToolUseContentBlockSchema = z.object({
  type: z.literal("tool_use"),
  id: z.string(),
  name: z.string(),
  input: z.record(z.string(), z.any()),
});

const ToolResultContentBlockSchema = z.object({
  type: z.literal("tool_result"),
  tool_use_id: z.string(),
  content: z.union([z.string(), z.array(z.any())]),
  is_error: z.boolean().optional(),
});

const ContentBlockSchema = z.union([
  TextContentBlockSchema,
  ImageContentBlockSchema,
  ToolUseContentBlockSchema,
  ToolResultContentBlockSchema,
]);

const MessageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.union([
    z.string(), // Shorthand for [{type: "text", text: string}]
    z.array(ContentBlockSchema), // Array of content blocks
  ]),
});

const MetadataSchema = z
  .object({
    user_id: z.string().max(256).optional(),
  })
  .optional();

export const AnthropicApiRequestSchema = z.object({
  model: z.string().min(1).max(256),
  messages: z.array(MessageSchema).min(1).max(100000), // Up to 100,000 messages per spec
  max_tokens: z.number().int().min(1),
  system: z
    .union([z.string(), z.array(z.record(z.string(), z.any()))])
    .optional(),
  temperature: z.number().min(0).max(1).default(1.0).optional(),
  top_p: z.number().min(0).max(1).optional(),
  top_k: z.number().int().min(1).optional(), // ≥1 per specification
  stop_sequences: z.array(z.string()).optional(),
  stream: z.boolean().optional(),
  tools: z.array(ToolSchema).optional(),
  tool_choice: ToolChoiceSchema.optional(),
  metadata: MetadataSchema,
  service_tier: z.enum(["auto", "standard_only"]).optional(),
  extra_headers: z.record(z.string(), z.string()).optional(),
  thinking: z
    .object({
      type: z.literal("enabled"),
      budget_tokens: z.number().int().min(1024),
    })
    .optional(),
});

const ResponseTextContentBlockSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
});

const ResponseToolUseContentBlockSchema = z.object({
  type: z.literal("tool_use"),
  id: z.string(),
  name: z.string(),
  input: z.record(z.string(), z.any()),
});

const ResponseContentBlockSchema = z.union([
  ResponseTextContentBlockSchema,
  ResponseToolUseContentBlockSchema,
]);

const UsageSchema = z.object({
  input_tokens: z.number().int().min(0),
  output_tokens: z.number().int().min(0),
  cache_creation_input_tokens: z.number().int().min(0).optional(),
  cache_read_input_tokens: z.number().int().min(0).optional(),
});

const StopReasonSchema = z
  .enum(["end_turn", "max_tokens", "stop_sequence", "tool_use"])
  .nullable();

/**
 * Complete Anthropic API response schema
 * @see https://docs.anthropic.com/claude/reference/messages_post
 */
export const AnthropicResponseSchema = z.object({
  id: z.string().min(1),
  type: z.literal("message"),
  role: z.literal("assistant"),
  content: z.array(ResponseContentBlockSchema),
  model: z.string().min(1),
  stop_reason: StopReasonSchema,
  stop_sequence: z.string().nullable(),
  usage: UsageSchema,
});

export type AnthropicApiRequest = z.infer<typeof AnthropicApiRequestSchema>;
export type AnthropicMessage = z.infer<typeof MessageSchema>;
export type AnthropicContentBlock = z.infer<typeof ContentBlockSchema>;
export type AnthropicResponse = z.infer<typeof AnthropicResponseSchema>;
