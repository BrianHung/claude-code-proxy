import { describe, it, expect } from "vitest";
import {
  toModelMessages,
  toAnthropicResponse,
  toTools,
  isAnthropicMessageFormat,
  normalizeToAnthropicMessages,
} from "../src/convert";
import type { AnthropicTool } from "../src/types";
import {
  AnthropicApiRequestSchema,
  AnthropicResponseSchema,
  type AnthropicResponse,
  type AnthropicApiRequest,
} from "../src/schemas";

describe("Anthropic ↔ AI SDK Conversion", () => {
  describe("toModelMessages", () => {
    it("should convert simple text messages", () => {
      const anthropicRequest: AnthropicApiRequest = {
        model: "claude-3-5-haiku-20241022",
        messages: [
          { role: "user", content: "Hello!" },
          { role: "assistant", content: "Hi there!" },
          { role: "user", content: "How are you?" },
        ],
        max_tokens: 100,
      };

      const coreMessages = toModelMessages(anthropicRequest);

      expect(coreMessages).toHaveLength(3);
      expect(coreMessages[0].role).toBe("user");
      expect(coreMessages[0].content).toEqual([
        { type: "text", text: "Hello!" },
      ]);
      expect(coreMessages[1].role).toBe("assistant");
      expect(coreMessages[1].content).toEqual([
        { type: "text", text: "Hi there!" },
      ]);
    });

    it("should convert system prompts", () => {
      const anthropicRequest: AnthropicApiRequest = {
        model: "claude-3-5-sonnet-20241022",
        system: "You are a helpful assistant.",
        messages: [{ role: "user", content: "What is 2+2?" }],
        max_tokens: 50,
      };

      const coreMessages = toModelMessages(anthropicRequest);

      expect(coreMessages).toHaveLength(2);
      expect(coreMessages[0].role).toBe("system");
      expect(coreMessages[0].content).toBe("You are a helpful assistant.");
      expect(coreMessages[1].role).toBe("user");
    });

    it("should convert structured content with images", () => {
      const anthropicRequest: AnthropicApiRequest = {
        model: "claude-3-5-haiku-20241022",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What do you see?" },
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: "image/png",
                  data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                },
              },
            ],
          },
        ],
        max_tokens: 100,
      };

      const coreMessages = toModelMessages(anthropicRequest);

      expect(coreMessages).toHaveLength(1);
      expect(coreMessages[0].role).toBe("user");
      expect(coreMessages[0].content).toHaveLength(2);
      expect(coreMessages[0].content[0]).toEqual({
        type: "text",
        text: "What do you see?",
      });
      expect(coreMessages[0].content[1]).toEqual({
        type: "image",
        image:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
      });
    });

    it("should handle tool results", () => {
      const anthropicRequest: AnthropicApiRequest = {
        model: "claude-3-5-sonnet-20241022",
        messages: [
          { role: "user", content: "Calculate 15 * 23" },
          {
            role: "assistant",
            content: [
              { type: "text", text: "I'll calculate that for you." },
              {
                type: "tool_use",
                id: "toolu_123",
                name: "calculator",
                input: { operation: "multiply", a: 15, b: 23 },
              },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu_123",
                content: "345",
              },
            ],
          },
        ],
        max_tokens: 100,
      };

      const coreMessages = toModelMessages(anthropicRequest);

      expect(coreMessages).toHaveLength(3); // user + assistant + tool (user message empty, so skipped)
      expect(coreMessages[2].role).toBe("tool");
      expect(coreMessages[2].content[0]).toMatchObject({
        type: "tool-result",
        toolCallId: "toolu_123",
        result: "345",
      });
    });
  });

  describe("toAnthropicResponse", () => {
    it("should convert simple text response", () => {
      const coreOutput = {
        text: "Hello! How can I help you today?",
        usage: {
          inputTokens: 10,
          outputTokens: 8,
        },
        finishReason: "stop",
      };

      const anthropicResponse = toAnthropicResponse(
        coreOutput,
        "claude-3-5-haiku-20241022",
        "msg_test_123"
      );

      expect(anthropicResponse.id).toBe("msg_test_123");
      expect(anthropicResponse.type).toBe("message");
      expect(anthropicResponse.role).toBe("assistant");
      expect(anthropicResponse.model).toBe("claude-3-5-haiku-20241022");
      expect(anthropicResponse.stop_reason).toBe("end_turn");
      expect(anthropicResponse.content).toHaveLength(1);
      expect(anthropicResponse.content[0]).toEqual({
        type: "text",
        text: "Hello! How can I help you today?",
      });
      expect(anthropicResponse.usage).toEqual({
        input_tokens: 10,
        output_tokens: 8,
      });
    });

    it("should convert tool calls", () => {
      const coreOutput = {
        text: "I'll help you calculate that.",
        toolCalls: [
          {
            id: "call_123",
            name: "calculator",
            args: { operation: "add", a: 5, b: 3 },
          },
        ],
        usage: {
          inputTokens: 15,
          outputTokens: 12,
        },
        finishReason: "tool-calls",
      };

      const anthropicResponse = toAnthropicResponse(
        coreOutput,
        "claude-3-5-sonnet-20241022"
      );

      expect(anthropicResponse.content).toHaveLength(2);
      expect(anthropicResponse.content[0]).toEqual({
        type: "text",
        text: "I'll help you calculate that.",
      });
      expect(anthropicResponse.content[1]).toEqual({
        type: "tool_use",
        id: "call_123",
        name: "calculator",
        input: { operation: "add", a: 5, b: 3 },
      });
      expect(anthropicResponse.stop_reason).toBe("tool_use");
    });

    it("should handle different finish reasons", () => {
      const testCases = [
        { input: "stop", expected: "end_turn" },
        { input: "length", expected: "max_tokens" },
        { input: "tool-calls", expected: "tool_use" },
        { input: "content-filter", expected: "end_turn" },
        { input: "stop-sequence", expected: "stop_sequence" },
        { input: "unknown", expected: "end_turn" },
      ];

      testCases.forEach(({ input, expected }) => {
        const response = toAnthropicResponse(
          { text: "Test", finishReason: input },
          "claude-3-5-haiku-20241022"
        );
        expect(response.stop_reason).toBe(expected);
      });
    });

    it("should handle cache usage", () => {
      const coreOutput = {
        text: "Response with cache usage",
        usage: {
          inputTokens: 50,
          outputTokens: 25,
          cacheCreationInputTokens: 10,
          cacheReadInputTokens: 5,
        },
        finishReason: "stop",
      };

      const anthropicResponse = toAnthropicResponse(
        coreOutput,
        "claude-3-5-haiku-20241022"
      );

      expect(anthropicResponse.usage).toEqual({
        input_tokens: 50,
        output_tokens: 25,
        cache_creation_input_tokens: 10,
        cache_read_input_tokens: 5,
      });
    });
  });

  describe("isAnthropicMessageFormat", () => {
    it("should identify valid Anthropic messages", () => {
      const validMessages = [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
      ];

      expect(isAnthropicMessageFormat(validMessages)).toBe(true);
    });

    it("should identify invalid messages with system role", () => {
      const invalidMessages = [
        { role: "system", content: "You are helpful" },
        { role: "user", content: "Hello" },
      ];

      expect(isAnthropicMessageFormat(invalidMessages)).toBe(false);
    });

    it("should identify invalid messages with missing fields", () => {
      const invalidMessages = [
        { role: "user" }, // missing content
        { content: "Hello" }, // missing role
      ];

      expect(isAnthropicMessageFormat(invalidMessages)).toBe(false);
    });

    it("should handle structured content", () => {
      const validMessages = [
        {
          role: "user",
          content: [
            { type: "text", text: "Hello" },
            {
              type: "image",
              source: { type: "base64", media_type: "image/png", data: "abc" },
            },
          ],
        },
      ];

      expect(isAnthropicMessageFormat(validMessages)).toBe(true);
    });

    it("should normalize messages to Anthropic format", () => {
      const mixedMessages = [
        { role: "system", content: "You are helpful" },
        { role: "user", content: "Hello" },
        { role: "assistant", content: { text: "Hi!" } },
      ];

      const normalized = normalizeToAnthropicMessages(mixedMessages);

      expect(normalized).toEqual([
        { role: "user", content: "You are helpful" }, // system -> user
        { role: "user", content: "Hello" },
        { role: "assistant", content: '{"text":"Hi!"}' }, // object -> string
      ]);
    });
  });

  describe("Edge Cases", () => {
    it("should handle empty content arrays", () => {
      const anthropicRequest: AnthropicApiRequest = {
        model: "claude-3-5-haiku-20241022",
        messages: [{ role: "user", content: [] as any }],
        max_tokens: 10,
      };

      const coreMessages = toModelMessages(anthropicRequest);

      // Should not add user message if content is empty
      expect(coreMessages).toHaveLength(0);
    });

    it("should handle missing usage information", () => {
      const response = toAnthropicResponse(
        { text: "Hello" },
        "claude-3-5-haiku-20241022"
      );

      expect(response.usage).toEqual({
        input_tokens: 0,
        output_tokens: 0,
      });
    });

    it("should generate IDs when not provided", () => {
      const response = toAnthropicResponse(
        { text: "Hello" },
        "claude-3-5-haiku-20241022"
      );

      expect(response.id).toMatch(/^msg_\d+_[a-z0-9]{8}$/);
    });
  });

  describe("toTools", () => {
    it("should convert valid Anthropic tools to AI SDK format", () => {
      const anthropicTools = [
        {
          name: "calculator",
          description: "Perform basic arithmetic operations",
          input_schema: {
            type: "object" as const,
            properties: {
              operation: {
                type: "string",
                enum: ["add", "subtract", "multiply", "divide"],
              },
              a: { type: "number" },
              b: { type: "number" },
            },
            required: ["operation", "a", "b"],
          },
          cache_control: undefined,
        },
        {
          name: "weather",
          description: "Get weather information",
          input_schema: {
            type: "object" as const,
            properties: {
              location: { type: "string" },
            },
            required: ["location"],
          },
          cache_control: undefined,
        },
      ] as AnthropicTool[];

      const aiSdkTools = toTools(anthropicTools);

      expect(aiSdkTools).toHaveLength(2);

      expect(aiSdkTools[0]).toEqual({
        type: "function",
        function: {
          name: "calculator",
          description: "Perform basic arithmetic operations",
          parameters: {
            type: "object",
            properties: {
              operation: {
                type: "string",
                enum: ["add", "subtract", "multiply", "divide"],
              },
              a: { type: "number" },
              b: { type: "number" },
            },
            required: ["operation", "a", "b"],
          },
        },
      });

      expect(aiSdkTools[1]).toEqual({
        type: "function",
        function: {
          name: "weather",
          description: "Get weather information",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string" },
            },
            required: ["location"],
          },
        },
      });
    });

    it("should handle tools with missing description", () => {
      const anthropicTools = [
        {
          name: "simple_tool",
          description: undefined,
          input_schema: {
            type: "object" as const,
            properties: {
              param1: { type: "string" },
            },
          },
          cache_control: undefined,
        },
      ] as AnthropicTool[];

      const aiSdkTools = toTools(anthropicTools);

      expect(aiSdkTools[0].function.description).toBe("");
    });

    it("should fix malformed schema types", () => {
      const anthropicTools = [
        {
          name: "broken_tool",
          description: "Tool with incorrect schema type",
          input_schema: {
            type: "None" as any, // This should be fixed to 'object'
            properties: {
              param1: { type: "string" },
            },
            required: ["param1"],
          },
        },
      ] as AnthropicTool[];

      const aiSdkTools = toTools(anthropicTools);

      expect(aiSdkTools[0].function.parameters.type).toBe("object");
    });

    it("should handle tools with missing input_schema", () => {
      const anthropicTools = [
        {
          name: "minimal_tool",
          description: "Tool without input schema",
          input_schema: {
            type: "object" as const,
          },
          cache_control: undefined,
        },
      ] as AnthropicTool[];

      const aiSdkTools = toTools(anthropicTools);

      expect(aiSdkTools[0].function.parameters).toEqual({
        type: "object",
        properties: {},
        required: [],
      });
    });

    it("should convert all valid tools including system tools", () => {
      const mixedTools = [
        // Standard function tool
        {
          name: "calculator",
          description: "A calculator tool",
          input_schema: {
            type: "object",
            properties: { a: { type: "number" } },
            required: ["a"],
          },
          cache_control: undefined,
        },
        // Computer use tool
        {
          name: "computer",
          type: "computer_20250124",
          display_width_px: 1920,
          display_height_px: 1080,
          display_number: 1,
        },
        // Text editor tool
        {
          name: "text_editor",
          type: "text_editor_20250124",
        },
        // Invalid tools that should be filtered out
        null,
        { description: "Missing name" },
        { name: "", description: "Empty name" },
      ] as AnthropicTool[];

      const result = toTools(mixedTools);

      expect(result).toHaveLength(3); // 3 valid tools converted

      // Check function tool
      expect(result[0].function.name).toBe("calculator");
      expect(result[0].function.parameters.properties).toEqual({
        a: { type: "number" },
      });
      expect(result[0].function.parameters.required).toEqual(["a"]);

      // Check computer tool
      expect(result[1].function.name).toBe("computer");
      expect(result[1].function.description).toBe(
        "Anthropic computer_20250124 tool"
      );
      expect(result[1].function.parameters.properties).toEqual({});

      // Check text editor tool
      expect(result[2].function.name).toBe("text_editor");
      expect(result[2].function.description).toBe(
        "Anthropic text_editor_20250124 tool"
      );
      expect(result[2].function.parameters.properties).toEqual({});
    });

    it("should handle non-array input gracefully", () => {
      const result = toTools(null as any);
      expect(result).toEqual([]);

      const result2 = toTools("not an array" as any);
      expect(result2).toEqual([]);
    });
  });

  describe("Schema Validation", () => {
    it("should validate correct Anthropic responses", () => {
      const validResponse: AnthropicResponse = {
        id: "msg_123",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hello!" }],
        model: "claude-3-5-sonnet-20241022",
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: {
          input_tokens: 10,
          output_tokens: 5,
        },
      };

      expect(() => AnthropicResponseSchema.parse(validResponse)).not.toThrow();

      const validation = AnthropicResponseSchema.safeParse(validResponse);
      expect(validation.success).toBe(true);
      if (validation.success) {
        expect(validation.data).toEqual(validResponse);
      }
    });

    it("should reject invalid Anthropic responses", () => {
      const invalidResponses = [
        // Missing required fields
        { id: "msg_123" },
        // Wrong type
        {
          id: "msg_123",
          type: "completion",
          role: "assistant",
          content: [],
          model: "test",
          stop_reason: "end_turn",
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
        // Wrong role
        {
          id: "msg_123",
          type: "message",
          role: "user",
          content: [],
          model: "test",
          stop_reason: "end_turn",
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
        // Invalid stop_reason
        {
          id: "msg_123",
          type: "message",
          role: "assistant",
          content: [],
          model: "test",
          stop_reason: "invalid_reason",
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
        // Negative token counts
        {
          id: "msg_123",
          type: "message",
          role: "assistant",
          content: [],
          model: "test",
          stop_reason: "end_turn",
          stop_sequence: null,
          usage: { input_tokens: -1, output_tokens: 0 },
        },
      ];

      invalidResponses.forEach((invalidResponse) => {
        const validation = AnthropicResponseSchema.safeParse(invalidResponse);
        expect(validation.success).toBe(false);
      });
    });

    it("should validate requests correctly", () => {
      const validRequest = {
        model: "claude-3-5-sonnet-20241022",
        messages: [{ role: "user", content: "Hello!" }],
        max_tokens: 1000,
      };

      const validation = AnthropicApiRequestSchema.safeParse(validRequest);
      expect(validation.success).toBe(true);
    });
  });

  describe("End-to-End Conversion", () => {
    it("should handle complete request-response cycle", () => {
      const request: AnthropicApiRequest = {
        model: "claude-3-5-sonnet-20241022",
        messages: [{ role: "user", content: "Calculate 2+2" }],
        max_tokens: 100,
      };

      // Convert to core messages
      const coreMessages = toModelMessages(request);
      expect(coreMessages).toHaveLength(1);

      // Simulate AI SDK response
      const mockAiSdkOutput = {
        text: "The answer is 4.",
        usage: { inputTokens: 8, outputTokens: 5 },
        finishReason: "stop",
      };

      // Convert back to Anthropic format
      const response = toAnthropicResponse(
        mockAiSdkOutput,
        request.model
      );

      // Validate the response
      const validation = AnthropicResponseSchema.safeParse(response);
      expect(validation.success).toBe(true);

      if (validation.success) {
        expect(validation.data.content).toEqual([
          { type: "text", text: "The answer is 4." },
        ]);
        expect(validation.data.model).toBe("claude-3-5-sonnet-20241022");
        expect(validation.data.stop_reason).toBe("end_turn");
      }
    });
  });
});
