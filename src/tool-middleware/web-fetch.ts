import { JSONSchema7 } from "@ai-sdk/provider";

import {
  LanguageModelV2Middleware,
  LanguageModelV2FunctionTool,
} from "@ai-sdk/provider";
import type { MiddlewareOptions } from "../types";

const WEB_FETCH_TOOL_NAME = "WebFetch";

type WebFetchTool = {
  type: "function";
  inputSchema: JSONSchema7;
  toolName: string; // Need to map and re-map tool name.
};

const providerFromModelId = (modelId: string) => modelId.split(":").at(0) ?? "";

/**
 * Middleware to use model's web search tool if available.
 * Otherwise, re-routes request to Anthropic model if Anthropic API key is available.
 */
export function webFetchMiddleware(
  options: MiddlewareOptions
): LanguageModelV2Middleware {
  const provider = providerFromModelId(options.overrideModelId);
  const providerTool = webFetch[provider];

  return {
    transformParams: async ({ params }) => {
      const tools = params.tools || [];

      const webFetchTool = tools.find(
        (t): t is LanguageModelV2FunctionTool =>
          t.name === WEB_FETCH_TOOL_NAME && t.type === "function"
      );

      if (webFetchTool && provider !== "anthropic") {
        const tool = providerTool;
        if (tool) {
          console.warn(
            `Routing web_fetch request to ${options.overrideModelId} from ${options.modelId}.`
          );
          params.tools = tools
            .filter((t) => t !== webFetchTool)
            .concat({
              ...tool,
              name: tool.toolName,
            } as LanguageModelV2FunctionTool);
        }
      }

      return params;
    },
  };
}

const webFetch: Record<string, WebFetchTool> = {
  google: {
    type: "function",
    toolName: "WebFetch",
    inputSchema: {
      type: "object",
      properties: {
        url: {
          type: "string",
          description: "The URL to fetch content from",
        },
        prompt: {
          type: "string",
          description: "The prompt to run on the fetched content",
        },
      },
      required: ["url", "prompt"],
    },
  },
};
