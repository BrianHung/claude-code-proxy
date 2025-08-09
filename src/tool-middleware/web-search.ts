import { webSearch_20250305ArgsSchema as WebSearchSchema } from "../tool/web-search_20250305";
import { z } from "zod/v4";

export type AnthropicWebSearchArgs = z.infer<typeof WebSearchSchema>;

import { openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";
import { google } from "@ai-sdk/google";
import { Tool } from "ai";

import {
  LanguageModelV2Middleware,
  SharedV2ProviderOptions as ProviderOptions,
  LanguageModelV2StreamPart,
  LanguageModelV2ProviderDefinedTool,
} from "@ai-sdk/provider";

import type { MiddlewareOptions } from "../types";

const WEB_SEARCH_TOOL_NAME = "web_search";

type WebSearchTool =
  | {
      type: "tool";
      tool: (args: AnthropicWebSearchArgs) => Tool;
      toolName: string; // Need to map and re-map tool name.
    }
  | { type: "provider"; providerOptions: ProviderOptions };

const providerFromModelId = (modelId: string) => modelId.split(":").at(0) ?? "";

/**
 * Middleware to use model's web search tool if available.
 * Otherwise, re-routes request to Anthropic model if Anthropic API key is available.
 */
export function webSearchMiddleware(
  options: MiddlewareOptions
): LanguageModelV2Middleware {
  const env = options.env;
  const provider = providerFromModelId(options.overrideModelId);
  const providerTool = webSearch[provider];

  let useAnthropic = false;
  const anthropicModel = options.providers.languageModel(
    options.modelId as any
  );

  return {
    transformParams: async ({ params }) => {
      const tools = params.tools || [];
      const webSearchTool = tools.find(
        (t): t is LanguageModelV2ProviderDefinedTool =>
          t.name === WEB_SEARCH_TOOL_NAME && t.type === "provider-defined"
      );

      if (webSearchTool && provider !== "anthropic") {
        const tool = providerTool;
        if (tool) {
          console.warn(
            `Routing web_search request to ${options.overrideModelId} from ${options.modelId}.`
          );
          if (tool.type === "tool") {
            params.tools = tools
              .filter((t) => t !== webSearchTool)
              .concat({
                ...tool.tool(webSearchTool.args as AnthropicWebSearchArgs),
                name: tool.toolName,
              } as LanguageModelV2ProviderDefinedTool);
          } else {
            params.providerOptions = {
              ...params.providerOptions,
              [provider]: {
                ...(params.providerOptions?.[provider] ?? {}),
                ...tool.providerOptions,
              },
            };
          }
        } else {
          // Route request to Anthropic model if Anthropic API key is available.
          if (env.ANTHROPIC_API_KEY) {
            useAnthropic = true;
            console.warn(
              `Routing web_search request to ${options.modelId} from ${options.overrideModelId}.`
            );
          }
        }
      }

      return params;
    },

    wrapGenerate: async ({ model, params }) => {
      const reqModel = useAnthropic ? anthropicModel : model;
      const result = await reqModel.doGenerate(params);

      let toolName = WEB_SEARCH_TOOL_NAME;
      if (providerTool && providerTool.type === "tool") {
        toolName = providerTool.toolName;
      }

      return {
        ...result,
        content: result.content.map((c) =>
          c.type === "tool-call" && c.toolName === toolName
            ? {
                ...c,
                toolName: WEB_SEARCH_TOOL_NAME,
              }
            : c
        ),
      };
    },

    wrapStream: async ({ model, params }) => {
      const reqModel = useAnthropic ? anthropicModel : model;
      const result = await reqModel.doStream(params);

      let toolName = WEB_SEARCH_TOOL_NAME;
      if (providerTool && providerTool.type === "tool") {
        toolName = providerTool.toolName;
      }

      return {
        ...result,
        stream: result.stream.pipeThrough(
          mapToolPart({
            [toolName]: {
              "tool-input-start": {
                toolName: WEB_SEARCH_TOOL_NAME,
              },
              "tool-result": {
                toolName: WEB_SEARCH_TOOL_NAME,
              },
              "tool-call": {
                toolName: WEB_SEARCH_TOOL_NAME,
                input: JSON.stringify({ query: "" }),
              },
            },
          })
        ),
      };
    },
  };
}

/**
 * WebSearch is implemented as a tool or provider options depending on the model.
 * This is a mapping of provider to the tool or provider options.
 */
const webSearch: Record<string, WebSearchTool> = {
  openai: {
    type: "tool",
    toolName: "web_search_preview",
    tool: (args) =>
      openai.tools.webSearchPreview({
        userLocation: args.userLocation,
      }),
  },
  anthropic: {
    type: "tool",
    toolName: "web_search",
    tool: anthropic.tools.webSearch_20250305,
  },
  google: {
    type: "tool",
    toolName: "google_search",
    tool: () =>
      google.tools.googleSearch({
        mode: "MODE_DYNAMIC",
        dynamicThreshold: 0.5,
      }),
  },
  xai: {
    type: "provider",
    providerOptions: {
      searchParameters: {
        mode: "auto",
        sources: ["web"],
        returnCitations: true,
      },
    },
  },
};

type StreamPartByType = {
  [K in LanguageModelV2StreamPart as K["type"]]: K;
};

const TOOL_PART_TYPES = [
  "tool-input-start",
  "tool-call",
  "tool-result",
] as const;

type ToolPartTypes = (typeof TOOL_PART_TYPES)[number];
const toolPartNames = new Set<string>(TOOL_PART_TYPES);

type ToolPartOverrides = {
  [T in ToolPartTypes]?: Partial<StreamPartByType[T]>;
};

type ToolPart = Extract<LanguageModelV2StreamPart, { type: ToolPartTypes }>;

const isToolPart = (part: LanguageModelV2StreamPart): part is ToolPart =>
  toolPartNames.has(part.type);

export function mapToolPart(toolNames: Record<string, ToolPartOverrides>) {
  return new TransformStream<
    LanguageModelV2StreamPart,
    LanguageModelV2StreamPart
  >({
    transform(part, controller) {
      if (isToolPart(part)) {
        controller.enqueue({
          ...part,
          ...toolNames[part.toolName]?.[part.type],
        } as LanguageModelV2StreamPart);
        return;
      }
      controller.enqueue(part);
    },
  });
}
