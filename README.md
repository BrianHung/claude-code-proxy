# Claude Code Polyglot

A Cloudflare Worker proxy that converts Anthropic Claude API requests to Vercel AI SDK format, enabling Claude Code CLI to work with any AI provider supported by Vercel AI SDK.

## Quick Start

### Prerequisites

- Node.js 18+
- API key for your preferred provider (OpenAI, Anthropic, etc.)

### Local Development

```bash
git clone https://github.com/BrianHung/claude-code-proxy
cd claude-code-proxy
npm install
```

Configure environment variables in `wrangler.toml`:

```toml
[vars]
OPENAI_API_KEY = "your-openai-key"
ANTHROPIC_API_KEY = "your-anthropic-key"
GOOGLE_GENERATIVE_AI_API_KEY = "your-google-key"
XAI_API_KEY = "your-xai-key"

HAIKU_MODEL_ID = "openai:gpt-4o-mini"
SONNET_MODEL_ID = "openai:gpt-4o-mini"
OPUS_MODEL_ID = "openai:gpt-4o-mini"

DEBUG = "true"
```

Start the proxy:

```bash
npm run dev
```

Use with Claude CLI:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8787
claude "Hello world"
```

### Deploy to Cloudflare

```bash
npx wrangler login

# Set API keys
npx wrangler secret put OPENAI_API_KEY
npx wrangler secret put ANTHROPIC_API_KEY
npx wrangler secret put GOOGLE_GENERATIVE_AI_API_KEY
npx wrangler secret put XAI_API_KEY

# Set model overrides
npx wrangler secret put HAIKU_MODEL_ID
npx wrangler secret put SONNET_MODEL_ID
npx wrangler secret put OPUS_MODEL_ID

# Deploy
npm run deploy
```

## Model Configuration

The proxy maps Claude model requests to your configured models:

| Claude Model      | Maps To         | Environment Variable          |
| ----------------- | --------------- | ----------------------------- |
| claude-_-haiku-_  | HAIKU_MODEL_ID  | Default: `openai:gpt-4o-mini` |
| claude-_-sonnet-_ | SONNET_MODEL_ID | Default: `openai:gpt-4o-mini` |
| claude-_-opus-_   | OPUS_MODEL_ID   | Default: `openai:gpt-4o-mini` |

### Configuration Examples

**OpenAI models:**

```toml
HAIKU_MODEL_ID = "openai:gpt-4o-mini"
SONNET_MODEL_ID = "openai:gpt-4o-mini"
OPUS_MODEL_ID = "openai:gpt-4o-mini"
```

**Anthropic models:**

```toml
HAIKU_MODEL_ID = "anthropic:claude-3-5-haiku-20241022"
SONNET_MODEL_ID = "anthropic:claude-3-5-sonnet-20241022"
OPUS_MODEL_ID = "anthropic:claude-3-opus-20240229"
```

**Mixed providers:**

```toml
HAIKU_MODEL_ID = "openai:gpt-4o-mini"
SONNET_MODEL_ID = "anthropic:claude-3-5-sonnet-20241022"
OPUS_MODEL_ID = "openai:gpt-4o"
```

## API Usage

### Basic Request

```bash
curl -X POST http://localhost:8787/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-haiku-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8787/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "max_tokens": 100,
    "stream": true
  }'
```

## Limitations

**Unsupported Features:**

- Websearch and other Anthropic system tools are not supported at this time
- Computer use tools are not supported
- Some advanced Anthropic-specific features may not work correctly when proxied to other providers

## Commands

```bash
npm run dev        # Start local development server (port 8787)
npm run deploy     # Deploy to Cloudflare Workers
npm test          # Run test suite
npm run build     # Build the worker bundle
```

## License

MIT
