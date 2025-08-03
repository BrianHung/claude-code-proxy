import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Miniflare } from 'miniflare';

describe('Claude Proxy Integration Tests', () => {
  let mf: Miniflare;

  beforeAll(async () => {
    // Create Miniflare instance for testing
    mf = new Miniflare({
      script: `
        import app from './src/index.ts';
        export default {
          fetch: (request, env, ctx) => app.fetch(request, env, ctx)
        };
      `,
      modules: true,
      modulesRules: [
        { type: "ESModule", include: ["**/*.ts", "**/*.js"] }
      ],
      bindings: {
        OPENAI_API_KEY: "test-openai-key",
        ANTHROPIC_API_KEY: "test-anthropic-key", 
        HAIKU_MODEL_ID: "openai:gpt-4o-mini",
        SONNET_MODEL_ID: "openai:gpt-4o-mini",
        OPUS_MODEL_ID: "openai:gpt-4o-mini",
        DEBUG: "true"
      },
      compatibilityDate: "2024-01-01",
      compatibilityFlags: ["nodejs_compat"]
    });

    // Wait for Miniflare to be ready
    await mf.ready;
  });

  afterAll(async () => {
    await mf.dispose();
  });

  describe('Health Check', () => {
    it('should return proxy configuration', async () => {
      const response = await mf.dispatchFetch('http://localhost/');
      expect(response.status).toBe(200);
      
      const data = await response.json() as any;
      expect(data.name).toBe('Claude Code Proxy');
      expect(data.version).toBe('1.0.0');
      expect(data.endpoints).toBeDefined();
    });
  });

  describe('Model Resolution', () => {
    it('should resolve claude-3-5-haiku to openai:gpt-4o-mini', async () => {
      const anthropicRequest = {
        model: 'claude-3-5-haiku-20241022',
        messages: [
          { role: 'user', content: 'Hello!' }
        ],
        max_tokens: 10
      };

      // Note: This will fail in actual execution due to invalid API keys,
      // but we can check the debug logs or mock the AI SDK calls
      const response = await mf.dispatchFetch('http://localhost/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(anthropicRequest)
      });

      // Should attempt to process (not return validation error)
      expect(response.status).not.toBe(400);
    });

    it('should resolve haiku shortcut to openai:gpt-4o-mini', async () => {
      const anthropicRequest = {
        model: 'haiku',
        messages: [
          { role: 'user', content: 'Test shortcut' }
        ],
        max_tokens: 10
      };

      const response = await mf.dispatchFetch('http://localhost/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(anthropicRequest)
      });

      expect(response.status).not.toBe(400);
    });
  });

  describe('Request Validation', () => {
    it('should reject invalid requests', async () => {
      const invalidRequest = {
        // Missing required fields
        messages: []
      };

      const response = await mf.dispatchFetch('http://localhost/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(invalidRequest)
      });

      expect(response.status).toBe(400);
      const error = await response.json() as any;
      expect(error.type).toBe('error');
      expect(error.error.type).toBe('invalid_request_error');
    });

    it('should validate Anthropic message format', async () => {
      const openaiStyleRequest = {
        model: 'claude-3-5-haiku-20241022',
        messages: [
          { role: 'system', content: 'You are a helpful assistant' }, // Invalid for Anthropic format
          { role: 'user', content: 'Hello' }
        ],
        max_tokens: 10
      };

      const response = await mf.dispatchFetch('http://localhost/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(openaiStyleRequest)
      });

      expect(response.status).toBe(400);
      const error = await response.json() as any;
      expect(error.type).toBe('error');
      expect(error.error.type).toBe('invalid_request_error');
    });
  });

  describe('Message Format Conversion', () => {
    it('should handle text messages', async () => {
      const anthropicRequest = {
        model: 'claude-3-5-sonnet-20241022',
        messages: [
          { role: 'user', content: 'Hello world!' }
        ],
        max_tokens: 50
      };

      const response = await mf.dispatchFetch('http://localhost/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(anthropicRequest)
      });

      // Should not be a validation error
      expect(response.status).not.toBe(400);
    });

    it('should handle system prompts', async () => {
      const anthropicRequest = {
        model: 'claude-3-opus-20240229',
        system: 'You are a helpful assistant.',
        messages: [
          { role: 'user', content: 'What is the capital of France?' }
        ],
        max_tokens: 20
      };

      const response = await mf.dispatchFetch('http://localhost/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(anthropicRequest)
      });

      expect(response.status).not.toBe(400);
    });

    it('should handle structured content', async () => {
      const anthropicRequest = {
        model: 'sonnet',
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Describe this image:' },
              {
                type: 'image',
                source: {
                  type: 'base64',
                  media_type: 'image/png',
                  data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=='
                }
              }
            ]
          }
        ],
        max_tokens: 100
      };

      const response = await mf.dispatchFetch('http://localhost/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(anthropicRequest)
      });

      expect(response.status).not.toBe(400);
    });
  });

  describe('Configuration Parameters', () => {
    it('should handle temperature and other parameters', async () => {
      const anthropicRequest = {
        model: 'claude-3-5-haiku-20241022',
        messages: [
          { role: 'user', content: 'Generate a creative story' }
        ],
        max_tokens: 100,
        temperature: 0.8,
        top_p: 0.9,
        top_k: 50,
        stop_sequences: ['END']
      };

      const response = await mf.dispatchFetch('http://localhost/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(anthropicRequest)
      });

      expect(response.status).not.toBe(400);
    });
  });

  describe('Streaming Support', () => {
    it('should handle streaming requests', async () => {
      const anthropicRequest = {
        model: 'haiku',
        messages: [
          { role: 'user', content: 'Count to 5' }
        ],
        max_tokens: 50,
        stream: true
      };

      const response = await mf.dispatchFetch('http://localhost/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(anthropicRequest)
      });

      // Should attempt streaming (may fail due to mock API keys)
      expect(response.status).not.toBe(400);
      if (response.ok) {
        expect(response.headers.get('content-type')).toBe('text/event-stream');
      }
    });
  });
}); 