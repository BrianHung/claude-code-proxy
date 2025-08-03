import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'miniflare',
    // Test files
    include: ['test/**/*.test.ts'],
    globals: true,
  },
}); 