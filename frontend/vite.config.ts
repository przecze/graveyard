import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: ['graveyard.janczechowski.com'],
    hmr: {
      // Use a dedicated path so nginx can route the WebSocket directly
      path: '/__vite_hmr',
    },
    watch: {
      // Required for file-change detection inside Docker on macOS
      usePolling: true,
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})
