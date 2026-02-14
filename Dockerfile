# Multi-stage Dockerfile for frontend dev and production
# Build context should be project root

FROM node:24-alpine AS builder

WORKDIR /app

# Copy package files from frontend/
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copy frontend source and build
COPY frontend/ .
RUN npm run build

# Production stage with nginx
FROM nginx:alpine AS production

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config template
COPY nginx.conf.template /etc/nginx/templates/nginx.conf.template

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

# Development stage - just node with dependencies
FROM node:24-alpine AS development

WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copy source (will be overridden by volumes in docker-compose)
COPY frontend/ .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
