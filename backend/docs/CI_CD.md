# AURA Platform CI/CD Documentation

## Overview

This document describes the CI/CD pipeline configuration for the AURA Agentic Platform.

## Pipeline Architecture

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Push/  │────>│  Build  │────>│  Test   │────>│ Deploy  │
│   PR    │     │         │     │         │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

## GitHub Actions Workflows

### Main CI Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  PYTHON_VERSION: "3.11"
  POETRY_VERSION: "1.7.0"

jobs:
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install black ruff mypy

      - name: Run Black
        run: black --check src tests

      - name: Run Ruff
        run: ruff check src tests

      - name: Run MyPy
        run: mypy src --ignore-missing-imports

  test:
    name: Test
    runs-on: ubuntu-latest
    needs: lint

    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_USER: aura
          POSTGRES_PASSWORD: aura
          POSTGRES_DB: aura_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov

      - name: Run tests
        env:
          DATABASE_URL: postgresql://aura:aura@localhost:5432/aura_test
          REDIS_URL: redis://localhost:6379
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    name: Build Docker
    runs-on: ubuntu-latest
    needs: test

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: aura-platform:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: lint

    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'
```

### Deploy Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

      - name: Deploy to staging
        if: github.ref == 'refs/heads/main'
        run: |
          echo "Deploying to staging..."
          # Add deployment commands here

      - name: Deploy to production
        if: startsWith(github.ref, 'refs/tags/v')
        run: |
          echo "Deploying to production..."
          # Add deployment commands here
```

### Release Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        id: changelog
        uses: mikepenz/release-changelog-builder-action@v4
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          body: ${{ steps.changelog.outputs.changelog }}
          draft: false
          prerelease: false
```

## Pipeline Stages

### 1. Lint Stage

**Tools**:
- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **MyPy**: Type checking

**Configuration**:

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "D"]
ignore = ["D100", "D104"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

### 2. Test Stage

**Test Types**:
- Unit tests
- Integration tests
- API tests

**Coverage Requirements**:
- Minimum: 80%
- Target: 90%

### 3. Build Stage

**Docker Build**:
- Multi-stage builds
- Layer caching
- Security scanning

### 4. Security Stage

**Scans**:
- Dependency vulnerabilities (Trivy)
- Secret detection
- Container scanning

### 5. Deploy Stage

**Environments**:
- Development (on PR)
- Staging (on merge to main)
- Production (on tag)

## Environment Configuration

### Secrets

Required secrets in GitHub:

| Secret | Description |
|--------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for tests |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `DATABASE_URL` | Production database URL |
| `REDIS_URL` | Production Redis URL |

### Environment Variables

```yaml
env:
  ENVIRONMENT: production
  LOG_LEVEL: INFO
  API_HOST: 0.0.0.0
  API_PORT: 8080
```

## Deployment Strategies

### Rolling Deployment

```yaml
# kubernetes/deployment.yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

### Blue-Green Deployment

1. Deploy new version to green environment
2. Run smoke tests
3. Switch traffic from blue to green
4. Keep blue as rollback

### Canary Deployment

1. Deploy to small percentage of traffic
2. Monitor metrics
3. Gradually increase traffic
4. Full rollout or rollback

## Monitoring & Alerts

### Deployment Notifications

```yaml
- name: Notify Slack
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "Deployment ${{ job.status }}",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "Deployment to ${{ env.ENVIRONMENT }}: ${{ job.status }}"
            }
          }
        ]
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### Health Checks

```yaml
- name: Health check
  run: |
    for i in {1..30}; do
      if curl -f http://localhost:8080/health; then
        exit 0
      fi
      sleep 10
    done
    exit 1
```

## Docker Configuration

### Dockerfile

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 aura

COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

COPY src/ src/
COPY config/ config/

USER aura
EXPOSE 8080

CMD ["python", "-m", "src.main"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  aura-platform:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://aura:aura@postgres:5432/aura
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Best Practices

### Version Control

- Use semantic versioning (v1.2.3)
- Tag releases
- Maintain changelog

### Security

- Scan dependencies regularly
- Rotate secrets
- Use least privilege

### Performance

- Cache dependencies
- Use multi-stage builds
- Parallel jobs where possible

### Reliability

- Automated rollbacks
- Health checks
- Gradual rollouts

## Troubleshooting

### Common Issues

**Build Failures**
- Check dependency versions
- Verify Docker context
- Review build logs

**Test Failures**
- Check service connections
- Verify environment variables
- Review test logs

**Deploy Failures**
- Check secrets
- Verify permissions
- Review deployment logs

### Debugging

```yaml
- name: Debug
  run: |
    echo "Event: ${{ github.event_name }}"
    echo "Ref: ${{ github.ref }}"
    echo "SHA: ${{ github.sha }}"
```

## Future Improvements

1. **Infrastructure as Code**: Terraform/Pulumi
2. **GitOps**: ArgoCD/Flux
3. **Feature Flags**: LaunchDarkly integration
4. **Performance Testing**: Load tests in pipeline
5. **Chaos Engineering**: Automated resilience tests
