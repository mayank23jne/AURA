#!/bin/bash

# AURA Agentic Platform - Docker Run Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}AURA Agentic Platform - Docker Deployment${NC}"
echo "============================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${RED}Please edit .env and add your API keys before running again.${NC}"
    exit 1
fi

# Check for required environment variables
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY not set in .env file${NC}"
    echo "Please add your OpenAI API key to the .env file"
    exit 1
fi

# Parse command line arguments
COMMAND=${1:-up}

case $COMMAND in
    build)
        echo -e "${GREEN}Building Docker images...${NC}"
        docker-compose build --no-cache
        ;;
    up)
        echo -e "${GREEN}Starting AURA Platform...${NC}"
        docker-compose up -d
        echo -e "${GREEN}Platform started! Access at http://localhost:8080${NC}"
        echo "View logs: docker-compose logs -f aura-platform"
        ;;
    down)
        echo -e "${YELLOW}Stopping AURA Platform...${NC}"
        docker-compose down
        ;;
    logs)
        docker-compose logs -f aura-platform
        ;;
    status)
        echo -e "${GREEN}Platform Status:${NC}"
        docker-compose ps
        echo ""
        echo "Health check:"
        curl -s http://localhost:8080/health | python -m json.tool || echo "Platform not responding"
        ;;
    restart)
        echo -e "${YELLOW}Restarting AURA Platform...${NC}"
        docker-compose restart aura-platform
        ;;
    clean)
        echo -e "${RED}Cleaning up Docker resources...${NC}"
        docker-compose down -v --remove-orphans
        docker system prune -f
        ;;
    *)
        echo "Usage: $0 {build|up|down|logs|status|restart|clean}"
        exit 1
        ;;
esac
