#!/bin/bash

# Maven Drive Copilot - Simple Dev Script
# Just run: ./dev.sh

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${BLUE}¶${NC} $1"; }
success() { echo -e "${GREEN}${NC} $1"; }
warn() { echo -e "${YELLOW} ${NC} $1"; }
error() { echo -e "${RED}${NC} $1"; }

# Get script directory
cd "$( dirname "${BASH_SOURCE[0]}" )"

clear
echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
echo "  Maven Drive Copilot"
echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
echo ""

# Check if .env exists
if [ ! -f "backend/.env" ]; then
    if [ -f "backend/.env.example" ]; then
        warn ".env not found. Creating from .env.example..."
        cp backend/.env.example backend/.env
        warn "Please edit backend/.env with your Google OAuth credentials!"
        echo ""
        read -p "Press Enter after updating .env..."
    fi
fi

# Install backend dependencies if needed
if [ ! -d "backend/venv" ]; then
    info "Setting up Python virtual environment..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    cd ..
    success "Backend setup complete!"
fi

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    info "Installing frontend dependencies..."
    cd frontend
    npm install -q
    cd ..
    success "Frontend setup complete!"
fi

# Create logs directory
mkdir -p logs

# Kill any existing processes on ports
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    info "Stopping existing backend..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fi

if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    info "Stopping existing frontend..."
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
fi

echo ""
info "Starting services..."
echo ""

# Start backend
cd backend
source venv/bin/activate
python -m src.main > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid
cd ..

sleep 2

if ! kill -0 $BACKEND_PID 2>/dev/null; then
    error "Backend failed to start!"
    cat logs/backend.log
    exit 1
fi

success "Backend started"

# Start frontend
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid
cd ..

sleep 2

if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    error "Frontend failed to start!"
    cat logs/frontend.log
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

success "Frontend started"

echo ""
echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
success "Services are running!"
echo "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
echo ""
echo "  < Frontend: http://localhost:3000"
echo "  =' Backend:  http://localhost:8000"
echo "  =Ú API Docs: http://localhost:8000/docs"
echo ""
info "Logs:"
echo "  " Backend:  logs/backend.log"
echo "  " Frontend: logs/frontend.log"
echo ""
warn "Press Ctrl+C to stop watching logs (services keep running)"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    info "Stopping services..."

    if [ -f "logs/backend.pid" ]; then
        BACKEND_PID=$(cat logs/backend.pid)
        kill $BACKEND_PID 2>/dev/null || true
        kill -9 $BACKEND_PID 2>/dev/null || true
        rm -f logs/backend.pid
    fi

    if [ -f "logs/frontend.pid" ]; then
        FRONTEND_PID=$(cat logs/frontend.pid)
        kill $FRONTEND_PID 2>/dev/null || true
        kill -9 $FRONTEND_PID 2>/dev/null || true
        rm -f logs/frontend.pid
    fi

    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true

    success "Services stopped!"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Show live logs
tail -f logs/backend.log logs/frontend.log
