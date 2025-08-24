#!/bin/bash

echo "🚀 Starting SkinCareAI Project..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if backend is already running
if curl -s http://localhost:3000/health > /dev/null; then
    print_status "Backend is already running on port 3000"
else
    print_status "Starting backend..."
    cd backend
    npm start &
    cd ..
    sleep 3
fi

# Check if frontend is already running
if curl -s http://localhost:5173 > /dev/null; then
    print_status "Frontend is already running on port 5173"
else
    print_status "Starting frontend..."
    cd frontend/Skinsavvy
    npm run dev &
    cd ../..
    sleep 3
fi

print_status "🎉 SkinCareAI is starting up!"
echo ""
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend API: http://localhost:3000"
echo "🏥 Health Check: http://localhost:3000/health"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user to stop
wait
