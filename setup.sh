#!/bin/bash

echo "🚀 Setting up SkinCareAI Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_warning "Docker is not installed. Docker setup will be skipped."
    DOCKER_AVAILABLE=false
else
    DOCKER_AVAILABLE=true
fi

print_status "Creating virtual environment..."
python3 -m venv .venv

print_status "Activating virtual environment..."
source .venv/bin/activate

print_status "Installing Python dependencies..."
pip install -r requirements.txt

print_status "Installing backend dependencies..."
cd backend
npm install
cd ..

print_status "Installing frontend dependencies..."
cd frontend/Skinsavvy
npm install
cd ../..

print_status "Creating uploads directory..."
mkdir -p backend/uploads

print_status "Creating .env file from template..."
if [ ! -f .env ]; then
    cp env.example .env
    print_warning "Please edit .env file with your configuration"
else
    print_status ".env file already exists"
fi

if [ "$DOCKER_AVAILABLE" = true ]; then
    print_status "Setting up Docker..."
    docker-compose build
    print_status "Docker setup complete!"
else
    print_warning "Skipping Docker setup (Docker not available)"
fi

print_status "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Start the backend: cd backend && npm run dev"
echo "3. Start the frontend: cd frontend/Skinsavvy && npm run dev"
echo "4. Train the AI model: cd ai-model && python train_skin_model.py"
echo ""
echo "Or use Docker: docker-compose up"
