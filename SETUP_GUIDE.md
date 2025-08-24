# 🚀 SkinCareAI Setup Guide

This guide will help you set up and run the SkinCareAI project on your local machine.

## 📋 Prerequisites

Before starting, make sure you have the following installed:

- **Python 3.8+** - [Download here](https://www.python.org/downloads/)
- **Node.js 16+** - [Download here](https://nodejs.org/)
- **npm** (comes with Node.js)
- **Docker** (optional) - [Download here](https://www.docker.com/products/docker-desktop/)
- **Git** - [Download here](https://git-scm.com/)

## 🛠️ Quick Setup (Recommended)

### Option 1: Automated Setup Script

```bash
# Make the setup script executable and run it
chmod +x setup.sh
./setup.sh
```

This script will:
- ✅ Create a Python virtual environment
- ✅ Install all Python dependencies
- ✅ Install backend Node.js dependencies
- ✅ Install frontend dependencies
- ✅ Create necessary directories
- ✅ Set up environment variables
- ✅ Build Docker containers (if Docker is available)

### Option 2: Manual Setup

If you prefer to set up manually or the script fails, follow these steps:

#### 1. Python Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

#### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Create uploads directory
mkdir uploads

# Return to root
cd ..
```

#### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend/Skinsavvy

# Install dependencies
npm install

# Return to root
cd ../..
```

#### 4. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit the .env file with your configuration
nano .env  # or use your preferred editor
```

## 🏃‍♂️ Running the Application

### Development Mode

#### 1. Start the Backend

```bash
cd backend
npm run dev
```

The backend will be available at: `http://localhost:3000`

#### 2. Start the Frontend

```bash
cd frontend/Skinsavvy
npm run dev
```

The frontend will be available at: `http://localhost:5173`

#### 3. Train the AI Model (Optional)

```bash
cd ai-model
python train_skin_model.py
```

### Production Mode with Docker

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d
```

## 📁 Project Structure

```
SkinCareAI/
├── 📁 backend/                 # Node.js/Express API
├── 📁 frontend/                # React.js Web App
├── 📁 ai-model/                # Python AI/ML Service
├── 📁 database_schema/         # Database schemas
├── 📁 docs/                    # Documentation
├── 📁 infrastructure/          # Deployment configs
├── 📁 scripts/                 # Utility scripts
├── 📁 nginx/                   # Nginx configuration
├── docker-compose.yaml         # Docker services
├── Dockerfile                  # Backend Dockerfile
├── Dockerfile.ai               # AI service Dockerfile
├── requirements.txt            # Python dependencies
├── setup.sh                    # Setup script
└── README.md                   # Project overview
```

## 🔧 Configuration

### Environment Variables

Edit the `.env` file to configure:

- **Database settings** (PostgreSQL)
- **AI service URL**
- **JWT secrets**
- **File upload settings**
- **External API keys**

### Database Setup

The project uses PostgreSQL. You can:

1. **Use Docker** (recommended for development):
   ```bash
   docker-compose up database
   ```

2. **Install PostgreSQL locally**:
   - Install PostgreSQL
   - Create database: `skinsavvy`
   - Update `.env` with your credentials

## 🧪 Testing the Setup

### Backend API Test

```bash
# Test the health endpoint
curl http://localhost:3000/health

# Test the main endpoint
curl http://localhost:3000/
```

### Frontend Test

1. Open `http://localhost:5173` in your browser
2. You should see the SkinCareAI interface

### AI Model Test

```bash
cd ai-model
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))
"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 3000
lsof -i :3000

# Kill the process
kill -9 <PID>
```

#### 2. Python Dependencies Issues
```bash
# Reinstall in clean environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. Node.js Dependencies Issues
```bash
# Clear npm cache and reinstall
cd backend
rm -rf node_modules package-lock.json
npm install
```

#### 4. Docker Issues
```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Getting Help

If you encounter issues:

1. Check the logs: `docker-compose logs`
2. Verify all prerequisites are installed
3. Check the `.env` configuration
4. Review the error messages in the console

## 🚀 Next Steps

After successful setup:

1. **Train the AI model** with your dataset
2. **Configure the database** with product data
3. **Set up external APIs** for e-commerce integration
4. **Deploy to production** using the infrastructure scripts

## 📚 Additional Resources

- [Project Documentation](./docs/)
- [API Documentation](./docs/api/)
- [Database Schema](./database_schema/)
- [Deployment Guide](./infrastructure/)

---

**Happy coding! 🎉**

If you need help, check the troubleshooting section or create an issue in the project repository.
