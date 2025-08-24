# ⚡ Quick Start Guide

Your SkinCareAI project is now set up and ready to run!

## 🚀 Start the Application

### Option 1: Quick Start (Recommended)
```bash
./start.sh
```

### Option 2: Manual Start
```bash
# Terminal 1 - Backend
cd backend
npm start

# Terminal 2 - Frontend  
cd frontend/Skinsavvy
npm run dev
```

## 🌐 Access Your Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3000
- **Health Check**: http://localhost:3000/health

## 📋 What's Working

✅ **Backend API** - Express.js server with endpoints:
- `GET /` - Welcome message
- `GET /health` - Health check
- `POST /api/analyze` - Skin analysis (mock)
- `GET /api/products` - Product recommendations

✅ **Python Environment** - All AI/ML dependencies installed
✅ **File Structure** - Complete project organization
✅ **Docker Configuration** - Ready for containerization

## 🔧 Next Steps

1. **Train the AI Model**:
   ```bash
   cd ai-model
   python train_skin_model.py
   ```

2. **Set up Database** (optional):
   ```bash
   # Install PostgreSQL or use Docker
   docker-compose up database
   ```

3. **Configure Environment**:
   ```bash
   # Edit the .env file
   nano .env
   ```

## 🧪 Test the API

```bash
# Health check
curl http://localhost:3000/health

# Get product recommendations
curl "http://localhost:3000/api/products?condition=acne"

# Test skin analysis (requires image file)
curl -X POST -F "image=@your_image.jpg" http://localhost:3000/api/analyze
```

## 📚 Documentation

- [Full Setup Guide](./SETUP_GUIDE.md)
- [Project Documentation](./PROJECT_DOCUMENTATION.md)
- [API Documentation](./docs/)

## 🐛 Troubleshooting

If you encounter issues:

1. **Backend not starting**: Check if port 3000 is free
2. **Frontend issues**: Try `npm install --force` in frontend directory
3. **Python issues**: Activate virtual environment: `source .venv/bin/activate`

## 🎉 You're All Set!

Your SkinCareAI project is ready for development. The backend is running and the API endpoints are functional. You can now:

- Build the frontend interface
- Train and integrate the AI model
- Add database functionality
- Deploy to production

Happy coding! 🚀
