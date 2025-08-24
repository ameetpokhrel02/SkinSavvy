const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

// Routes
app.get('/', (req, res) => {
  res.json({ 
    message: 'Welcome to SkinCareAI Backend API!',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      analyze: '/api/analyze',
      products: '/api/products'
    }
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Skin analysis endpoint
app.post('/api/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    // TODO: Send image to AI service for analysis
    // For now, return mock response
    const mockAnalysis = {
      condition: 'acne',
      confidence: 0.85,
      severity: 'moderate',
      recommendations: [
        'Use gentle cleanser',
        'Apply benzoyl peroxide',
        'Avoid touching face'
      ]
    };

    res.json({
      success: true,
      analysis: mockAnalysis,
      imagePath: req.file.path
    });
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ error: 'Analysis failed' });
  }
});

// Products endpoint
app.get('/api/products', (req, res) => {
  const { condition } = req.query;
  
  // Mock product data
  const products = {
    acne: [
      { id: 1, name: 'Gentle Cleanser', price: 15.99, brand: 'SkinCare Pro' },
      { id: 2, name: 'Benzoyl Peroxide Gel', price: 12.99, brand: 'ClearSkin' }
    ],
    dark_spots: [
      { id: 3, name: 'Vitamin C Serum', price: 25.99, brand: 'BrightSkin' },
      { id: 4, name: 'Retinol Cream', price: 35.99, brand: 'AgeDefy' }
    ],
    wrinkles: [
      { id: 5, name: 'Anti-Aging Cream', price: 45.99, brand: 'Youthful' },
      { id: 6, name: 'Hyaluronic Acid Serum', price: 28.99, brand: 'HydratePro' }
    ]
  };

  const recommendedProducts = condition ? products[condition] || [] : [];
  
  res.json({
    success: true,
    products: recommendedProducts
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

app.listen(PORT, () => {
  console.log(`🚀 SkinCareAI Backend running on port ${PORT}`);
  console.log(`�� Health check: http://localhost:${PORT}/health`);
});