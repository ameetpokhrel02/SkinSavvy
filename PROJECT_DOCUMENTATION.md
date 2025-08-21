# SkinCareAI Project Documentation

## Overview
SkinCareAI is an end-to-end system that uses AI/ML to analyze skin conditions from user-uploaded photos and recommends suitable skincare products. It integrates a web/mobile frontend, a backend server, an AI model, a database, and e-commerce APIs.

## System Architecture

```mermaid
flowchart LR
    subgraph Frontend
        A[User Uploads Photo via Web/Mobile App]
    end
    subgraph Backend
        B[Server (Django/Flask/Node.js)]
        C[AI Model (TensorFlow/PyTorch, CNN)]
        D[Database (PostgreSQL)]
    end
    subgraph External
        E[E-commerce API or Affiliate Link]
    end
    A -- HTTP Request: Image Data --> B
    B -- Send Image for Analysis --> C
    C -- Returns Analysis (e.g., 'Acne', 'Dryness') --> B
    B -- Queries Database for Matching Products --> D
    D -- Returns Product List --> B
    B -- Sends Structured Response --> A
    B -- Creates Affiliate Link --> E
    A -- 'Buy Now' Link --> E
```

## Components
- **Frontend:** Web/mobile app for image upload and displaying results.
- **Backend:** Handles requests, runs AI model, queries database, and integrates with e-commerce APIs.
- **AI Model:** CNN-based image classifier for skin condition detection.
- **Database:** Stores product info, categories, and (optionally) user data.
- **External APIs:** For direct product purchase or affiliate linking.

## Workflow
1. User uploads a photo.
2. Backend receives image, sends it to AI model for analysis.
3. AI model returns detected skin condition.
4. Backend queries database for matching products.
5. Product list is sent to frontend.
6. User can click 'Buy Now' to purchase via e-commerce API.

## Disclaimer
This tool is for informational purposes only and is not a substitute for professional medical advice.

## Getting Started
- Place training/validation images in `model/data/train` and `model/data/val`.
- Run `train_skin_model.py` to train the AI model.
- Set up backend and frontend as described above.

## Contact
For questions or contributions, contact the project maintainer.
