# 🪪 AI-Based ID Verification System

An end-to-end AI system for identity verification that combines ID classification and face verification into a unified pipeline.

## 🚀 Overview
The system operates in two main stages:

1. ID Classification (Gate Step)  
A computer vision model first determines whether the uploaded image is a valid Egyptian national ID card or not.

2. Face Verification (Main Step)  
If the image is classified as an ID, the system proceeds to verify identity by comparing the face extracted from the ID with a selfie image.

This design ensures robustness by filtering invalid inputs before verification.

## 🧠 Pipeline
Input Image (ID) + Selfie  
→ ID Classifier (MobileNetV3)  
→ If valid ID  
→ Face Detection (MTCNN)  
→ Face Embedding (FaceNet)  
→ Cosine Similarity  
→ Match / Not Match  

## 🔍 Features
- ID vs Non-ID classification using MobileNetV3
- Face detection using MTCNN
- Face embedding extraction using FaceNet
- Identity verification using cosine similarity
- RESTful API built with FastAPI
- Deployed on Hugging Face Spaces

## 🗂 Dataset
- Custom dataset of Egyptian national ID cards and non-ID images
- Includes preprocessing:
  - Image resizing (224x224)
  - Normalization
  - Train / Validation / Test split

## 🛠 Tech Stack
Python, PyTorch, OpenCV, FastAPI, FaceNet (facenet-pytorch), NumPy, Hugging Face Spaces

## ⚙️ API Endpoints

POST /ID_check  
Input: Image  
Output:
{
  "prediction": "ID" or "NOT_ID"
}

POST /verify  
Input: ID image + Selfie image  
Output:
{
  "match": true,
  "distance": 0.32,
  "threshold": 0.5
}


## 📌 Notes
- The system follows a two-stage pipeline (ID classification → face verification)
- Designed as a backend AI system that can be integrated into real applications
