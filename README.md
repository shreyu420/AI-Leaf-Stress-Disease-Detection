🌱 AI-Based Leaf Disease & Stress Detection System
An Explainable, Video-Aware, Multilingual Decision Support Tool for Farmers
📌 Overview

This project is an end-to-end AI-powered decision support system designed to help farmers detect crop leaf diseases and nutritional stress at an early stage using images and short videos.

Unlike basic image-classification projects, this system focuses on real-world agricultural challenges such as:

Unclear images
Motion blur in videos
Varying lighting conditions
Need for trust and explainability
Language barriers for farmers

The system not only predicts what problem exists, but also explains why the model made that decision, estimates severity, and provides expert-style advisory guidance in multiple languages.

🚜 Real-World Problem & Motivation
Why this problem matters

In real farming conditions:

Leaf diseases and nutrient deficiencies are often detected too late
Farmers rely on visual guesswork or delayed expert visits
Late diagnosis leads to:
Reduced crop yield
Increased pesticide usage
Financial losses
Environmental harm
Limitations of existing solutions

Most existing AI solutions:

Work only on single images
Fail under poor image quality
Provide no explanation for predictions
Are not accessible to non-English-speaking farmers

💡 Our Solution

We designed a practical, farmer-centric AI system that:

Works with both images and videos
Filters out low-quality frames automatically
Uses multi-frame voting for reliable video predictions
Allows human-in-the-loop frame selection
Explains decisions using Explainable AI (Grad-CAM)
Estimates severity to guide urgency
Provides multilingual expert advisory

This makes the system suitable for real deployment, not just academic demonstration.

✨ Key Features (What Makes This Project Stand Out)
🔍 1. Image-Based Analysis

Upload a single leaf image
Hierarchical prediction:
Disease detection
If healthy → stress detection
Confidence & severity estimation

🎥 2. Video-Based Leaf Analysis (Major Innovation)

Accepts short leaf videos (5–15 seconds)

Automatically:

Extracts frames
Removes blurry / overexposed / irrelevant frames
Detects leaf presence
Performs multi-frame aggregation for robust prediction

👤 3. Human-in-the-Loop AI

Displays all high-quality frames
User can select a representative frame
Enables manual verification and trust
Mirrors real medical & agricultural decision-support systems

🧠 4. Explainable AI (Grad-CAM)

Visual heatmaps highlight regions influencing prediction
Helps users and evaluators understand why a disease was detected
Increases transparency and trust

📊 5. Severity & Confidence Estimation

Confidence score visualization

Severity levels:

🟢 Low
🟠 Moderate
🔴 High

Helps farmers decide urgency of intervention

🌐 6. Multilingual Expert Advisory Chatbot

Supports English & Hindi

Answers questions related to:

Symptoms
Causes
Treatment
Prevention

Makes the system accessible to regional farmers

🧠 System Architecture

Pipeline Explanation
Image / Video Input
Frame Extraction (for video)
Quality Filtering
Blur detection
Brightness validation
Leaf presence detection
Disease Detection (EfficientNet-B0)
Stress Detection (if healthy)
Grad-CAM Explainability
Severity & Confidence Estimation
Multilingual Expert Advisory

🖼️ Application Screenshots
Image Analysis
Video Analysis
Explainable AI (Grad-CAM)
Multilingual Advisory

🧪 Quick Test for Evaluators (Important)

To test the system instantly without searching for data:

Use the provided sample inputs:

sample_inputs/sample_leaf.jpg
sample_inputs/sample_video.mp4
Run the application:
pip install -r requirements.txt
streamlit run src/app.py


Upload the sample image or video from the sample_inputs/ folder.

This ensures a smooth and frustration-free evaluation experience.

🛠️ Technologies Used

Python
PyTorch
EfficientNet-B0
Streamlit
OpenCV
Grad-CAM
NumPy / PIL

🎯 Use Cases

Farmers
Agricultural extension officers
Precision agriculture platforms
Crop health monitoring systems
Early-warning agricultural AI tools

🚀 Future Improvements

Larger domain-specific field datasets
Temporal learning models for video sequences
Active learning from low-confidence cases
More regional language support
Mobile and cloud deployment

⚠️ Disclaimer

This system is a decision-support tool and does not replace professional agronomist consultation. Final decisions should be taken with expert advice when necessary.

👨‍💻 Author

Shrey Patel
AI Innovation Challenge Project

⭐ Final Note to Evaluators

This project focuses not only on model accuracy, but on:

Trust
Explainability
Usability
Real-world agricultural constraints
It is designed as a deployable AI system, not just a machine-learning demo.
