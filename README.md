# 🐱🐶 Cats vs Dogs Classification - HOG + SVM

This project classifies cat and dog images using a classical machine learning pipeline:
- Feature extraction using Histogram of Oriented Gradients (HOG)
- Classification using Support Vector Machine (SVM)

## 📁 Dataset
Used Kaggle's Cats vs Dogs dataset:
- 4000 cats + 4000 dogs for training
- 1000 cats + 1000 dogs for testing

## ⚙️ Tools Used
- Python
- OpenCV
- Scikit-learn
- HOG Descriptor
- SVM Classifier
- ReportLab (PDF report)

## 📊 Results
- Validation Accuracy: 67.75%
- Test Accuracy: 63.05%
- Detailed classification reports are included in the PDF.

## 📄 Report
See `svm_cats_vs_dogs_report.pdf` for full metrics, explanation, and conclusion.

## 📌 Run
```bash
python svm_cats_vs_dogs.py
python svm_report.py

