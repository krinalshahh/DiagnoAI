# **DiagnoAI – AI-Powered Medical Diagnosis System**

## **Overview**

**DiagnoAI** is an **AI-driven healthcare platform** designed to **automate medical diagnostics and reporting**. It integrates **deep learning models for X-ray analysis** with **machine learning-based disease prediction** based on symptoms provided by patients. The system enhances **clinical efficiency, diagnostic accuracy, and accessibility**, making it a valuable tool for radiologists, doctors, and healthcare providers.

**Key Features:**

- **Automated X-ray Image Analysis** using deep learning models.
- **Symptom-Based Disease Prediction** powered by machine learning.
- **AI-Generated Medical Reports** with professional formatting.
- **User-Friendly Web Application** built with Django and HTML,CSS,Javascript.
- **Scalable & Deployable on Cloud or Local Servers.**

---

## **System Architecture**

### **1. Input: Datasets**

DiagnoAI processes specialized datasets to train dedicated models for different diseases:

**X-ray-Based Diagnoses:**

- **Abdominal Trauma Detection** → (e.g., RSNA 2023 Dataset)
- **Chest X-ray Classification** → (Normal, Bacterial Pneumonia, Viral Pneumonia)
- **Kidney Disease Classification** → (Cysts, Stones, Tumors)
- **Bone Fracture Detection** → (Multi-region Fracture Dataset)
- **Knee Osteoporosis Classification** → (Normal, Early, Advanced)

**Symptom-Based Disease Prediction Dataset:**

- [Disease Prediction Dataset (Kaggle)](https://www.kaggle.com/neelima98/disease-prediction-using-machine-learning)

### **2. AI Model Processing (Deep Learning + Machine Learning)**

**Deep Learning Models (CNN-Based) for X-ray Image Analysis:**

- **EfficientNet** for Abdominal Trauma Detection
- **ResNet** for Chest X-ray & Bone Fracture Classification
- **Custom CNN** for Kidney Diseases & Knee Osteoporosis

**Machine Learning Model for Symptom-Based Disease Prediction:**

- Uses **Decision Trees**, **Random Forest**, and **Naïve Bayes** classifiers.

### **3. Automated Report Generation**

Model predictions are processed by **Large Language Models (LLMs)** to generate a detailed diagnostic report that includes:

- **Detected Disease Name**
- **Confidence Score**
- **Medical Recommendations**

### **4. User Interface & Deployment**

**Web Application:**

- **Upload X-ray Images** for analysis.
- **Get Instant AI-Generated Reports.**
- **Predict Diseases** based on symptoms.

**Output Screenshots:**
https://drive.google.com/file/d/16Z-G7SqpX6AaAhfVNXVR_ToGR_MQK7pL/view?usp=drive_link

**Deployment Options:**

- **Local Server:** Django & PostgreSQL.
- **Cloud Deployment:** AWS, Azure, or GCP.

---

## **AI Models & Training**

### **Preprocessing & Model Training Notebooks**

- `AbdominalTraumaDetection.py` → Binary multi-label classification (liver, kidney, bowel injury).
- `BoneFractures.py` → Detects fractures across multiple anatomical regions.
- `ChestXRay.py` → Classifies X-rays into Normal, Bacterial Pneumonia, and Viral Pneumonia.
- `KidneyDiseasesClassification.py` → Detects cysts, stones, and tumors.
- `Knee_Osteoporosis.py` → Classifies Normal, Osteopenia, and Osteoporosis.

### **Model Performance & Results**

| **Disease Category**          | **Model Used**  | **Accuracy (%)** |
| ----------------------------- | --------------- | ---------------- |
| Abdominal Trauma              | EfficientNet B0 | **90.21%**       |
| Chest X-ray Classification    | ResNet          | **91.5%**        |
| Kidney Disease Classification | Custom CNN      | **99%**          |
| Bone Fracture Detection       | ResNet          | **89.3%**        |
| Knee Osteoporosis             | Custom CNN      | **83.2%**        |

### **Automated Report Generation Process**

- AI model predicts disease.
- **LLM** converts outputs into a structured medical report.
- Report includes **explanations**, **severity levels**, and **recommendations**.

---

## **Web Application & User Interface**

### **Features of DiagnoAI Web App**

- **Upload X-ray Images:** Get AI-powered predictions.
- **View Model Results:** See disease classification and confidence scores.
- **Symptom-Based Prediction:** Enter symptoms to get possible diagnoses.
- **Download AI-Generated Reports:** Professional reports in seconds.

### **System Workflow**

![System Workflow](app.gif)

---

## **Setup & Deployment Guide**

### **Local Installation**

1. **Database Setup:**  
    First make sure **PostgreSQL** and pgadmin is install in your system.
   then you have to manually create a DB instance on PostgreSQL named **DiagnoAI**, better use PgAdmin for that.
   make a new environment(recommended) and run...

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Database Migrations:**
   ```bash
   python manage.py makemigrations
    python manage.py migrate
   ```
4. **Start the Development Server**
   ```bash
   python manage.py runserver
   ```
5. **Open your web browser and navigate to http://127.0.0.1:8000/ to start using DiagnoAI.**
