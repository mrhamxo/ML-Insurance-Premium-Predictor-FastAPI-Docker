# ML-Insurance-Premium-Predictor-FastAPI-Docker
A fully containerized, production-ready AI-driven insurance premium prediction system built using FastAPI and Docker. The system classifies insurance categories based on personal and lifestyle factors (age, BMI, smoking status, income, city tier, occupation).

## 🚀 Key Features

### ✅ **FastAPI-based Insurance Predictor**

* Predicts insurance premium categories: *low*, *medium*, or *high*
* Considers:

  * Age group
  * BMI (auto-calculated)
  * Smoking habits
  * Income (in LPA)
  * City tier logic (custom based on major Pakistani cities)
  * Occupation type

## 📦 **Tech Stack**

| Component        | Technology                  |
| ---------------- | --------------------------- |
| Backend API      | FastAPI                     |
| ML Model         | scikit-learn, pandas        |
| Containerization | Docker, Uvicorn             |

## 📊 **Machine Learning Model**

* Preprocessing: Categorical encoding, BMI calculation, normalization
* Algorithm: RandomForestClassifier or LogisticRegression (tuned for class balance)
* Trained and saved as `model.pkl` in `insurance_premium_ml.ipynb`

## 🐳 **Run the Project with Docker**

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ml-insurance-fastapi-docker.git
cd ml-insurance-fastapi-docker
```

### 2. Build and Run with Docker

```bash
docker build -t insurance-fastapi-api .
docker run -d -p 8000:8000 insurance-fastapi-api
```

### 🎯 Prediction Endpoint

```http
POST /predict
```

**Example Request:**

```json
{
  "age": 35,
  "weight": 70,
  "height": 1.75,
  "income_lpa": 10,
  "smoker": true,
  "city": "Lahore",
  "occupation": "private_job"
}
```

**Response:**

```json
{
  "predicted_category": "high"
}
```

## 📬 **Questions?**

For queries or collaboration, feel free to connect via [LinkedIn](https://www.linkedin.com/in/muhammad-hamza-khattak/)
