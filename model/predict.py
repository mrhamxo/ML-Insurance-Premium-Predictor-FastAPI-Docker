import pickle
import pandas as pd

# -------------------- Load ML Model --------------------

# Load the trained model from file using binary mode
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# -------------------- MODEL VERSION --------------------

# MLFlow 
MODEL_VERSION = '1.0.0'

# Get class labels from model (important for matching probabilities to class names)
class_labels = model.classes_.tolist()

def predict_output(user_input: dict):
    
    df = pd.DataFrame([user_input])
    
    # predict the class label using the model (high, medium, low)
    predicted_class = model.predict(df)[0]
    
    # Get probabilities for all classes (highest confidence)
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)
    
    # Create mapping: {class_name: probability}
    class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probabilities)))

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs
    }