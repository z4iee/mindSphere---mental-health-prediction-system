import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the data
def load_and_preprocess_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)
    
    # Clean column names by stripping whitespace
    data.columns = data.columns.str.strip()
    
    # Define the target variable - we'll create one based on happiness and stress levels
    # This is a simplified approach - you might want to refine this based on your specific needs
    conditions = [
        (data['How often do you feel happy with your daily life?'].isin(['Very Often'])) |
        (data['How often do you feel overwhelmed by your academic responsibilities?'].isin(['Never', 'Rarely'])),
        (data['How often do you feel happy with your daily life?'].isin(['Never', 'Rarely'])) |
        (data['How often do you feel overwhelmed by your academic responsibilities?'].isin(['Always', 'Often']))
    ]
    choices = ['Good', 'Poor']
    data['Mental_Health_Status'] = np.select(conditions, choices, default='Average')
    
    # Select features - using the most relevant columns
    features = [
        'Your Age?', 'Your Gender?', 'What is your current level of education?',
        'How often do you feel happy with your daily life?',
        'Do you feel that your academic workload affects your mental health?',
        'How many hours of sleep do you typically get?',
        'What are the main sources of your academic stress?',
        'How often do you feel overwhelmed by your academic responsibilities?',
        'Do you feel you have enough time to balance academics, personal life, and hobbies?',
        'How do you usually cope with stress? (Select all that apply)',
        'How supportive do you feel your school/college environment is regarding mental health?',
        'Do you feel isolated or lonely in your academic environment?'
    ]
    
    X = data[features]
    y = data['Mental_Health_Status']
    
    # Encode categorical variables
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == object:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            label_encoders[column] = le
    
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    return X, y, label_encoders, le_y, features

# Train the Random Forest model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.2f}")
    
    return model

# Get user input for prediction
def get_user_input(label_encoders, features):
    user_input = {}
    
    print("\nPlease answer the following questions about your mental health:\n")
    
    # Age
    print("1. Your Age?")
    print("Options: under 18, 18-20, 21-24")
    age = input("Your answer: ").strip()
    user_input['Your Age?'] = age
    
    # Gender
    print("\n2. Your Gender?")
    print("Options: male, female, other")
    gender = input("Your answer: ").strip().lower()
    user_input['Your Gender?'] = gender
    
    # Education level
    print("\n3. What is your current level of education?")
    print("Options: Schooling, Undergraduate, Graduate")
    education = input("Your answer: ").strip()
    user_input['What is your current level of education?'] = education
    
    # Happiness frequency
    print("\n4. How often do you feel happy with your daily life?")
    print("Options: Never, Rarely, Sometimes, Very Often")
    happiness = input("Your answer: ").strip()
    user_input['How often do you feel happy with your daily life?'] = happiness
    
    # Academic workload impact
    print("\n5. Do you feel that your academic workload affects your mental health?")
    print("Options: No, not really, Yes, but only slightly, Yes, significantly")
    workload = input("Your answer: ").strip()
    user_input['Do you feel that your academic workload affects your mental health?'] = workload
    
    # Sleep hours
    print("\n6. How many hours of sleep do you typically get?")
    print("Options: less than 6 hours, 6-8 hours, 8-10 hours")
    sleep = input("Your answer: ").strip()
    user_input['How many hours of sleep do you typically get?'] = sleep
    
    # Academic stress sources
    print("\n7. What are the main sources of your academic stress?")
    print("Options: Exams and grades, Assignment deadlines, Parental or teacher expectations, Fear of failure, Other")
    stress_source = input("Your answer: ").strip()
    user_input['What are the main sources of your academic stress?'] = stress_source
    
    # Feeling overwhelmed
    print("\n8. How often do you feel overwhelmed by your academic responsibilities?")
    print("Options: Never, Rarely, Sometimes, Often, Always")
    overwhelmed = input("Your answer: ").strip()
    user_input['How often do you feel overwhelmed by your academic responsibilities?'] = overwhelmed
    
    # Time balance
    print("\n9. Do you feel you have enough time to balance academics, personal life, and hobbies?")
    print("Options: Yes, I have a good balance, No, I struggle to find time for myself, I don't have any personal time")
    time_balance = input("Your answer: ").strip()
    user_input['Do you feel you have enough time to balance academics, personal life, and hobbies?'] = time_balance
    
    # Coping mechanisms
    print("\n10. How do you usually cope with stress? (Select one main method)")
    print("Options: Talking to friends or family, Exercising or physical activity, Engaging in hobbies, Meditation or mindfulness, Ignoring the stress")
    coping = input("Your answer: ").strip()
    user_input['How do you usually cope with stress? (Select all that apply)'] = coping
    
    # School support
    print("\n11. How supportive do you feel your school/college environment is regarding mental health?")
    print("Options: Not supportive at all, Somewhat supportive, Very supportive, I'm not sure")
    support = input("Your answer: ").strip()
    user_input['How supportive do you feel your school/college environment is regarding mental health?'] = support
    
    # Isolation feeling
    print("\n12. Do you feel isolated or lonely in your academic environment?")
    print("Options: Never, Rarely, Sometimes, Often, Always")
    isolation = input("Your answer: ").strip()
    user_input['Do you feel isolated or lonely in your academic environment?'] = isolation
    
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Encode user input using the same label encoders
    for column in user_df.columns:
        if column in label_encoders:
            # Handle unseen labels by adding them to the encoder
            try:
                user_df[column] = label_encoders[column].transform(user_df[column])
            except ValueError:
                # If label wasn't seen during training, assign it to the most frequent category
                user_df[column] = 0  # You might want to handle this differently
    
    return user_df

# Main function
def main():
    # Load and preprocess data
    filepath = 'mental_health_data.csv'  # Replace with your file path
    X, y, label_encoders, le_y, features = load_and_preprocess_data(filepath)
    
    # Train model
    model = train_model(X, y)
    
    # Get user input
    user_df = get_user_input(label_encoders, features)
    
    # Make prediction
    prediction = model.predict(user_df)
    mental_health_status = le_y.inverse_transform(prediction)[0]
    
    # Display result
    print("\n=== Mental Health Prediction Result ===")
    print(f"Your predicted mental health status is: {mental_health_status}")
    
    # Provide some interpretation
    if mental_health_status == 'Good':
        print("You seem to be managing well! Keep up the good work and maintain your healthy habits.")
    elif mental_health_status == 'Average':
        print("You're doing okay, but there might be some areas where you could improve your mental wellbeing.")
    else:
        print("Your responses suggest you might be struggling. Consider reaching out for support from friends, family, or professionals.")

if __name__ == "__main__":
    main()