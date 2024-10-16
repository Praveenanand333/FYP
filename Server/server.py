from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
app = Flask(__name__)
CORS(app) 

aq10_model=joblib.load("../Models/AQ10/rfmodel.joblib")
#aq10_model = tf.keras.models.load_model("../Models/AQ10/ann.h5")
eeg_model = joblib.load('../Models/EEG/dtmodel.joblib')
image_model = tf.keras.models.load_model("../Models/Image/EfficientNetB4.h5")
et_model=joblib.load('../Models/ET/knn.joblib')
feature_names = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender',
       'jaundice', 'ethnicity_Asian', 'ethnicity_Black', 'ethnicity_Hispanic',
       'ethnicity_Latino', 'ethnicity_Middle Eastern ', 'ethnicity_Others',
       'ethnicity_Pasifika', 'ethnicity_South Asian', 'ethnicity_Turkish',
       'ethnicity_White-European', 'relation_Health care professional',
       'relation_Others', 'relation_Parent', 'relation_Relative',
       'relation_Self']
@app.route('/predict/aq10', methods=['POST'])
def predict_aq10():
    data = request.json 
    df = pd.DataFrame([data])

    for category in feature_names:
        if 'ethnicity_' in category or 'relation_' in category:
            df[category] = 0

    df[f"ethnicity_{data['ethnicity']}"] = 1
    df[f"relation_{data['relation']}"] = 1

    df.drop(columns=['ethnicity', 'relation'], inplace=True)

    df = df.reindex(columns=feature_names, fill_value=0)

    prediction = aq10_model.predict(df)
    predicted_class = int(prediction[0])

    class_probabilities = aq10_model.predict_proba(df)[0]

    return jsonify({
        'predicted_class': predicted_class,
        'class_0_probability': float(class_probabilities[0]),
        'class_1_probability': float(class_probabilities[1])
    })


@app.route('/predict/eeg', methods=['POST'])
def predict_eeg():
    file = request.files['file'] 
    new_data = pd.read_csv(file)
    probabilities = eeg_model.predict_proba(new_data)
    prediction = eeg_model.predict(new_data)
    class_probabilities = probabilities[0]
    eeg_class=1
    print(class_probabilities)
    print(prediction[0])
    if(float(class_probabilities[0])>class_probabilities[1]):
        eeg_class=0
    return jsonify({
        'predicted_class': eeg_class,      
        'class_0_probability': float(class_probabilities[0]), 
        'class_1_probability': float(class_probabilities[1]) 
    })


@app.route('/predict/image', methods=['POST'])
def predict_image():
    file = request.files['file']
    temp_file_path = 'temp_image.jpg'  
    file.save(temp_file_path) 
    img = image.load_img(temp_file_path, target_size=(224, 224)) 
    img_tensor = image.img_to_array(img) 
    img_tensor = np.expand_dims(img_tensor, axis=0)  
    print(img_tensor.shape) 

    predictions = image_model.predict(img_tensor) 
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_probabilities = predictions[0]
    print(class_probabilities)
    os.remove(temp_file_path)  
    img_class=0
    if(float(class_probabilities[0])>float(class_probabilities[1])):
        img_class=1
    return jsonify({
        'predicted_class': img_class,
        'class_0_probability': float(class_probabilities[1]), 
        'class_1_probability': float(class_probabilities[0])
    })


@app.route('/predict/et', methods=['POST'])
def predict():
    data = request.json
    print(data)
    features = [
        int(data['trial']),      
        int(data['stimulus']),   
        float(data['exportStartTime']),
        float(data['exportEndTime']), 
        int(data['color']),       
        int(data['categoryGroup']),
        float(data['categoryRight']), 
        float(data['categoryLeft']),  
        int(data['gender']),     
        float(data['age']),                    
        float(data['diameter']),            
        float(data['pog']),                   
        float(data['gazeVector']),           
        float(data['pupilSize']),        
        float(data['eyePosition']),  
        float(data['pupilPosition']),               
        float(data['index'])             
    ]
    
    reshaped_data = [features]
    class_probabilities = et_model.predict_proba(reshaped_data)[0]
    predicted_class = et_model.predict(reshaped_data)[0]
    
    return jsonify({
        'predicted_class': int(predicted_class),
        'class_0_probability': float(class_probabilities[0]), 
        'class_1_probability': float(class_probabilities[1])
    })




if __name__ == '__main__':
    app.run(debug=True)
