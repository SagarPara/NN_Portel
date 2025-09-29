import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
#from tensorflow.keras.models import load_model
from tensorflow import keras
import pickle
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
import joblib
#get_custom_objects = tf.keras.utils.get_custom_objects

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import get_custom_objects
#import keras


import os
print(os.path.exists("NN_model_Portel.h5"))  # Should print True

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)


# Register custom objects - LeakyReLU
get_custom_objects().update({'LeakyReLU': LeakyReLU})

# Then load the model
model = load_model("NN_model_Portel.h5", custom_objects={'LeakyReLU': LeakyReLU}, compile=False)

#Manally reassign the loss function
model.compile(loss=MeanSquaredError(), optimizer='adam')

#model = load_model("NN_model_Portel.h5", compile=False)
model.save("NN_model_Portel.keras")  # Save in the new `.keras` format

#reload the model in keras format
model = load_model("NN_model_Portel.keras")


#Print model summary
#model.summary





#Load Encoders
with open("encoders.pkl", "rb") as f:
     encoders = pickle.load(f)

# Validate encoders
if not isinstance(encoders, dict):
    raise ValueError("Encoders file is not loaded correctly!")


target_encoder = encoders.get("target_encoder", None)
category_encoder = encoders.get("store_primary_category_encoder", None)
weekday_encoder = encoders.get("weekday_encoder", None)


# Load the scaler
scaler = joblib.load("scaler.pkl")


app = Flask(__name__)

#lets create API endpoints
@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/submit", methods=["GET", "POST"])
def submit_form():
    if request.method == "POST":
    #    return "I will make the prediction but this on GET method"
    #else:
        
        sales_req = request.form


        try:    
            
            # Encode store_id using Target Encoder
            store_id = sales_req.get("store_id", "")
            subtotal = float(sales_req.get("subtotal", 0))  # Required for target encoding
            
            print("store_id received:", store_id)
            
            # Convert store_id to Dataframe before transformation
            store_id_df = pd.DataFrame({"store_id": [store_id]})
            print("store_id DataFrame:\n", store_id_df)

            store_id_encoded = target_encoder.transform(store_id_df)["store_id"].values[0]


            print("Type of target_encoder:", type(target_encoder))
            print("target_encoder mapping keys:", target_encoder.mapping.keys())
    
            
            print("Known store_id values in encoder:", target_encoder.mapping.keys())
            print("Target Encoder Type:", type(target_encoder))
            print("Target Encoder Mapping Keys:", target_encoder.mapping.keys())
            print("Encoded store_id:", store_id_encoded)


            # Encode store_primary_category
            category = sales_req.get("store_primary_category", "")
            encoded_category = category_encoder.transform([category])[0] if category in category_encoder.classes_ else -1

            # Encode weekday
            weekday = sales_req.get("weekday", "")
            encoded_weekday = weekday_encoder.transform([weekday])[0] if weekday in weekday_encoder.classes_ else -1

            
            # Extract created_at details
            created_at_str = sales_req.get("created_at", "")
            created_at_dt = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M")

            created_at_day = created_at_dt.day
            created_at_month = created_at_dt.month
            created_at_year = created_at_dt.year


            # Prepare final feature array in exact training order
            input_data = np.array([[
                float(sales_req.get("market_id", 0)),         # market_id
                store_id_encoded,                             # store_id (target encoded)
                encoded_category,                             # store_primary_category (label encoded)
                float(sales_req.get("order_protocol", 0)),    # order_protocol
                float(sales_req.get("total_items", 0)),       # total_items
                float(sales_req.get("subtotal", 0)),          # subtotal
                float(sales_req.get("num_distinct_items", 0)),# num_distinct_items
                float(sales_req.get("min_item_price", 0)),    # min_item_price
                float(sales_req.get("max_item_price", 0)),    # max_item_price
                float(sales_req.get("total_onshift_partners", 0)),  # total_onshift_partners
                float(sales_req.get("total_busy_partners", 0)),     # total_busy_partners
                float(sales_req.get("total_outstanding_orders", 0)),# total_outstanding_orders
                encoded_weekday,                               # weekday (label encoded)
                created_at_day,                                # created_at_day
                created_at_month,                              # created_at_month
                created_at_year                                # created_at_year
            ]], dtype=np.float32)

            
            print("Input Data:", input_data)

            # --- Match training feature names ---
            feature_names = [
                'market_id', 'store_id', 'store_primary_category', 'order_protocol',
                'total_items', 'subtotal', 'num_distinct_items', 'min_item_price',
                'max_item_price', 'total_onshift_partners', 'total_busy_partners',
                'total_outstanding_orders', 'weekday', 'created_at_day',
                'created_at_month', 'created_at_year'
            ]

            # Convert to DataFrame for scaler
            input_df = pd.DataFrame(input_data, columns=feature_names)

            # --- Scale correctly ---
            features_scaled = scaler.transform(input_df)


            #Make predictions
            raw_prediction = model.predict(features_scaled)[0][0]  # Extract single value
            print("Predicted Result:", raw_prediction)


            # since target is not scaled
            final_prediction = raw_prediction
            

            # Step 8: Convert seconds → hours
            # predicted_hours = final_prediction / 3600
            predicted_hours = final_prediction

            
            
            #TO GET SINGLE VALUE CODE FROM POSTMAN WE WILL UNCOMMENT ON BELOW CODE
            return jsonify({"Predicted Delivery Time": round(float(predicted_hours), 2)})  # Ensure JSON serializable output    

        except ValueError as e:
             
            return jsonify({"error": f"Invalid input: {str(e)}"}), 400    

if __name__ == "__main__":
    app.run(debug=True)    
