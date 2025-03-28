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

@app.route("/submit", methods=["GET"])
def submit_form():
    #if request.method == "POST":
    #    return "I will make the prediction but this on GET method"
    #else:
        
        sales_req = request.args

        required_fields = [
            "market_id", "created_at", "actual_delivery_time", "store_id",
            "store_primary_category", "order_protocol", "total_items",
            "subtotal", "num_distinct_items", "min_item_price",
            "max_item_price", "total_onshift_partners", "total_busy_partners",
            "total_outstanding_orders", "diff", "weekday"
        ]

        # Validate if all fields exist in the request
        for field in required_fields:
            if field not in sales_req:
                return jsonify({"error": f"Missing field: {field}"}), 400
            

        # Convert 'diff' (time difference) into seconds
        diff_str = sales_req.get("diff", "00:00:00")  # Default to zero
        h, m, s = map(int, diff_str.split(":"))
        diff_seconds = h * 3600 + m * 60 + s  # Convert to seconds


        try:    
            
            # Encode store_id using Target Encoder
            store_id = sales_req.get("store_id", "")
            subtotal = float(sales_req.get("subtotal", 0))  # Required for target encoding
            
            print("store_id received:", store_id)
            
            # Convert store_id to Dataframe before transformation
            store_id_df = pd.DataFrame({"store_id": [store_id]})
            print("store_id DataFrame:\n", store_id_df)

            if store_id in target_encoder.mapping.keys():
                #store_id_encoded = target_encoder.transform(store_id_df).iloc[0, 0]
                store_id_encoded = target_encoder.transform(store_id_df)["store_id"].values[0]
            else:
                store_id_encoded = 0  # Default value for unseen categories

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


            # Convert datetime strings to timestamps
            def to_timestamp(date_str):
                 return datetime.strptime(date_str, "%Y-%m-%dT%H:%M").timestamp() if date_str else 0

            created_at_timestamp = to_timestamp(sales_req.get("created_at", ""))
            actual_delivery_time_timestamp = to_timestamp(sales_req.get("actual_delivery_time", ""))

            # Convert input to NumPy array and reshape for model
            input_data = np.array([[
                float(sales_req.get("market_id", 0)), 
                created_at_timestamp, 
                actual_delivery_time_timestamp, 
                store_id_encoded,  # Target encoded store_id
                encoded_category,  # Encoded "store_primary_category" category
                float(sales_req.get("order_protocol", 0)), 
                float(sales_req.get("total_items", 0)),
                float(sales_req.get("subtotal", 0)), 
                float(sales_req.get("num_distinct_items", 0)), 
                float(sales_req.get("min_item_price", 0)),
                float(sales_req.get("max_item_price", 0)), 
                float(sales_req.get("total_onshift_partners", 0)), 
                float(sales_req.get("total_busy_partners", 0)),
                float(sales_req.get("total_outstanding_orders", 0)), 
                diff_seconds/60, # Convert to minutes 
                encoded_weekday,  # Encoded weekday
            ]], dtype=np.float32)
            
            
            print("Input Data:", input_data)

            # Step 4: Convert input to NumPy array
            #features = np.array([list(input_data.values())])    
            features = input_data


            # Preprocess data (if needed)
            features_scaled = scaler.transform(features)  # Apply scaling


            #Make predictions
            result = model.predict(features_scaled)[0][0]  # Extract single value
            print("Predicted Result:", result)

            
            
            #TO GET SINGLE VALUE CODE FROM POSTMAN WE WILL UNCOMMENT ON BELOW CODE
            return jsonify({"Predicted Delivery Time": round(float(result), 2)})  # Ensure JSON serializable output    

        except ValueError as e:
             
            return jsonify({"error": f"Invalid input: {str(e)}"}), 400    

if __name__ == "__main__":
    app.run(debug=True)    
