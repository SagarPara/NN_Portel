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

print("Target Encoder Mapping Type:", type(target_encoder.mapping))  
print("Encoders loaded:", encoders.keys())  # Check what was loaded
print("Target Encoder Mapping:", target_encoder.mapping)  # Verify contents

print("Target Encoder Mapping Keys:", target_encoder.mapping.keys())
print("First few values in target_encoder.mapping['store_id']:", target_encoder.mapping["store_id"].head())
print("Type of target_encoder.mapping['store_id']:", type(target_encoder.mapping["store_id"]))


print("LabelEncoder Classes (Unique Categories):", category_encoder.classes_)  # Check unique categories
print("LabelEncoder Classes (Unique Categories):", weekday_encoder.classes_)  # Check unique categories



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

        sales_req = request.form

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
            
            # Ensure `store_id` is an integer (fallback to -999 for errors)
            try:
                store_id = int(store_id)  # Convert to integer
            except ValueError:
                store_id = -999  # Fallback value for invalid store_id
            
            print("store_id received:", store_id)
            
            
            
            '''
            # Convert store_id to Dataframe before transformation
            store_id_df = pd.DataFrame({"store_id": [store_id]})
            print("store_id DataFrame:\n", store_id_df)

            if store_id in target_encoder.mapping.keys():
                #store_id_encoded = target_encoder.transform(store_id_df).iloc[0, 0]
                store_id_encoded = target_encoder.transform(store_id_df)["store_id"].values[0]
            else:
                store_id_encoded = 0  # Default value for unseen categories
            '''


            # Check if the store_id exists in the target encoder mapping
            if store_id in target_encoder.mapping["store_id"].index:
                store_id_encoded = target_encoder.mapping["store_id"].loc[store_id]
            else:
                store_id_encoded = target_encoder.mapping["store_id"].mean()  # Use mean as a fallback

            print("Encoded store_id:", store_id_encoded)  # Debugging print



            print("Type of target_encoder:", type(target_encoder))
            print("target_encoder mapping keys:", target_encoder.mapping.keys())
    
            
            print("Known store_id values in encoder:", target_encoder.mapping.keys())
            print("Target Encoder Type:", type(target_encoder))
            print("Target Encoder Mapping Keys:", target_encoder.mapping.keys())
            print("Encoded store_id:", store_id_encoded)





            
            # Encode store_primary_category
            category = sales_req.get("store_primary_category", "")
            encoded_category = category_encoder.transform([category])[0] if category in category_encoder.classes_ else -1
            print("Encoded store_primary_category:", encoded_category)


            # Encode weekday
            weekday = sales_req.get("weekday", "").strip() # remove extra spaces
            weekday = weekday.capitalize() # ensure first letter is uppercase


            encoded_weekday = weekday_encoder.transform([weekday])[0] if weekday in weekday_encoder.classes_ else -1
            print("Normalized Weekday:", weekday)
            print("Encoded Weekday:", encoded_weekday)

            def extract_date_components(date_str):
            #"""Helper function to extract day, month, and year from a date string."""
                if date_str:
                    try:
                        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")
                        return dt.day, dt.month, dt.year
                    except ValueError:
                        return 0, 0, 0  # Default values if parsing fails
                return 0, 0, 0  # Default if no date is provided

            created_at_timestamp = sales_req.get("created_at", "")
            actual_delivery_time_timestamp = sales_req.get("actual_delivery_time", "")

            created_day, created_month, created_year = extract_date_components(created_at_timestamp)
            delivery_day, delivery_month, delivery_year = extract_date_components(actual_delivery_time_timestamp)

            
            # Convert input to NumPy array and reshape for model
            input_data = np.array([[
                float(sales_req.get("market_id", 0)), 
                #created_at_timestamp, 
                #actual_delivery_time_timestamp, 
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
                #diff_seconds/60, # Convert to minutes 
                encoded_weekday,
                created_day, 
                created_month, 
                created_year,  # Encoded weekday
            ]], dtype=np.float32)
            
            
            print("Input Data:", input_data)
            
            # Preprocess data (if needed)
            features_scaled = scaler.transform(input_data)  # Apply scaling


            #Make predictions
            result = model.predict(features_scaled)[0][0]  # Extract single value
            print("Predicted Result:", result)

            
            # Render the result.html template and pass data
            return render_template("result.html", 
                market_id=sales_req["market_id"],
                created_at=sales_req["created_at"],
                actual_delivery_time=sales_req["actual_delivery_time"],
                store_id=sales_req["store_id"],
                store_primary_category=sales_req["store_primary_category"],
                order_protocol=sales_req["order_protocol"],
                total_items=sales_req["total_items"],
                subtotal=sales_req["subtotal"],
                num_distinct_items=sales_req["num_distinct_items"],
                min_item_price=sales_req["min_item_price"],
                max_item_price=sales_req["max_item_price"],
                total_onshift_partners=sales_req["total_onshift_partners"],
                total_busy_partners=sales_req["total_busy_partners"],
                total_outstanding_orders=sales_req["total_outstanding_orders"],
                diff=sales_req["diff"],
                weekday=sales_req["weekday"],
                result=result,
                round=round)  # Send prediction result
            




            '''
            #TO GET SINGLE VALUE CODE FROM POSTMAN WE WILL UNCOMMENT ON BELOW CODE
            return jsonify({"Predicted Delivery Time": round(float(result), 2)})  # Ensure JSON serializable output    
            '''
        except ValueError as e:
             
            return f"Error: {e}", 500
            
            



if __name__ == "__main__":
    app.run(debug=True)    
