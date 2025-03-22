import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
#from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import get_custom_objects

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
model.summary






app = Flask(__name__)

#lets create API endpoints
@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/submit", methods=["GET", "POST"])
def submit_form():
    if request.method == "GET":
        return "I will make the prediction but this on GET method"
    else:
        
        sales_req = request.get_json(force=True)  # Force ensures JSON parsing

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
            
        
        
        # Extract values and convert to NumPy array
        input_data = np.array([[sales_req[field] for field in required_fields]])
        print("Input Data:", input_data)
        

        #Make predictions
        result = model.predict(input_data)[0][0]  # Extract single value
        print("Predicted Result:", result)

        
        #store data for rendering in template
        cont = sales_req
        cont["predicted_delivery_time"] = round(result, 0)
        return render_template("result.html", **cont, round=round) # Render the result.html page with data
        
        

if __name__ == "__main__":
    app.run(debug=True)    
