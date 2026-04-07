import io
import base64

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

sample_images = ""

# Load model once
model = tf.keras.models.load_model("model_epoch_07.keras")

MODEL_NAME = model.name
INPUT_SIZE = model.input_shape[1:3]  # (224,224)

@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        input_size=INPUT_SIZE,
        sample_images=sample_images
    )

@app.route("/predict", methods=["POST"])
def predict():
    import numpy as np
    from PIL import Image

    file = request.files.get("image")
    selected_image = None

    if file:
        file_bytes = file.read()
        img = Image.open(io.BytesIO(file_bytes)).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        encoded = base64.b64encode(file_bytes).decode("utf-8")
        mime_type = file.mimetype or "image/png"
        selected_image = {
            "data_uri": f"data:{mime_type};base64,{encoded}",
            "name": file.filename
        }
        
        print("Image shape:", img_array.shape)
        print("Min/max values:", img_array.min(), img_array.max())
    else:
        return render_template(
            "index.html",
            model_name=MODEL_NAME,
            input_size=INPUT_SIZE,
            sample_images=sample_images,
            result=None,
            selected_image=None,
        )

    # else:
    #     sample_path = request.form.get("sample_path")
    #     img = Image.open("test_images/" + sample_path).resize((224, 224))
    #     img = np.array(img) / 255.0
    #     img = np.expand_dims(img, axis=0)

    preds = model.predict(img_array)[0]

    predicted_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = preds[predicted_index]

    # Top 5 predictions
    top_indices = preds.argsort()[-5:][::-1]

    top_predictions = [
        {
            "class_name": CLASS_NAMES[i],
            "confidence": float(preds[i])
        }
        for i in top_indices
    ]
    
    return render_template(
        "index.html",   # IMPORTANT: same page
        model_name=MODEL_NAME,
        input_size=INPUT_SIZE,
        result={
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions
        },
        selected_image=selected_image,
        sample_images=sample_images
    )

@app.route("/summary")
def summary():
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))

    summary_str = "\n".join(stringlist)

    layer_type_counts = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_type_counts[layer_type] = layer_type_counts.get(layer_type, 0) + 1

    trainable_params = int(
        np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    )
    non_trainable_params = int(
        np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    )

    summary_metrics = {
        "layer_count": len(model.layers),
        "total_params": int(model.count_params()),
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }

    return render_template(
        "summary.html",
        summary=summary_str,
        summary_metrics=summary_metrics,
        layer_type_labels=list(layer_type_counts.keys()),
        layer_type_values=list(layer_type_counts.values()),
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)