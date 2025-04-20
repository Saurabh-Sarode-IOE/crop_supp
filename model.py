from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained ML model
model = load_model('model/tomato_disease_model.h5')

# Define labels with proper indices
disease_labels = {
    0: "Tomato Bacterial spot",
    1: "Tomato Early blight",
    2: "Tomato Late blight",
    3: "Tomato Leaf Mold",
    4: "Tomato Septoria leaf spot",
    5: "Tomato Spider mites Two-spotted spider mite",
    6: "Tomato Target Spot",
    7: "Tomato Tomato Yellow Leaf Curl Virus",
    8: "Tomato Tomato mosaic virus",
    9: "Tomato healthy"
}

disease_info = {
    "Tomato Bacterial spot": {
        "Symptoms": "Water-soaked lesions on leaves, stems, and fruit which can turn brown and necrotic, causing defoliation and fruit rot.",
        "Causes": "Bacteria Xanthomonas spp.",
        "Cure": "Use resistant cultivars, remove infected plant debris, and apply copper-based bactericides.",
        "Treatment": "CUREAL Best Fungicide & Bactericide",
        "Purchase Link": "https://agribegri.com/products/cureal---best-fungicide--bactericide-zinc-based-250-ml.php",
        "Image": "bactorial spot.png"
    },
    "Tomato Early blight": {
        "Symptoms": "Small, brown lesions on lower leaves, which can enlarge and cause defoliation and reduced yield.",
        "Causes": "Fungus Alternaria solani.",
        "Cure": "Use resistant cultivars, remove infected plant debris, and apply fungicides.",
        "Treatment": "NATIVO FUNGICIDE",
        "Purchase Link": "https://farmagritech.com/product/nativo-fungicide/",
        "Image": "early blight.png"
    },
    "Tomato Late blight": {
        "Symptoms": "Water-soaked lesions on leaves, stems, and fruit which can turn brown and necrotic, causing defoliation and fruit rot.",
        "Causes": "Fungus Phytophthora infestans.",
        "Cure": "Use resistant cultivars, remove infected plant debris, and apply fungicides.",
        "Treatment": "ACROBAT FUNGICIDE",
        "Purchase Link": "https://www.bighaat.com/products/acrobat-fungicide",
        "Image": "late blight.png"
    },
    "Tomato healthy": {
        "Symptoms": "This is a healthy Tomato leaf.",
        "Causes": "Not applicable.",
        "Cure": "Not applicable.",
        "Treatment": "Tomato Fertilizer Organic",
        "Purchase Link": "https://www.casagardenshop.com/products/tomato-fertilizer-for-home-terrace-outdoor-gardening",
        "Image": "healthy.png"
    },

    "Tomato Leaf Mold": {
        "Symptoms": "Yellowing and browning of the leaves, fuzzy greyish-brown mold may be visible on the underside.",
        "Causes": "Fungus (Fulvia fulva).",
        "Cure": "Remove infected leaves and use fungicides such as chlorothalonil or copper-based fungicides.",
        "Treatment": "TVirus Special (Set of Immuno 1 ltr + Enviro 1 ltr)",
        "Purchase Link": "https://agribegri.com/products/virus-special-enviroimmuno-1-litre.php",
        "Image": "mold.png"
    },

    "Tomato Septoria leaf spot": {
        "Symptoms": "Small brown spots with yellow halos that merge, causing leaves to die and fall off.",
        "Causes": "Fungus (Septoria lycopersici).",
        "Cure": "Remove infected leaves and use chlorothalonil or copper-based fungicides.",
        "Treatment": "Roko Fungicide",
        "Purchase Link": "https://farmagritech.com/product/roko-fungicide/?attribute_pa_size=500gm&utm_source=Google%20Shopping&utm_campaign=Google%20shopping%20feed%201&utm_medium=cpc&utm_term=3239",
        "Image": "spectrol spot.png"
    },

    "Tomato Spider mites Two-spotted spider mite": {
        "Symptoms": "Leaves develop yellow stippling, become dry, brittle, and fall off. Fine webbing may be visible.",
        "Causes": "Mites (Tetranychus urticae).",
        "Cure": "Use neem oil or pyrethrin-based insecticides.",
        "Treatment": "OMITE INSECTICIDE",
        "Purchase Link": "https://www.bighaat.com/products/omite-insecticide?variant=31276117196823&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic",
        "Image": "spider.png"
    },

    "Tomato Target Spot": {
        "Symptoms": "Brown or black spots with concentric circles on leaves, leading to leaf death.",
        "Causes": "Fungus (Corynespora cassiicola).",
        "Cure": "Remove infected leaves and apply fungicides such as chlorothalonil or copper-based fungicides.",
        "Treatment": "Propi Propineb 70% WP Fungicide for Plants Diesese Control Pesticide",
        "Purchase Link": "https://www.flipkart.com/propi-propineb-70-wp-fungicide-plants-diesese-control-pesticide/p/itm9db96656402f8?pid=SMNFMX8FJJHZAYFJ&lid=LSTSMNFMX8FJJHZAYFJDGFGXW&marketplace=FLIPKART&cmpid=content_soil-manure_730597647_g_8965229628_gmc_pla&tgi=sem,1,G,11214002,g,search,,476044024748,,,,c,,,,,,,&ef_id=Cj0KCQjwsLWDBhCmARIsAPSL3_2bIC4skU03mHTgG2GvlhsFQstQaLrFyAaL10NTTCDsuI9BoffpPFUaAjn1EALw_wcB:G:s&s_kwcid=AL!739!3!476044024748!!!g!293946777986!&gclsrc=aw.ds",
        "Image": "target spot.png"
    },

    "Tomato Yellow Leaf Curl Virus": {
        "Symptoms": "Yellowing and upward curling of the leaves, reduced growth, and misshapen fruits.",
        "Causes": "Virus (Begomovirus).",
        "Cure": "No cure. Remove infected plants and control whiteflies with insecticides.",
        "Treatment": "Syngenta Amistor Top Fungicide",
        "Purchase Link": "	https://krushikendra.com/Buy-Syngenta-Amistor-Top-Fungicide-100-ml-Online",
        "Image": "yello.png"
    },
    
    "Tomato mosaic virus": {
        "Symptoms": "Yellow mottling and mosaic patterns on leaves, reduced growth, and distorted fruits.",
        "Causes": "Virus (Tobamovirus).",
        "Cure": "No cure. Remove infected plants and control aphids with insecticides.",
        "Treatment": "	V Bind Viral Disease Special",
        "Purchase Link": "	https://agribegri.com/products/viricide-online-.php",
        "Image": "moasic.png"
    },
    
}


# Function for disease predictiona
def predict_disease(image_path):
    """
    Function to read an image, preprocess it, and predict the disease along with confidence.
    """
    print(f"🔍 Received image path: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")  # ✅ Read using PIL
    except Exception as e:
        print(f"🚨 Error: PIL failed to open image! {e}")
        return "Error", "Could not read image", "Invalid file", "Try another image", "N/A", "N/A", "static/images/default.jpg", 0.0

    image = image.resize((224, 224))  # ✅ Resize using PIL
    img = np.array(image).astype('float32') / 224.0  # ✅ Convert to numpy and normalize
    img = np.expand_dims(img, axis=0)  # ✅ Add batch dimension


    print(f"📏 Processed image shape: {img.shape}")

    # 🔮 Predict disease
    prediction = model.predict(img)
    print(f"🔮 Raw Model Prediction: {prediction}")

    predicted_label = np.argmax(prediction)
    confidence = float(np.max(prediction))  # Get confidence score

    print(f"🔢 Predicted Label Index: {predicted_label}")
    print(f"📊 Confidence Score: {confidence:.2%}")

    # Get disease details
    disease_name = disease_labels.get(predicted_label, "Unknown Disease")
    disease_details = disease_info.get(disease_name, {
        "Symptoms": "No information available.",
        "Causes": "Unknown.",
        "Cure": "No cure available.",
        "Treatment": "No treatment available.",
        "Purchase Link": "N/A",
        "Image": "static/images/default.jpg"
    })

    symptoms = disease_details["Symptoms"]
    causes = disease_details["Causes"]
    cure = disease_details["Cure"]
    treatment = disease_details["Treatment"]
    purchase_link = disease_details["Purchase Link"]
    image_url = disease_details["Image"]

    print(f"🔬 Predicted Disease: {disease_name}")
    print(f"📌 Symptoms: {symptoms}")
    print(f"🦠 Causes: {causes}")
    print(f"💊 Cure: {cure}")
    print(f"🛒 Recommended Treatment: {treatment}")
    print(f"🔗 Purchase Here: {purchase_link}")
    print(f"🖼️ Disease Image: {image_url}")
    print(f"📊 Confidence: {confidence:.2%}")
    
    return disease_name, symptoms, causes, cure, treatment, purchase_link, image_url,confidence

# Example usage
image_path = "test_images/tomato_leaf.jpg"
disease, symptoms, causes, cure, treatment, purchase_link, image_url, confidence = predict_disease(image_path)

# ✅ Print final results
print("\n📝 Final Prediction Result:")
print(f"🍅 Disease: {disease}")
print(f"📌 Symptoms: {symptoms}")
print(f"🔍 Causes: {causes}")
print(f"💊 Cure: {cure}")
print(f"🛒 Recommended Treatment: {treatment}")
print(f"🔗 Purchase Here: {purchase_link}")
print(f"🖼️ Image Path: {image_url}")
print(f"📊 Confidence: {confidence:.2%}")
