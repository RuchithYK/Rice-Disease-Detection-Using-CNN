import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set page config
st.set_page_config(page_title="Rice Leaf Disease Classifier", layout="centered")

# Load model
@st.cache_resource
def load_model():
    # Get the current directory of the script (src/)
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "RiceCNN.h5")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    return tf.keras.models.load_model(model_path)

model = load_model()

@st.cache_resource
def load_Pre_Model():
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
feature_model = load_Pre_Model()

model_directory = os.path.dirname(os.path.abspath(__file__))
model_path_1 = os.path.join(model_directory, "reference_rice")

@st.cache_resource
def load_reference_features(ref_folder=model_path_1):
    features = []
    for fname in os.listdir(ref_folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(ref_folder, fname)
            img = Image.open(img_path).convert('RGB').resize((224, 224))
            x = np.expand_dims(np.array(img), axis=0)
            x = preprocess_input(x)
            feat = feature_model.predict(x, verbose=0)
            features.append(feat)
    print(f"✅ Loaded {len(features)} reference features (cached).")
    return np.vstack(features)

rice_features = load_reference_features()




rice_classifiers = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']  

# Disease Information (for display)
disease_info = {
    "Bacterialblight": {
        "About": "Bacterial blight is caused by *Xanthomonas oryzae pv. oryzae*. It spreads through infected seeds, rainwater, and wind.",
        "Symptoms": "Water-soaked stripes from leaf tips turning yellow or white; drying of leaves in severe cases.",
        "Management": [
            "Use resistant rice varieties like IR20, IR64.",
            "Avoid over-fertilization with nitrogen.",
            "Use balanced fertilizers and proper water management.",
            "Apply copper-based bactericides if detected early."
        ]
    },
    "Blast": {
        "About": "Blast disease is caused by the fungus *Magnaporthe oryzae* and is one of the most destructive rice diseases worldwide.",
        "Symptoms": "Diamond-shaped gray lesions with brown borders on leaves, neck rot in panicles.",
        "Management": [
            "Use resistant varieties (e.g., CO 47, ADT 37).",
            "Avoid excess nitrogen fertilizers.",
            "Spray tricyclazole (0.6g/litre) or carbendazim as preventive measures.",
            "Ensure good field drainage and avoid water stagnation."
        ]
    },
    "Brownspot": {
        "About": "Brown spot is caused by the fungus *Bipolaris oryzae*. It occurs in nutrient-deficient soils, especially low nitrogen.",
        "Symptoms": "Brown circular to oval spots with yellow halo on leaves and grains.",
        "Management": [
            "Apply balanced fertilizers, especially nitrogen and potassium.",
            "Use disease-free seeds.",
            "Soak seeds in carbendazim (2g/kg) before sowing.",
            "Improve soil fertility with organic manure."
        ]
    },
    "Tungro": {
        "About": "Tungro is a viral disease transmitted by green leafhoppers (*Nephotettix virescens*).",
        "Symptoms": "Stunted growth, yellow-orange leaf discoloration, delayed maturity.",
        "Management": [
            "Use resistant varieties like ADT 36, CO 43.",
            "Control vector insects using neem oil spray or insecticides like imidacloprid.",
            "Avoid overlapping of crops to break vector life cycle.",
            "Remove and destroy infected plants immediately."
        ]
    }
}

# Image preprocessing function
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)

    # Normalize like your pre_process function
    img_array -= img_array.min()
    img_array /= (img_array.max() - img_array.min())

    # Add batch dimension
    return np.expand_dims(img_array, axis=0)



def calculate_green_ratio_pil(image_path):
    """Calculate the proportion of green pixels using PIL + NumPy."""
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

    # Green pixel condition: G > R and G > B + some margin
    green_pixels = (g > r + 20) & (g > b + 20)
    green_ratio = np.sum(green_pixels) / (img_np.shape[0] * img_np.shape[1])
    return green_ratio

def extract_features(image_path):
    """Extract CNN feature vector using MobileNetV2."""
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)
    feat = feature_model.predict(x)
    return feat.flatten()

def is_rice_leaf(image_path, green_thresh=0.15, sim_thresh=0.40):
    """Decide if image is rice leaf based on color + feature similarity."""
    green_ratio = calculate_green_ratio_pil(image_path)
    feat = extract_features(image_path)
    sims = cosine_similarity([feat], rice_features)
    avg_sim = np.mean(sims)

    print(f"\nImage: {image_path}")
    print(f"🟢 Green Ratio: {green_ratio:.2f}")
    print(f"🧠 Similarity with rice features: {avg_sim:.2f}")

    if green_ratio > green_thresh and avg_sim > sim_thresh:
        print("✅ Likely a rice leaf image!")
        return True
    elif green_ratio > 0.5 and avg_sim > 0.35:
        print("⚠️ Possibly a rice leaf (uncertain, low similarity)")
        return True
    else:
        print("❌ Not a rice leaf image.")
        return False






#Sidebar
st.sidebar.title("Dash Board")
app_mode = st.sidebar.radio("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode == "Home"):
    # Title
    st.header(" 🌾 Rice Leaf Disease Classifier")
    
    image_Path = "riceBackground.jpg"
    st.image(image_Path,use_container_width=True)
    st.markdown("""
                
        ### Created a CNN model that can classify images of the three major attacking diseases of Rice plants they are Bacterial blight, Brown spot, and Leaf smut.

        ### 1.Bacterial Blight
                
        - It is a deadly bacterial disease that is among the most destructive afflictions of cultivated rice (Oryza sativa and O. glaberrima). 
        The disease was first observed in 1884-85 in Kyushu, Japan, and the causal agent, the bacterium Xanthomonas oryzae pathovar oryzae (also referred to as Xoo), was identified in 1911,
        at that time having been named Bacillus oryzae. Thriving in warm, humid environments, bacterial blight has been observed in rice-growing regions of Asia, 
        the western coast of Africa, Australia, Latin America, and the Caribbean.

        - Bacterial blight first becomes evident as water-soaked streaks that spread from the leaf tips and margins, becoming larger and eventually releasing a milky ooze that dries into yellow droplets.
        Characteristic grayish white lesions then appear on the leaves, signaling the late stages of infection, when leaves dry out and die. In seedlings, the leaves dry out and wilt, a syndrome known as kresek.
        Infected seedlings usually are killed by bacterial blight within two to three weeks of being infected; adult plants may survive, though rice yield and quality are diminished.

        - Since rice paddies are flooded throughout most of the growing season, Xoo may easily spread among crops, bacteria travel through the water from infected plants to the roots and leaves of neighboring rice plants.
        Wind and water may also help spread Xoo bacteria to other crops and rice paddies. Various mechanisms of disease, including quorum sensing and biofilm formation, have been observed in rice bacterial blight and Xoo. 
        In addition to rice, Xoo may infect other plants, such as rice cut-grass (Leersia oryzoides), Chinese sprangletop (Leptochloa chinensis), and common grasses and weeds. 
        In nongrowing seasons, Xoo may survive in rice seeds, straw, other living hosts, water, or, for brief periods, soil.

        ### 2.Brown Spot
                
        - The brown spot has been historically largely ignored as one of the most common and most damaging rice diseases.

        - It is a fungal disease that infects the coleoptile, leaves, leaf sheath, panicle branches, glumes, and spikelets. Its most observable damage is the numerous big spots on the leaves which can kill the whole leaf. 
        When infection occurs in the seed, unfilled grains or spotted or discolored seeds are formed.

        - The disease can develop in areas with high relative humidity (86-100%) and temperature between 16 and 36°C. It is common in unflooded and nutrient-deficient soil, or in soils that accumulate toxic substances.
        For infection to occur, the leaves must be wet for 8-24 hours.
        The fungus can survive in the seed for more than four years and can spread from plant to plant through air. Major sources of brown spots in the field include

        - Brown spots can occur at all crop stages, but the infection is most critical during maximum tillering up to the ripening stages of the crop. It also causes both quantity and quality losses.
        On average, the disease causes a 5% yield loss across all lowland rice production in South and Southeast Asia. Severely infected fields can have as high as 45% yield loss.
        Heavily infected seeds cause seedling blight and lead to (10-58%) seedling mortality. It also affects the quality and the number of grains per panicle and reduces the kernel weight.
        In terms of history, the Brown Spot was considered to be the major factor contributing to the Great Bengal Famine of 1943.

        ### 3.Leaf blast
                
        - Leaf blast in rice, caused by the fungus Magnaporthe oryzae (also known as Pyricularia oryzae), is one of the most destructive diseases affecting rice crops globally.
        It primarily targets the leaves but can also spread to other aerial parts of the plant, including the collar, nodes, neck, and panicles.
        Initial symptoms appear as white to gray-green lesions with dark brown borders, which later develop into spindle-shaped or diamond-shaped spots. 
        
        - These lesions can enlarge and coalesce, leading to complete leaf death, especially in young seedlings and during the tillering stage. Severe infections reduce the leaf area available for photosynthesis, ultimately lowering grain yield.
        The disease thrives under cool daytime temperatures, frequent rain showers, high humidity, and low soil moisture—conditions common in upland rice fields. Dew formation due to large day-night temperature differences also promotes fungal growth. 
        The fungus spreads via airborne spores and can survive on infected plant debris, straw, and seeds.

        - The disease thrives under cool daytime temperatures, frequent rain showers, high humidity, and low soil moisture—conditions common in upland rice fields. Dew formation due to large day-night temperature differences also promotes fungal growth. 
        The fungus spreads via airborne spores and can survive on infected plant debris, straw, and seeds.
                
        ### 4.Tungro Disease
                
        - Tungro is a viral disease complex in rice caused by two viruses: Rice Tungro Bacilliform Virus (RTBV) and Rice Tungro Spherical Virus (RTSV). 
        It is transmitted primarily by the green leafhopper (Nephotettix virescens), which acquires the virus by feeding on infected plants and spreads it to healthy ones.
                
        - The disease affects all growth stages of rice but is most severe during the vegetative and tillering stages. Infected plants exhibit stunted growth, reduced tillering, and delayed flowering. Leaves show yellow to orange discoloration, starting from the tip and progressing downward. 
        Rust-colored spots and mottling may appear, and panicles may be small, sterile, or poorly filled.
        
        ### Overall Analysis
                
        - Each disease has distinct symptoms and impacts on rice plants, affecting their growth, photosynthesis, and overall yield. 
        Effective disease management strategies include using disease-resistant rice varieties, implementing proper field hygiene practices, and avoiding conditions that promote disease spread. 
        
        - Early detection and prompt intervention are crucial to minimizing the economic and agronomic losses caused by these diseases.
        The classification model developed in this project will contribute to early disease detection and timely management, helping to ensure healthier rice crops and higher yields. 
        Remember that accurate disease diagnosis requires the expertise of plant pathologists. This analysis serves as a general overview, and specific diagnosis should involve consultation with experts in the field. """)
    
#about page
elif(app_mode=="About"):
    st.header("About the Project")
    st.markdown("""
    ### 🌾 Rice Leaf Disease Classification using CNN

    This project is designed to assist **farmers and agricultural researchers** by automatically identifying **diseases in rice leaves** using **Deep Learning**.  
    It leverages **Convolutional Neural Networks (CNNs)** trained on images of rice leaves affected by common diseases.

    ### 🔍 Objective
    The main goal is to provide a **fast, accurate, and user-friendly tool** to detect diseases in rice plants, allowing for **early diagnosis** and **better crop management**.

    ### 🧠 Model Overview
    - Built using **TensorFlow** and **Keras** frameworks.  
    - Model trained on four major classes of rice leaf conditions:  
      - Bacterial Blight  
      - Brown Spot  
      - Leaf Blast  
      - Tungro  
    - Input images are resized to **256x256 pixels** and normalized before being passed to the CNN model.

    ### ⚙️ Technologies Used
    - **Python** for backend model development  
    - **TensorFlow / Keras** for CNN model training  
    - **Streamlit** for the interactive web interface  
    - **NumPy & PIL** for image preprocessing  
    

    ### 💡 Features
    - Simple and clean web interface for uploading images.  
    - Provides detailed information about the predicted disease.  
    - Includes **management and prevention tips** for each disease.  
    - Can be deployed locally or hosted online (e.g., Streamlit Cloud, Render, or Hugging Face Spaces).

    ### 🎯 Impact
    This project aims to help farmers **reduce crop losses** by identifying diseases at early stages and taking preventive measures.  
    It also showcases the power of **Machine Learning in agriculture** to support smart farming practices.

    ---
    ### **Team Members:** 
    - Ruchith Y K - [4MC22CS127]
    - Samrudh H M - [4MC22CS133]
    - Sarvagna Y V - [4MC22CS135]
      
    **Supervised by:** Dr. B Ramesh  
    **Institution:** Malnad College of Engineering, Hassan, Karnataka  
    **Year:** 2025
    """)
    


#prediction Page

elif(app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    st.write("Upload a rice leaf image to predict the disease class.")
    # Upload image
    uploaded_file = st.file_uploader("CHOOSE AN IMAGE:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if(st.button("Predict")):
            with st.spinner("Please Wait.."):

                if(is_rice_leaf(uploaded_file)):
                    # Preprocess and predict
                    img_input = preprocess_image(image)
                    prediction = model.predict(img_input)
                    predicted_class = rice_classifiers[np.argmax(prediction)]
                    predicted_index = np.argmax(prediction)
                    confidence_score = prediction[0][predicted_index]
                    # Display result
                    st.success(f"### Predicted Class: `{predicted_class}`")
                    st.success(f"### Confidence: {round(confidence_score*100)}%")
                    st.progress(int(confidence_score*100))
                    # Show detailed info for predicted class
                    info = disease_info[predicted_class]
                    st.markdown("---")
                    st.subheader("📋 Disease Details")
                    st.write(f"**About:** {info['About']}")
                    st.write(f"**Symptoms:** {info['Symptoms']}")
                    st.markdown("**🧑‍🌾 Management & Prevention Tips:**")
                    for tip in info['Management']:
                        st.markdown(f"- {tip}")
                else:
                    st.error("### Not a Rice Leaf Image")  



