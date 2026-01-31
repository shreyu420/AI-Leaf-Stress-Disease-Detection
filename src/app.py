import cv2
import numpy as np
import os
import streamlit as st
import torch
import timm
import torch.nn.functional as F
from torchvision import transforms
from collections import Counter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

disease_classes = ["Early_blight", "Healthy", "Leaf_Curl"]
stress_classes = ["Healthy", "Nitrogen_Deficiency"]

@st.cache_resource
def load_disease_model():
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=len(disease_classes)
    )
    model.load_state_dict(
        torch.load("models/disease_model.pth", map_location="cpu")
    )
    model.eval()
    return model

disease_model = load_disease_model()

@st.cache_resource
def load_stress_model():
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=len(stress_classes)
    )
    model.load_state_dict(
        torch.load("models/stress_model.pth", map_location="cpu")
    )
    model.eval()
    return model

stress_model = load_stress_model()

FAQ_KB = {
    "English": {
        "Early_blight": {
            "symptoms": "Brown concentric spots on older leaves.",
            "cause": "Fungal infection caused by Alternaria solani.",
            "solution": "Apply recommended fungicide and remove infected leaves.",
            "prevention": "Avoid overhead irrigation and ensure good air circulation."
        },
        "Leaf_Curl": {
            "symptoms": "Upward curling and yellowing of leaves.",
            "cause": "Virus transmitted by whiteflies.",
            "solution": "Control whiteflies and remove infected plants.",
            "prevention": "Use insect nets and resistant varieties."
        },
        "Nitrogen_Deficiency": {
            "symptoms": "Yellowing of older leaves and slow growth.",
            "cause": "Insufficient nitrogen in soil.",
            "solution": "Apply nitrogen-rich fertilizer in recommended dosage.",
            "prevention": "Regular soil testing and balanced fertilization."
        },
        "Healthy": {
            "symptoms": "Leaf appears healthy and green.",
            "cause": "No disease or stress detected.",
            "solution": "Maintain current agricultural practices.",
            "prevention": "Continue proper irrigation and nutrition."
        }
    },

    "Hindi": {
        "Early_blight": {
            "symptoms": "पुराने पत्तों पर भूरे गोल धब्बे दिखाई देते हैं।",
            "cause": "यह Alternaria solani नामक फफूंद से होता है।",
            "solution": "उपयुक्त फफूंदनाशक का प्रयोग करें और संक्रमित पत्तियाँ हटाएँ।",
            "prevention": "ऊपर से सिंचाई से बचें और हवा का अच्छा संचार रखें।"
        },
        "Leaf_Curl": {
            "symptoms": "पत्तियाँ ऊपर की ओर मुड़ जाती हैं और पीली हो जाती हैं।",
            "cause": "सफेद मक्खी द्वारा फैलने वाला वायरस।",
            "solution": "सफेद मक्खी को नियंत्रित करें और संक्रमित पौधे हटाएँ।",
            "prevention": "कीट-जाल और प्रतिरोधी किस्मों का उपयोग करें।"
        },
        "Nitrogen_Deficiency": {
            "symptoms": "पुराने पत्तों का पीला होना और धीमी वृद्धि।",
            "cause": "मिट्टी में नाइट्रोजन की कमी।",
            "solution": "सिफारिश अनुसार नाइट्रोजन युक्त खाद डालें।",
            "prevention": "नियमित मिट्टी परीक्षण और संतुलित उर्वरक प्रयोग करें।"
        },
        "Healthy": {
            "symptoms": "पत्ता हरा और स्वस्थ दिखाई देता है।",
            "cause": "कोई रोग या तनाव नहीं।",
            "solution": "वर्तमान कृषि पद्धतियाँ बनाए रखें।",
            "prevention": "उचित सिंचाई और पोषण जारी रखें।"
        }
    }
}


def get_expert_advice(label, question, language="English"):
    kb = FAQ_KB.get(language, FAQ_KB["English"])
    info = kb.get(label, kb["Healthy"])

    q = question.lower()

    if "symptom" in q or "लक्षण" in q:
        return info["symptoms"]
    elif "cause" in q or "कारण" in q:
        return info["cause"]
    elif "solution" in q or "treatment" in q or "उपचार" in q:
        return info["solution"]
    elif "prevent" in q or "रोकथाम" in q:
        return info["prevention"]
    else:
        return (
            f"{info['symptoms']}\n\n"
            f"{info['cause']}\n\n"
            f"{info['solution']}"
        )

def extract_frames(video_path, out_dir="frames", fps=2):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(video_fps // fps, 1)

    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            cv2.imwrite(f"{out_dir}/frame_{saved}.jpg", frame)
            saved += 1

        count += 1

    cap.release()
    return saved

def is_blurry(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def bad_brightness(img, low=40, high=220):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    return mean < low or mean > high

def has_leaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return (mask.mean() / 255) > 0.1

def is_good_frame(img):
    return (not is_blurry(img)) and (not bad_brightness(img)) and has_leaf(img)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
def aggregate_predictions(preds):
    labels = [p[0] for p in preds]
    confs = [p[1] for p in preds]
    final_label = Counter(labels).most_common(1)[0][0]
    avg_conf = sum(confs) / len(confs)
    return final_label, avg_conf

def generate_gradcam(model, input_tensor, original_img, target_layers):
    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # (H, W)

    # Convert PIL image → float32 numpy in range [0, 1]
    img_np = np.array(original_img).astype(np.float32) / 255.0

    # Ensure same spatial size
    if img_np.shape[:2] != grayscale_cam.shape:
        img_np = cv2.resize(img_np, (grayscale_cam.shape[1], grayscale_cam.shape[0]))

    cam_image = show_cam_on_image(
        img_np,
        grayscale_cam,
        use_rgb=True
    )

    return cam_image

def get_severity(confidence):
    if confidence >= 0.85:
        return "High 🔴"
    elif confidence >= 0.65:
        return "Moderate 🟠"
    else:
        return "Low 🟢"

st.set_page_config(page_title="AI Leaf Analysis", layout="centered")
st.sidebar.title("🌱 Navigation")

section = st.sidebar.radio(
    "Go to",
    ["Image Analysis", "Video Analysis", "About"]
)


st.title("🌱 AI Leaf Disease & Stress Detection")
st.write("Upload a leaf image to get diagnosis and expert advice.")

st.subheader("🌐 Select Advisory Language")

language = st.selectbox(
    "Choose Language",
    ["English", "Hindi"]
)

from PIL import Image
if section == "Image Analysis":
    uploaded_image = st.file_uploader(
        "📸 Upload a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # ---------- IMAGE PREPROCESS ----------
        img_tensor = transform(image).unsqueeze(0)

        # ---------- DISEASE PREDICTION ----------
        with torch.no_grad():
            out = disease_model(img_tensor)
            probs = F.softmax(out, dim=1)
            pred_idx = probs.argmax(dim=1).item()

        disease = disease_classes[pred_idx]
        confidence = probs[0][pred_idx].item()

        CONF_THRESHOLD = 0.7

        if confidence < CONF_THRESHOLD:
            st.warning("⚠️ Model is not confident. Please upload a clearer image.")
            st.write(f"Confidence: {confidence*100:.2f}%")

        else:
            if disease != "Healthy":
                st.subheader("🦠 Disease Diagnosis (Image)")
                severity = get_severity(confidence)
                st.subheader("📊 Analysis Result")

                col1, col2, col3 = st.columns(3)

                col1.metric("🦠 Prediction", disease)
                col2.metric("📈 Confidence", f"{confidence*100:.2f}%")
                col3.metric("🚦 Severity", severity)

                st.progress(confidence)


                final_label = disease

            else:
                st.subheader("🌿 Leaf is Healthy — Checking Stress")

                # ---------- STRESS PREDICTION ----------
                with torch.no_grad():
                    out = stress_model(img_tensor)
                    probs = F.softmax(out, dim=1)
                    idx = probs.argmax(dim=1).item()

                stress = stress_classes[idx]
                stress_conf = probs[0][idx].item()

                st.subheader("🌱 Stress Diagnosis (Image)")
                st.write(f"**Stress Type:** {stress}")
                st.write(f"**Confidence:** {stress_conf*100:.2f}%")
                final_label = stress

            # ---------- IMAGE CHATBOT ----------
            st.subheader("🤖 Expert Advisory (Image Analysis)")

            question = st.text_input(
                "Ask about symptoms, causes, treatment, prevention (image)",
                key="image_chat"
            )

            if question:
                advice = get_expert_advice(final_label, question, language)
                st.info(advice)

    st.divider()
elif section == "Video Analysis":

    st.subheader("🎥 Video-based Leaf Analysis")

    uploaded_video = st.file_uploader(
        "Upload a short leaf video (5–15 seconds)",
        type=["mp4", "mov", "avi"]
    )
    if uploaded_video:
        with open("input_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        st.success("Video uploaded successfully!")
        st.video("input_video.mp4")
        
        # Extract frames
        total = extract_frames("input_video.mp4", "frames")
        st.info(f"Extracted {total} frames")

        good_frames = []

        for f in os.listdir("frames"):
            img = cv2.imread(f"frames/{f}")
            if img is not None and is_good_frame(img):
                good_frames.append(f)

        st.success(f"Selected {len(good_frames)} good quality frames")

        # ---------- STEP 6.7: RUN AI MODELS ON FRAMES ----------
        disease_preds = []
        stress_preds = []

        for frame_name in good_frames:
            img = Image.open(f"frames/{frame_name}").convert("RGB")
            img_tensor = transform(img).unsqueeze(0)

            # -------- Disease prediction --------
            with torch.no_grad():
                out = disease_model(img_tensor)
                probs = F.softmax(out, dim=1)
                idx = probs.argmax(dim=1).item()

            disease = disease_classes[idx]
            conf = probs[0][idx].item()

            if disease != "Healthy":
                disease_preds.append((disease, conf))
            else:
                # -------- Stress prediction --------
                with torch.no_grad():
                    out = stress_model(img_tensor)
                    probs = F.softmax(out, dim=1)
                    idx = probs.argmax(dim=1).item()

                stress_preds.append(
                    (stress_classes[idx], probs[0][idx].item())
                )

        # ---------- STEP 6.8: FINAL VIDEO DECISION ----------
        if disease_preds:
            final, conf = aggregate_predictions(disease_preds)
            severity = get_severity(conf)

            st.subheader("🎥 Video Analysis Result")

            col1, col2, col3 = st.columns(3)

            col1.metric("🦠 Result", final)
            col2.metric("📈 Confidence", f"{conf*100:.2f}%")
            col3.metric("🚦 Severity", severity)

            st.progress(conf)

            final_label = final
        else:
            final, conf = aggregate_predictions(stress_preds)
            severity = get_severity(conf)

            st.warning(f"🌿 Stress Detected (Video): {final}")
            st.metric("Confidence", f"{conf*100:.2f}%")
            st.metric("Severity Level", severity)
            st.progress(conf)

            final_label = final

        # ---------- STEP 6.9: EXPERT ADVISORY (VIDEO) ----------
        st.subheader("🤖 Expert Advisory (Video Analysis)")

        question = st.text_input(
            "Ask about symptoms, causes, treatment, prevention (video)"
        )

        if question:
            advice = get_expert_advice(final_label, question, language)
            st.info(advice)
        st.divider()
        st.subheader("🖼️ Review Extracted Frames (Optional)")
        if len(good_frames) == 0:
            st.warning("No good-quality frames available for manual review.")
        else:
            st.write("Select a frame if you want detailed single-image analysis.")
        selected_frame = st.selectbox(
            "Choose a frame for detailed analysis (optional)",
            ["None"] + good_frames
        )
        if selected_frame != "None":
            st.subheader("🔍 Detailed Analysis on Selected Frame")

            frame_img = Image.open(f"frames/{selected_frame}").convert("RGB")
            st.image(frame_img, caption=f"Selected Frame: {selected_frame}", use_container_width=True)

            img_tensor = transform(frame_img).unsqueeze(0)

            # ---- Disease prediction ----
            with torch.no_grad():
                out = disease_model(img_tensor)
                probs = F.softmax(out, dim=1)
                pred_idx = probs.argmax(dim=1).item()

            disease = disease_classes[pred_idx]
            conf = probs[0][pred_idx].item()

            if disease != "Healthy":
                st.subheader("🦠 Disease Diagnosis (Selected Frame)")
                severity = get_severity(conf)

                st.subheader("🖼️ Selected Frame Result")

                col1, col2, col3 = st.columns(3)

                col1.metric("🦠 Prediction", disease)
                col2.metric("📈 Confidence", f"{conf*100:.2f}%")
                col3.metric("🚦 Severity", severity)

                st.progress(conf)

                st.metric("Severity Level", severity)

                st.progress(conf)

                final_label_manual = disease
            else:
                st.subheader("🌿 Leaf is Healthy — Checking Stress")

                with torch.no_grad():
                    out = stress_model(img_tensor)
                    probs = F.softmax(out, dim=1)
                    idx = probs.argmax(dim=1).item()

                stress = stress_classes[idx]
                stress_conf = probs[0][idx].item()
                st.subheader("🌱 Stress Diagnosis (Selected Frame)")
                severity = get_severity(stress_conf)
                st.metric("Severity Level", severity)

                final_label_manual = stress

                st.subheader("🧠 Explainable AI – Grad-CAM Visualization")

            target_layers = [disease_model.conv_head]

            cam_image = generate_gradcam(
            model=disease_model,
            input_tensor=img_tensor,
            original_img=frame_img,
            target_layers=target_layers
        )

            st.image(
            cam_image,
            caption="Grad-CAM: Regions influencing the model decision",
            use_container_width=True
        )

        st.subheader("🤖 Expert Advisory (Selected Frame)")

        q_manual = st.text_input(
                "Ask about symptoms, causes, treatment, prevention (selected frame)",
                key="manual_frame_chat"
            )

        if q_manual:
                advice = get_expert_advice(final_label_manual, q_manual, language)
                st.info(advice)

elif section == "About":
    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### 🌱 AI-Based Leaf Disease & Stress Detection System

    This project is an **AI-powered decision support system for agriculture**
    that detects crop diseases and nutritional stress using **image and video analysis**.

    **Key Highlights:**
    - Multi-frame video analysis
    - Human-in-the-loop frame selection
    - Explainable AI (Grad-CAM)
    - Severity estimation
    - Multilingual expert advisory chatbot

    **Purpose:**
    Early detection helps farmers reduce crop loss, optimize fertilizer use,
    and improve overall yield.

    ⚠️ *This system is a decision-support tool and does not replace agronomist advice.*
    """)
