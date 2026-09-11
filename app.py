import streamlit as st
import numpy as np
from PIL import Image
import random
import time
import os

# Page config
st.set_page_config(
    page_title="AI Face Emotion Game",
    page_icon="😊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .challenge-card {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .emotion-display {
        font-size: 5rem;
        text-align: center;
    }
    .score-card {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        border: 2px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_game():
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'round' not in st.session_state:
        st.session_state.round = 1
    if 'target_emotion' not in st.session_state:
        st.session_state.target_emotion = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0
    if 'game_complete' not in st.session_state:
        st.session_state.game_complete = False
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'all_emotions' not in st.session_state:
        st.session_state.all_emotions = {}
    if 'score_awarded' not in st.session_state:
        st.session_state.score_awarded = False
    if 'prediction_error_detail' not in st.session_state:
        st.session_state.prediction_error_detail = None

# Select target emotion
def select_target_emotion():
    emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']
    return random.choice(emotions)

# Path to the sample images folder
SAMPLE_IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")

# Load available sample images grouped by emotion
def get_sample_images():
    samples = {}
    if not os.path.isdir(SAMPLE_IMG_DIR):
        return samples

    emotion_labels = {
        'happy': '😀 Happy',
        'sad': '😢 Sad',
        'angry': '😠 Angry',
        'surprise': '😮 Surprise',
        'neutral': '😐 Neutral',
        'fear': '😨 Fear',
        'disgust': '🤢 Disgust'
    }

    for fname in sorted(os.listdir(SAMPLE_IMG_DIR)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
        base = fname.split('_')[0].lower()
        if base in emotion_labels:
            label = f"{emotion_labels[base]} - {fname}"
            samples[label] = os.path.join(SAMPLE_IMG_DIR, fname)

    return samples

# Load sample images grouped by emotion for a compact picker
def get_sample_image_groups():
    groups = {}
    samples = get_sample_images()
    for label, path in samples.items():
        emotion = label.split(' - ')[0]
        groups.setdefault(emotion, []).append(path)
    return groups

# Select target emotion

# Get emotion with emoji
def get_emotion_emoji(emotion):
    emojis = {
        'happy': '😀',
        'sad': '😢',
        'angry': '😠',
        'surprise': '😮',
        'neutral': '😐',
        'fear': '😨',
        'disgust': '🤢'
    }
    return emojis.get(emotion, '😐')

def get_emotion_display(emotion):
    return f"{get_emotion_emoji(emotion)} {emotion.capitalize()}"

# Predict emotion from image - Uses DeepFace for high accuracy
def predict_emotion(image):
    import cv2

    img_array = np.array(image)

    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Phone / camera photos are often extremely large. The face detector
    # works best when the longest side is about 1200 px, so downscale first -
    # this is the difference between "no face found" and a real prediction.
    h, w = img_array.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img_array = cv2.resize(
            img_array,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    st.session_state.prediction_error_detail = None

    try:
        from deepface import DeepFace

        result = DeepFace.analyze(
            img_path=img_array,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=True,
            silent=True
        )
    except Exception as e:
        detail = str(e)
        st.session_state.prediction_error_detail = detail
        if "Face could not be detected" in detail or "FaceNotDetected" in type(e).__name__:
            return None, 0, {}, "no_face"
        return None, 0, {}, "model_error"

    if isinstance(result, list):
        if not result:
            return None, 0, {}, "no_face"
        if len(result) > 1:
            return None, 0, {}, "multiple_faces"
        result = result[0]

    emotions = result.get('emotion', {})

    emotion_map = {
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'surprise': 'surprise',
        'neutral': 'neutral',
        'fear': 'fear',
        'disgust': 'disgust'
    }

    mapped_emotions = {}
    for key, value in emotions.items():
        if key.lower() in emotion_map:
            mapped_emotions[emotion_map[key.lower()]] = float(value)

    if not mapped_emotions:
        return None, 0, {}, "model_error"

    total = sum(mapped_emotions.values())
    if total > 0:
        mapped_emotions = {k: v/total for k, v in mapped_emotions.items()}

    top_emotion = max(mapped_emotions, key=mapped_emotions.get)
    confidence = mapped_emotions[top_emotion] * 100

    # Honest uncertainty gate: if the scores are very flat (top emotion only
    # slightly ahead) or the overall confidence is very low, the AI is not
    # sure. Better to say "not sure" than to force a wrong answer.
    sorted_emotions = sorted(mapped_emotions.items(), key=lambda x: x[1], reverse=True)
    margin = (sorted_emotions[0][1] - sorted_emotions[1][1]) * 100 if len(sorted_emotions) > 1 else 100
    if confidence < 25 or margin < 5:
        return top_emotion, confidence, mapped_emotions, "uncertain"

    return top_emotion, confidence, mapped_emotions, "success"

def capture_local_camera():
    import cv2

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        if not camera.isOpened():
            return None
        success, frame = camera.read()
        if not success:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        camera.release()

# Check answer
def check_answer(target, prediction):
    return target == prediction

# Get fun reaction
def get_fun_reaction(emotion):
    reactions = {
        'happy': '😀 Wow! That\'s a happy expression!',
        'sad': '💙 It looks like a sad expression. Keep smiling!',
        'angry': '😮 That\'s a strong angry expression!',
        'surprise': '😮 Wow! You look surprised!',
        'neutral': '😐 A calm and neutral expression!',
        'fear': '😨 That looks like a scared expression!',
        'disgust': '🤢 That looks like a disgusted expression!'
    }
    return reactions.get(emotion, '🤔 Interesting expression!')

# Show prediction result
def show_prediction(prediction, confidence, all_emotions):
    st.markdown("---")
    st.markdown("## 🤖 AI SAYS...")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"<div class='emotion-display'>{get_emotion_emoji(prediction)}</div>", unsafe_allow_html=True)
        st.markdown(f"### {prediction.capitalize()}")

    with col2:
        st.markdown("### Confidence:")
        st.progress(int(confidence))
        st.markdown(f"### {confidence:.1f}%")

    st.markdown("---")

    if all_emotions:
        st.markdown("### 📊 All Predictions:")
        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_emotions:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.markdown(f"**{get_emotion_emoji(emotion)} {emotion.capitalize()}**")
            with col2:
                st.progress(int(score * 100))
            with col3:
                st.markdown(f"**{score*100:.1f}%**")

# Calculate final score message
def get_score_message(score):
    if score >= 45:
        return "🏆 Amazing! You are an Emotion Master!"
    elif score >= 30:
        return "🌟 Great job! You did very well!"
    elif score >= 20:
        return "👍 Good try! Keep practicing!"
    else:
        return "😊 Keep trying! You will get better!"

# Main app
def main():
    initialize_game()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎮 Game Controls")
        presentation_mode = st.checkbox("🎓 Presentation Mode")

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.markdown(f"**Round:** {st.session_state.round} / 5")
        st.markdown(f"**Score:** {st.session_state.score} / 50")

        st.markdown("---")
        st.markdown("### 🖼️ Sample Images")
        sample_groups = get_sample_image_groups()
        if sample_groups:
            sample_emotion = st.selectbox(
                "Pick an emotion to test:",
                ["-- None --"] + list(sample_groups.keys())
            )
            if sample_emotion != "-- None --":
                paths = sample_groups[sample_emotion]
                st.caption(f"**{len(paths)}** sample photos available.")
                if st.button("🎲 Show Random Sample"):
                    st.session_state.sample_path = random.choice(paths)
                if 'sample_path' in st.session_state:
                    thumbnail = Image.open(st.session_state.sample_path)
                    st.image(thumbnail, caption=f"🎲 Random {sample_emotion} Sample", width="stretch")
                    if st.button("📥 Use This Sample", key="use_sample"):
                        st.session_state.uploaded_image = Image.open(st.session_state.sample_path)
                        st.session_state.analyzed = False
                        st.session_state.prediction = None
                        st.session_state.confidence = 0
                        st.session_state.all_emotions = {}
                        st.rerun()
        else:
            st.info("No sample images found. Add face photos to the 'image' folder.")

        st.markdown("---")
        if st.button("🔄 Restart Game"):
            st.session_state.score = 0
            st.session_state.round = 1
            st.session_state.target_emotion = None
            st.session_state.prediction = None
            st.session_state.confidence = 0
            st.session_state.game_complete = False
            st.session_state.analyzed = False
            st.session_state.uploaded_image = None
            st.session_state.all_emotions = {}
            st.session_state.score_awarded = False
            st.rerun()

    # Presentation mode
    if presentation_mode:
        st.markdown("# 🎓 My Project Explanation")
        st.markdown("""
        **My project is called AI Face Emotion Game.**

        The user uploads a picture of a face.

        The AI looks at the facial expression and predicts an emotion.

        The game gives the player a target emotion, such as Happy.

        If the AI detects the same expression, the player gets 10 points.

        There are five rounds.

        I used Python and Streamlit to create my project.

        I learned about AI, image processing, conditions, functions and game logic.
        """)
        st.markdown("---")

    # Main header
    st.markdown("""
    <div class='main-header'>
        <h1>😊🤖 AI FACE EMOTION GAME</h1>
        <p><em>"Show a reaction. Upload a photo. Let AI guess!"</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Dashboard
    st.markdown("## 📊 Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Current Round", f"{st.session_state.round} / 5")
    with col2:
        st.metric("⭐ Score", st.session_state.score)
    with col3:
        st.metric("🏆 Max Possible", 50)

    st.markdown("---")

    # Game complete
    if st.session_state.game_complete:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 15px; color: white; margin: 1rem 0;'>
            <h1>🎉 GAME COMPLETE! 🎉</h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🏆 Your Score")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"### {st.session_state.score} / 50")
        with col2:
            st.markdown(f"### {st.session_state.score // 10} / 5 Correct")
        with col3:
            accuracy = (st.session_state.score / 50) * 100
            st.markdown(f"### {accuracy:.0f}% Accuracy")

        st.progress(int(st.session_state.score / 50 * 100))

        st.markdown(f"### {get_score_message(st.session_state.score)}")

        if st.button("🔄 PLAY AGAIN", key="play_again"):
            st.session_state.score = 0
            st.session_state.round = 1
            st.session_state.target_emotion = None
            st.session_state.prediction = None
            st.session_state.confidence = 0
            st.session_state.game_complete = False
            st.session_state.analyzed = False
            st.session_state.uploaded_image = None
            st.session_state.all_emotions = {}
            st.rerun()

        st.markdown("---")
        show_learning_section()
        return

    # Select target if not selected
    if st.session_state.target_emotion is None:
        st.session_state.target_emotion = select_target_emotion()

    # Challenge card
    target = st.session_state.target_emotion
    st.markdown(f"""
    <div class='challenge-card'>
        <h2>🎯 YOUR CHALLENGE</h2>
        <h3>Show me a:</h3>
        <h1 style='font-size: 4rem;'>{get_emotion_emoji(target)} {target.upper()} FACE</h1>
    </div>
    """, unsafe_allow_html=True)

    # Camera or upload area
    st.markdown("## 📷 Show Your Reaction")
    camera_file = st.camera_input("Take a picture with your camera")
    if st.button("📸 Capture from Computer Camera", key="local_camera"):
        with st.spinner("Opening your computer camera..."):
            local_image = capture_local_camera()
        if local_image is None:
            st.error("❌ Could not open the computer camera. Close other camera apps and try again.")
        else:
            st.session_state.uploaded_image = local_image
            st.session_state.analyzed = False
            st.session_state.prediction = None
            st.session_state.confidence = 0
            st.session_state.all_emotions = {}
            st.rerun()
    uploaded_file = st.file_uploader(
        "Or choose a clear face picture",
        type=['jpg', 'jpeg', 'png', 'webp'],
        key=f"uploader_{st.session_state.round}"
    )

    image_file = camera_file or uploaded_file
    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="📷 Your Camera Picture" if camera_file else "📷 Your Uploaded Picture", width="stretch")
        st.session_state.uploaded_image = image

    # Show a sample image loaded from the sidebar
    if uploaded_file is None and st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, caption="🖼️ Selected Sample Picture", width="stretch")

    # Analyze button
    if st.session_state.uploaded_image is not None and not st.session_state.analyzed:
        if st.button("🤖 ANALYZE MY FACE", key="analyze"):
            with st.spinner("🤖 AI is looking at the facial expression..."):
                time.sleep(1)
                prediction, confidence, all_emotions, status = predict_emotion(st.session_state.uploaded_image)

                if status == "no_face":
                    st.error("😕 I couldn't find a clear face in this picture. Please try another image.")
                    if st.session_state.prediction_error_detail:
                        st.caption(f"Detector detail: {st.session_state.prediction_error_detail[:200]}")
                elif status == "multiple_faces":
                    st.warning("👥 I found more than one face. Please upload a picture with one clear face.")
                elif status == "model_error":
                    st.error("❌ The AI model could not analyze this picture.")
                    if st.session_state.prediction_error_detail:
                        st.caption(f"Model detail: {st.session_state.prediction_error_detail[:300]}")
                elif status == "uncertain":
                    st.warning("🤔 Hmm, the AI is NOT SURE about this picture. The facial expression "
                               "is too unclear. Please try a clearer, brighter, single-face photo.")
                    st.markdown(f"Confidence was only **{confidence:.1f}%** on **{prediction.capitalize()}**.")
                elif status == "success":
                    st.session_state.prediction = prediction
                    st.session_state.confidence = confidence
                    st.session_state.all_emotions = all_emotions
                    st.session_state.analyzed = True

                    if confidence < 30:
                        st.warning("🤔 I'm not very sure about this prediction. Try a clearer face picture.")

                    st.rerun()
                else:
                    st.error("❌ Sorry, there was an error analyzing the image. Please try again.")

    # Show prediction if analyzed
    if st.session_state.analyzed and st.session_state.prediction is not None:
        show_prediction(st.session_state.prediction, st.session_state.confidence, st.session_state.all_emotions)

        # Check answer
        is_correct = check_answer(target, st.session_state.prediction)

        st.markdown("---")

        if is_correct:
            st.success(f"""
            ### 🎉 CORRECT!

            **Target:** {get_emotion_display(target)}

            **AI Prediction:** {get_emotion_display(st.session_state.prediction)}

            ⭐ **+10 Points**
            """)
            if not st.session_state.score_awarded:
                st.session_state.score += 10
                st.session_state.score_awarded = True
        else:
            st.warning(f"""
            ### 😄 TRY AGAIN!

            **Target:** {get_emotion_display(target)}

            **AI Prediction:** {get_emotion_display(st.session_state.prediction)}

            ⭐ **+0 Points**
            """)

        # Fun reaction
        st.info(get_fun_reaction(st.session_state.prediction))

        st.markdown("---")

        # Next round button
        st.markdown("## ➡️ NEXT ROUND")
        if st.button("➡️ Continue to Next Round", key="next_round"):
            if st.session_state.round >= 5:
                st.session_state.game_complete = True
            else:
                st.session_state.round += 1
                st.session_state.target_emotion = select_target_emotion()
                st.session_state.prediction = None
                st.session_state.confidence = 0
                st.session_state.analyzed = False
                st.session_state.uploaded_image = None
                st.session_state.all_emotions = {}
                st.session_state.score_awarded = False
            st.rerun()

    # No image uploaded message
    if st.session_state.uploaded_image is None:
        st.info("📷 Upload a face picture to start the challenge! 😊")

    st.markdown("---")
    show_learning_section()

def show_learning_section():
    # Learn about AI
    with st.expander("## 🤖 How Does This AI Work?"):
        st.markdown("""
        **The computer looks at the face in the picture and uses a model that has already learned from many example images. It looks for patterns in the facial expression and makes a prediction.**

        📷 **Image** → 👤 **Find Face** → 🤖 **AI Model** → 😊 **Expression Prediction** → 🎮 **Game Result**

        **⚠️ Important:** The AI prediction may sometimes be wrong. Facial expressions do not always show how a person truly feels. This project is only an educational demonstration of AI image classification.
        """)

    # Learning section
    st.markdown("## 🧠 What I Learned")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🐍 Python
        I used Python to build the project.

        ### 🌐 Streamlit
        I used Streamlit to create the web interface.
        """)

    with col2:
        st.markdown("""
        ### 🤖 AI
        I used a pre-trained AI model to recognize facial expressions.

        ### 🎮 Game Logic
        I used conditions and scores to create the game.
        """)

    # Presentation script
    with st.expander("## 🎤 Presentation Script"):
        st.markdown("""
        "Good morning everyone.

        My project is called AI Face Emotion Game.

        I made it using Python and Streamlit.

        First, the game gives us a target expression.

        Then we upload a picture of a face.

        The AI checks the facial expression and predicts an emotion.

        If the prediction matches the target, we get 10 points.

        There are five rounds.

        This project helped me learn about Python, AI, images and game logic.

        In the future, I can add OpenCV so the game can use a live camera.

        Thank you!"
        """)

if __name__ == "__main__":
    main()
