# 😊 AI Face Emotion Game

### "Can AI Guess My Reaction?" 🤖

---

## 📋 Project Idea

The student uploads a face image. The AI analyzes the facial expression and predicts an emotion. Then the application displays the detected emotion, confidence percentage, fun reaction, game result, and score.

**⚠️ Important:** This is an EDUCATIONAL emotion-recognition demonstration. The AI predicts visible facial expressions - it does NOT know a person's true feelings. Facial expressions do not always show how a person truly feels.

---

## 🎯 Features

- 🎮 5-round emotion guessing game
- 📷 Image upload support (JPG, JPEG, PNG, WEBP)
- 🤖 AI emotion detection using pre-trained model
- 📊 Confidence scores and predictions
- ⭐ Scoring system (max 50 points)
- 🎨 Attractive Streamlit UI with emojis
- 🎓 Presentation mode for school exhibitions
- 🧠 Learning section explaining AI concepts
- 🔄 Restart and next round functionality

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| 🐍 Python | Programming language |
| 🌐 Streamlit | Web interface |
| 🖼️ Pillow | Image processing |
| 🧠 FER | Facial emotion recognition |
| 📸 OpenCV | Image handling (for FER) |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Required Packages

```bash
pip install -r requirements.txt
```

Use Python 3.11 or 3.12 for DeepFace/TensorFlow compatibility. Python 3.14 can run the interface, but the emotion model may not install yet.

**Alternative (if DeepFace has issues):**
```bash
pip install streamlit pillow fer opencv-python
```

**The app automatically selects the best available method for emotion detection.**

---

## ▶️ Run the Application

**Important:** The emotion model (DeepFace/TensorFlow) works with Python 3.11
or 3.12. It does NOT work with the default Python 3.14 install, which is why
the game used to show "No emotion model is installed".

**Easiest:** double-click **`run_game.bat`** (uses Python 3.12 automatically).

Or from this folder in a terminal:

```bash
py -3.12 -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

> The webcam system runs separately with `py -3.12 realtime_webcam.py`.

---

## 🎮 How to Play

### Step 1: See the Challenge
The game shows you a target emotion:
```
🎯 YOUR CHALLENGE
Show me a: 😀 HAPPY FACE
```

### Step 2: Use the Camera or Upload a Photo
Allow camera access and take a clear face picture, or upload a face picture instead.

### Step 3: Analyze
Click "🤖 ANALYZE MY FACE" button.

### Step 4: See Results
- If AI detects the same emotion: **🎉 CORRECT! +10 Points**
- If different: **😄 TRY AGAIN! +0 Points**

### Step 5: Continue
Click "➡️ Continue to Next Round" for the next challenge.

### Step 6: Final Score
After 5 rounds, see your total score and accuracy!

---

## 🤖 How AI Works

```
📷 Image
   ↓
👤 Find Face
   ↓
🤖 AI Model
   ↓
😊 Expression Prediction
   ↓
🎮 Game Result
```

The computer looks at the face in the picture and uses a model that has already learned from many example images. It looks for patterns in the facial expression and makes a prediction.

**⚠️ The AI prediction may sometimes be wrong.**

---

## 📊 Scoring System

| Result | Points |
|--------|--------|
| Correct prediction | +10 points |
| Incorrect prediction | +0 points |
| **Maximum score** | **50 points** |

### Score Messages
- **45-50:** 🏆 Amazing! You are an Emotion Master!
- **30-40:** 🌟 Great job! You did very well!
- **20-29:** 👍 Good try! Keep practicing!
- **0-19:** 😊 Keep trying! You will get better!

---

## 🐍 Python Concepts Used

| Concept | Where Used |
|---------|------------|
| Functions | Game logic, predictions, UI |
| Conditionals (if/else) | Score checking, game flow |
| Loops | Processing emotions |
| Lists | Emotion storage |
| Dictionaries | Emotion-emoji mapping |
| Random | Target emotion selection |
| Session State | Game state management |
| Image Processing | Loading and analyzing images |

---

## 📷 Camera Support

The app uses Streamlit's camera input to capture a live picture in the browser. The browser must be allowed to use the camera. The app requires DeepFace or FER for real emotion predictions; it never invents a prediction when the model is unavailable.

---

## ⚠️ Limitations

1. Works best with clear, well-lit face photos
2. May not detect faces in very dark or blurry images
3. AI predictions are not always accurate
4. Only one face should be in the image
5. Requires internet connection for model download (first run)
6. When the AI is not sure (flat or very low confidence scores) the game now
   says *"the AI is NOT SURE"* instead of forcing a wrong guess - upload a
   clearer photo and try again
7. Very large phone photos are automatically resized before analysis so the
   face detector can find the face

---

## 🔒 Privacy

- Uploaded images are used only for the current prediction
- Images are NOT saved permanently
- No database is used
- No images are sent to external services
- All processing happens locally on your computer

---

## 🎤 Presentation Script

> "Good morning everyone.
>
> My project is called AI Face Emotion Game.
>
> I made it using Python and Streamlit.
>
> First, the game gives us a target expression.
>
> Then we upload a picture of a face.
>
> The AI checks the facial expression and predicts an emotion.
>
> If the prediction matches the target, we get 10 points.
>
> There are five rounds.
>
> This project helped me learn about Python, AI, images and game logic.
>
> In the future, I can add OpenCV so the game can use a live camera.
>
> Thank you!"

---

## ⚠️ AI Safety Note

**Do NOT say:** "AI knows your feelings"

**Instead say:** "AI predicts the visible facial expression"

> Facial expressions do not always show how a person truly feels. This project is only an educational demonstration of AI image classification.

---

## 📁 Project Structure

```
AI_Face_Emotion_Game/
│
├── app.py          # Main application
└── README.md       # This file
```

---

## 🎓 Class 3 Student Info

**Name:** _______________

**Class:** 3

**Subject:** Computer Science / AI Project

**Date:** _______________

---

**Made with ❤️ for Class 3 School Project**

---

# 🎥 Real-Time Emotion Recognition System

This project also contains a **real-time facial emotion recognition system** that
runs on your computer camera. Unlike the photo-upload game, it processes live
webcam frames and reports a **stable, evidence-based** emotion instead of
guessing.

## Honest behaviour (most important rule)

The system **never guesses**. If there is not enough evidence it reports:

```json
{
  "emotion": "uncertain",
  "confidence": 0.31,
  "stable": false,
  "face_detected": true
}
```

`uncertain` is returned when any of these happen:

- No face is visible (`face_detected: false`)
- The face is too small, too blurry, too dark or too bright
- The head is turned too far sideways
- More than one face is in the frame
- The emotion model probability is not confident (`confidence` below the threshold)
- The emotion scores are too flat (ambiguous expression)

> A **low-confidence "uncertain"** is always preferred over a high-confidence
> wrong answer.

## Pipeline

```
webcam frame
  -> face detection (DeepFace / OpenCV)
  -> face quality gate (size, blur, lighting, head pose, occlusion)
  -> emotion inference (DeepFace FER model)
  -> confidence + margin calculation (softmax calibrated)
  -> temporal smoothing + hysteresis (rolling window, debounce)
  -> final structured prediction
```

## Run it

Requires Python 3.12 with the model installed (that is the environment where
DeepFace + TensorFlow are available on this PC):

```powershell
py -3.12 realtime_webcam.py            # webcam + debug overlay
py -3.12 realtime_webcam.py --debug 0  # simple overlay
py -3.12 realtime_webcam.py --camera 1 # different camera
py -3.12 realtime_webcam.py --list-cameras  # probe which cameras the PC sees
```

Keys while running: `q`/`Esc` quit, `d` toggle debug overlay, `r` reset.

**Camera diagnostics:** the app probes camera indices and OpenCV backends at
startup, verifies the camera can actually deliver a frame, and logs everything
(e.g. "CAMERA OPENED SUCCESSFULLY: index=1 backend=dshow resolution=640x480",
or a list of what was tried plus suggested actions) to `realtime_webcam.log`
next to the script. When no face is in frame the overlay says
`NO FACE DETECTED - center your face in the frame` instead of a silent blank.
If the active camera never produces a face within 10 s, the app automatically
switches to the next camera index (so a wrong/rebooting camera assignment
cannot silently break detection). `--flip` mirrors the preview cosmetically.

**Face detector:** DeepFace's default `opencv` (Haar cascade) detector is very
strict on live webcam frames and often reports no face you can clearly see.
The app uses the `ssd` detector by default (weights auto-download to
`~/.deepface/weights`, with automatic fallback to `opencv` if unavailable).
Override with `DETECTOR_BACKEND=opencv` to force Haar.

**Voice announcements:** when the committed emotion changes, the app says it
out loud using the Windows voice (SAPI) — e.g. "You look happy!". It only
speaks stable, within-camera changes (never every frame, never repeats), at
most once every `SPEAK_INTERVAL` seconds (default 2). Disable with
`--no-voice` or `SPEAK_EMOTIONS=0`.

**Responsiveness:** defaults are tuned for live use - inference runs ~6x/s
(`PREDICTION_INTERVAL=0.15`, `FRAME_SKIP=1`) and a clear emotion change
commits within about a second (`SMOOTHING_WINDOW=10`, `HYSTERESIS_RATIO=1.08`)
while still ignoring one-frame flashes.

Test a still photo (no camera needed):

```powershell
py -3.12 run_image_test.py image\happy_45342163.jpg --repeat 8
py -3.12 run_image_test.py --dir image
```

## Debug overlay

The debug panel shows the raw model emotion, smoothed/committed emotion,
confidence, face quality score and reasons, estimated head pose, smoothing
window and FPS. Use it to understand **why** a prediction is (or is not)
accepted.

## Tests

```powershell
py -3.12 test_emotion_system.py
```

Covers: neutral/smile/big-smile/sad/angry/surprise/fear/disgust outputs,
fast-transition debounce, confidence gating, no-face, multiple faces, side
face, low light, blurry frame, partial obstruction, and structured output.

## Configuration

All knobs live in `emotion_system/config.py` (no magic numbers in the code).
They can be overridden via environment variables:

| Config | Env var | Default | Meaning |
|--------|---------|---------|---------|
| `EMOTION_CONFIDENCE_THRESHOLD` | same | 0.35 | minimum confidence to report a label |
| `MARGIN_THRESHOLD` | same | 0.10 | min top1-vs-top2 gap |
| `SMOOTHING_WINDOW` | same | 15 | rolling window size |
| `STABILITY_MAJORITY` | same | 0.60 | fraction of window needed to switch |
| `HYSTERESIS_RATIO` | same | 1.15 | how much a new emotion must beat the current one |
| `MIN_FACE_SIZE` | same | 64 | minimum face pixels |
| `FRAME_SKIP` | same | 2 | infer every N frames |
| `PREDICTION_INTERVAL` | same | 0.25 | minimum seconds between inferences |
| `BLUR_THRESHOLD` | same | 20 | laplacian variance (resolution-scaled) |
| `MAX_YAW_DEG / MAX_PITCH_DEG / MAX_ROLL_DEG` | same | 45/35/30 | head-pose hard limits |
| `DEBUG_MODE` | same | 1 | debug overlay on by default |

## Files

```
emotion_system/
  config.py            all tunable settings
  emotion_utils.py     softmax, confidence, margin, entropy
  face_quality.py      blur/size/lighting/pose/occlusion checks
  temporal_smoothing.py rolling window + hysteresis
  deepface_backend.py  DeepFace wrapper + alignment
  ensemble_backend.py  DeepFace + FER+ ensemble
  hsemotion_backend.py optional eNet-B0 (AffectNet) ONNX backend
  engine.py            full pipeline orchestrator
realtime_webcam.py     live webcam loop
run_image_test.py      still-image tester
test_emotion_system.py test suite
```

## Known limitations (honest)

- **DeepFace is the weakest link.** The seven-class facial-expression model
  that ships with DeepFace is small and frequently confuses *fear*, *disgust*
  and *sadness*, and can label an ambiguous face as *neutral*. The system
  compensates with confidence gating and smoothing, but a bad raw model limits
  peak accuracy.
- **Confidence is calibrated, not perfect.** The reported probability is a
  smoothed, entropy-adjusted score. It is more honest than the raw model
  output, but it is not a guarantee of correctness.
- **Head pose is estimated from just two eye landmarks** (DeepFace gives no
  full face mesh). Yaw is derived from the eyes' horizontal offset - a good
  probe for strong side faces but not a precise pitch measurement.
- **One face at a time.** Multiple faces correctly give `uncertain`.
- **CPU only here.** Each inference is ~0.15-0.2 s on CPU (≈5-6 inference FPS).
  Frame skipping keeps the display responsive.
- **Expression ≠ feeling.** The system classifies the *visible facial
  expression*. It cannot and does not know the person's real internal emotion.
