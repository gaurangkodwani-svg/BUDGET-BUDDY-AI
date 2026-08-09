"""BudgetBuddy AI - Streamlit financial dashboard."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import cv2
except ImportError:
    cv2 = None

import eda_numpy
import load_data
import ml_anomaly
import ml_classifier
import plot_matplotlib
import report_export

# ==================== FACE AUTHENTICATION MODULE ====================

FACE_DB_DIR = Path("face_database")
FACE_DB_DIR.mkdir(exist_ok=True)

CHART_DIR = Path("outputs/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)

FACE_METADATA_FILE = FACE_DB_DIR / "metadata.json"
FACE_RECOGNIZER_MODEL_FILE = FACE_DB_DIR / "recognizer.yml"
FACE_LABEL_MAP_FILE = FACE_DB_DIR / "label_map.pkl"
CASCADE_PATH = FACE_DB_DIR / "haarcascade_frontalface_default.xml"

# Initialize face cascade
FACE_CASCADE = None
if cv2 is not None and CASCADE_PATH.exists():
    try:
        FACE_CASCADE = cv2.CascadeClassifier(str(CASCADE_PATH))
    except Exception as e:
        st.error(f"Failed to initialize face detection: {e}")


def load_metadata() -> dict:
    """Load face metadata from JSON file."""
    if FACE_METADATA_FILE.exists():
        with open(FACE_METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"users": {}}


def save_metadata(metadata: dict) -> None:
    """Save face metadata to JSON file."""
    with open(FACE_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def detect_face(image: np.ndarray) -> Optional[tuple]:
    """Detect face in image and return (face_roi, (x, y, w, h))."""
    if FACE_CASCADE is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        return face_roi, (x, y, w, h)
    return None


def train_recognizer() -> Optional[cv2.face_LBPHFaceRecognizer]:
    """Train LBPH face recognizer on registered faces."""
    if cv2 is None:
        return None
    metadata = load_metadata()
    if len(metadata.get("users", {})) == 0:
        return None

    faces = []
    labels = []
    label_map = {}

    for idx, (user_id, user_data) in enumerate(metadata["users"].items()):
        label_map[idx] = user_id
        for face_file in user_data.get("face_files", []):
            face_path = FACE_DB_DIR / face_file
            if face_path.exists():
                face_img = cv2.imread(str(face_path), cv2.IMREAD_GRAYSCALE)
                if face_img is not None:
                    face_img = cv2.resize(face_img, (200, 200))
                    faces.append(face_img)
                    labels.append(idx)

    if len(faces) < 1:
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    recognizer.save(str(FACE_RECOGNIZER_MODEL_FILE))
    with open(FACE_LABEL_MAP_FILE, 'wb') as f:
        pickle.dump(label_map, f)

    return recognizer


def load_recognizer() -> Optional[tuple]:
    """Load trained recognizer and label map."""
    if not FACE_RECOGNIZER_MODEL_FILE.exists() or not FACE_LABEL_MAP_FILE.exists():
        return None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(FACE_RECOGNIZER_MODEL_FILE))
        with open(FACE_LABEL_MAP_FILE, 'rb') as f:
            label_map = pickle.load(f)
        return recognizer, label_map
    except Exception:
        return None


def register_face(username: str, face_roi: np.ndarray) -> bool:
    """Register a new face for a user. Returns True on success, False on failure."""
    try:
        FACE_DB_DIR.mkdir(parents=True, exist_ok=True)
        metadata = load_metadata()

        if username in metadata["users"]:
            user_data = metadata["users"][username]
            face_count = len(user_data.get("face_files", []))
        else:
            user_data = {"face_files": []}
            face_count = 0
            metadata["users"][username] = user_data

        face_filename = f"{username}_{face_count:03d}.jpg"
        face_path = FACE_DB_DIR / face_filename
        cv2.imwrite(str(face_path), face_roi)

        user_data["face_files"].append(face_filename)
        user_data["registered_at"] = str(pd.Timestamp.now())

        save_metadata(metadata)
        train_recognizer()

        return True
    except Exception as e:
        st.error(f"Registration failed: {e}")
        return False


def recognize_face(face_roi: np.ndarray, confidence_threshold: float = 70.0) -> Optional[str]:
    """Recognize a face and return username if confident."""
    recognizer_data = load_recognizer()
    if recognizer_data is None:
        return None

    recognizer, label_map = recognizer_data
    label, confidence = recognizer.predict(face_roi)

    if confidence < confidence_threshold:
        return label_map.get(label)
    return None


def delete_user(username: str) -> bool:
    """Delete a registered user and their faces."""
    metadata = load_metadata()
    if username not in metadata["users"]:
        return False

    for face_file in metadata["users"][username].get("face_files", []):
        face_path = FACE_DB_DIR / face_file
        if face_path.exists():
            face_path.unlink()

    del metadata["users"][username]
    save_metadata(metadata)

    if metadata["users"]:
        train_recognizer()
    else:
        if FACE_RECOGNIZER_MODEL_FILE.exists():
            FACE_RECOGNIZER_MODEL_FILE.unlink()
        if FACE_LABEL_MAP_FILE.exists():
            FACE_LABEL_MAP_FILE.unlink()

    return True


def capture_face_from_webcam(instruction: str = "Position your face in the frame", key: str = "face_capture") -> Optional[np.ndarray]:
    """Capture a face using Streamlit's built-in camera widget."""
    if cv2 is None:
        st.error("OpenCV not available for face detection.")
        return None

    if FACE_CASCADE is None:
        st.error("Face detection model not loaded.")
        return None

    st.info(instruction)
    camera_image = st.camera_input("📸 Capture your face", key=key)

    if camera_image is not None:
        file_bytes = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            result = detect_face(image)
            if result:
                face_roi, _ = result
                return face_roi
            else:
                st.warning("⚠️ No face detected. Please try again with better lighting and positioning.")

    return None


def render_register_page():
    """Render the face registration page."""
    st.markdown("""
        <div class="hero" style="text-align:center; padding:3rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;">📝</div>
            <h1 style="color:var(--text-primary);font-size:2.5rem;font-weight:800;margin-bottom:.5rem;">
                Register Your Face
            </h1>
            <p style="color:var(--text-secondary);font-size:1.2rem;">
                Capture your face to create a secure biometric login
            </p>
        </div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username",
                            help="This will be used to identify you")

    if not username:
        st.warning("Please enter a username to continue")
        return

    metadata = load_metadata()
    if username in metadata["users"]:
        st.warning(f"⚠️ User '{username}' already exists!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Re-register (replace)", type="secondary", width="stretch"):
                delete_user(username)
                st.rerun()
        with col2:
            if st.button("🔙 Back to Login", width="stretch"):
                st.session_state.auth_page = "login"
                st.rerun()
        return

    st.divider()

    st.markdown("""
        <div class="glass" style="padding:1.5rem; text-align:center;">
            <h3 style="color:var(--text-primary); margin-bottom:1rem;">Face Capture</h3>
            <p style="color:var(--text-secondary);">Click the button below and position your face in the frame.</p>
        </div>
    """, unsafe_allow_html=True)

    face_roi = capture_face_from_webcam("Look directly at the camera. Make sure your face is well-lit and centered.", key="register_capture")

    if face_roi is not None:
        st.success("✅ Face captured successfully!")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(face_roi, channels="GRAY", width=250, caption="Captured Face")

        if st.button("💾 Save & Register", type="primary", width="stretch"):
            success = register_face(username, face_roi)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.auth_page = "login"
                st.success(f"🎉 Welcome, {username}! Redirecting to login...")
                st.balloons()
                st.rerun()
            else:
                st.error("Registration failed. Please try again.")


def render_login_page():
    """Render the face login page."""
    st.markdown("""
        <div class="hero" style="text-align:center; padding:3rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;">🔐</div>
            <h1 style="color:var(--text-primary);font-size:2.5rem;font-weight:800;margin-bottom:.5rem;">
                Face Login
            </h1>
            <p style="color:var(--text-secondary);font-size:1.2rem;">
                Look at the camera to authenticate with your registered face
            </p>
        </div>
    """, unsafe_allow_html=True)

    metadata = load_metadata()
    if not metadata["users"]:
        st.warning("⚠️ No registered users found. Please register first.")
        if st.button("📝 Go to Register", type="primary", width="stretch"):
            st.session_state.auth_page = "register"
            st.rerun()
        return

    st.markdown("""
        <div class="glass" style="padding:1.5rem; text-align:center;">
            <h3 style="color:var(--text-primary); margin-bottom:1rem;">Face Recognition</h3>
            <p style="color:var(--text-secondary);">Position your face in the frame for authentication.</p>
        </div>
    """, unsafe_allow_html=True)

    face_roi = capture_face_from_webcam("Look directly at the camera for authentication.", key="login_capture")

    if face_roi is not None:
        username = recognize_face(face_roi)

        if username:
            st.success(f"✅ Welcome back, {username}!")
            st.balloons()

            if st.button("🚀 Continue to BudgetBuddy AI", type="primary", width="stretch"):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.auth_page = None
                st.rerun()
        else:
            st.error("❌ Face not recognized. Please try again or register.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Try Again", width="stretch"):
                    st.rerun()
            with col2:
                if st.button("📝 Register Instead", width="stretch"):
                    st.session_state.auth_page = "register"
                    st.rerun()


def render_auth_selection():
    """Render the initial authentication selection page."""
    st.markdown("""
        <div class="hero" style="text-align:center; padding:4rem 2rem;">
            <div style="font-size:5rem;margin-bottom:1.5rem;">💎</div>
            <h1 style="color:var(--text-primary);font-size:3rem;font-weight:800;margin-bottom:.5rem;">
                BudgetBuddy AI
            </h1>
            <p style="color:var(--text-secondary);font-size:1.3rem;max-width:600px;margin:0 auto;">
                Your intelligent financial hub with biometric security
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="glass" style="padding:2rem; text-align:center; margin-bottom:2rem;">
            <h2 style="color:var(--text-primary); margin-bottom:1rem;">Choose Authentication Method</h2>
            <p style="color:var(--text-secondary);">Secure your financial data with face recognition</p>
        </div>
    """, unsafe_allow_html=True)

    metadata = load_metadata()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
            <div class="glass" style="padding:2rem; height:100%;">
                <div style="font-size:4rem; margin-bottom:1rem;">📝</div>
                <h3 style="color:var(--text-primary); margin-bottom:1rem;">Register Face</h3>
                <p style="color:var(--text-secondary); margin-bottom:1.5rem;">
                    New user? Capture your face to create a secure biometric profile.
                </p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("📝 Register My Face", type="primary", width="stretch"):
            st.session_state.auth_page = "register"
            st.rerun()

    with col2:
        st.markdown("""
            <div class="glass" style="padding:2rem; height:100%;">
                <div style="font-size:4rem; margin-bottom:1rem;">🔐</div>
                <h3 style="color:var(--text-primary); margin-bottom:1rem;">Login with Face</h3>
                <p style="color:var(--text-secondary); margin-bottom:1.5rem;">
                    Already registered? Authenticate with your face to access your dashboard.
                </p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🔐 Login with Face", type="secondary", width="stretch"):
            st.session_state.auth_page = "login"
            st.rerun()

    if metadata["users"]:
        st.divider()
        st.markdown('<div class="glass"><div class="label">Registered Users</div></div>', unsafe_allow_html=True)
        for user_id in list(metadata["users"].keys()):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{user_id}**")
            with col2:
                st.caption(f"{len(metadata['users'][user_id].get('face_files', []))} face samples")
            with col3:
                if st.button("🗑️ Delete", key=f"del_{user_id}", help=f"Delete {user_id}"):
                    delete_user(user_id)
                    st.rerun()


def default_budgets(category_breakdown: dict) -> dict[str, float]:
    return {cat: round(amt * 1.1, 0) for cat, amt in category_breakdown.items()}


def compute_budget_status(category_breakdown: dict, budgets: dict) -> list[dict]:
    rows = []
    for cat, actual in category_breakdown.items():
        budget = budgets.get(cat, actual * 1.1)
        pct = (actual / budget * 100) if budget > 0 else 0
        if pct >= 100:
            status = "🔴 Over budget"
        elif pct >= 80:
            status = "🟡 Warning (80%+)"
        else:
            status = "🟢 On track"
        rows.append({
            "category": cat,
            "budget": budget,
            "actual": actual,
            "pct": pct,
            "status": status,
            "remaining": budget - actual,
        })
    return sorted(rows, key=lambda r: r["pct"], reverse=True)


def compute_whatif(category_breakdown: dict, reductions: dict) -> dict:
    original = sum(category_breakdown.values())
    adjusted = 0.0
    details = []
    for cat, amt in category_breakdown.items():
        cut_pct = reductions.get(cat, 0) / 100.0
        new_amt = amt * (1 - cut_pct)
        saved = amt - new_amt
        adjusted += new_amt
        if cut_pct > 0:
            details.append({"category": cat, "original": amt, "new": new_amt, "saved": saved})
    return {
        "original_expenses": original,
        "adjusted_expenses": adjusted,
        "total_saved": original - adjusted,
        "details": details,
    }


def call_groq(summary_data, anomalies_list, query, privacy_mode: bool):
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ GROQ_API_KEY is missing. Add it to your `.env` file."
    if Groq is None:
        return "⚠️ `groq` package not installed. Run: `pip install groq`"

    if privacy_mode:
        context_anomalies = [
            {"Amount": a.get("Amount"), "Category": a.get("Category"), "Explanation": a.get("Explanation")}
            for a in anomalies_list[:10]
        ]
    else:
        context_anomalies = anomalies_list[:10]

    client = Groq(api_key=api_key)
    system_prompt = (
        "You are BudgetBuddy AI, a specialized financial assistant.\n"
        "ONLY answer personal finance, budgeting, and bank statement questions.\n"
        "Refuse off-topic questions politely.\n\n"
        f"Financial Summary: {summary_data}\n"
        f"Flagged Anomalies: {context_anomalies}\n"
    )
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )
    return response.choices[0].message.content


def _badge_cls(net: float) -> str:
    return "green" if net >= 0 else "red"


def _badge_text(net: float) -> str:
    return "Surplus" if net >= 0 else "Deficit"


def render_landing():
    st.markdown("""
        <div class="hero">
            <h1>💎 BudgetBuddy AI</h1>
            <p>Intelligent finance hub — EDA · Budgets · ML Anomalies · What-If · AI Advisor</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero" style="text-align:center;padding:3rem 2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">📂</div>
            <h2 style="color:var(--text-primary);font-size:1.8rem;font-weight:700;margin-bottom:.6rem;">
                Upload your bank statement to begin
            </h2>
            <p>
                Drop a CSV file in the sidebar to unlock your full financial dashboard —
                budgets, anomaly detection, charts, and AI-powered insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div class="glass">
                <div class="label">📊 Smart Analytics</div>
                <div class="value">Real-time</div>
                <div class="description">Income, expenses, and category breakdowns.</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="glass">
                <div class="label">🤖 ML Powered</div>
                <div class="value">Auto</div>
                <div class="description">Isolation Forest anomalies and categorization.</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="glass">
                <div class="label">🔮 What-If Planning</div>
                <div class="value">Simulate</div>
                <div class="description">Model spending cuts and project savings.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="glass">
            <div class="label">Required CSV columns</div>
            <div style="color:var(--text-tertiary);font-size:.9rem;">
                Date · Description · Amount · Type · Category
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_summary_metrics(summary: dict):
    cols = st.columns(4)
    metrics = [
        ("💵 Total Income", summary.get("total_income", 0), "green", "+ Inflow"),
        ("💳 Total Expenses", summary.get("total_expenses", 0), "red", "- Outflow"),
        ("⚖️ Net Savings", summary.get("net_savings", 0),
         _badge_cls(summary.get("net_savings", 0)), _badge_text(summary.get("net_savings", 0))),
        ("📅 Daily Avg Expense", summary.get("daily_average", 0), "yellow", "Per day"),
    ]
    for col, (label, value, badge, txt) in zip(cols, metrics):
        with col:
            st.markdown(f"""
                <div class="glass delay-1">
                    <div class="label">{label}</div>
                    <div class="value">PKR {value:,.2f}</div>
                    <span class="badge {badge}">{txt}</span>
                </div>
            """, unsafe_allow_html=True)


def _fmt(v):
    return f"PKR {v:,.0f}"


def apply_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    theme = st.session_state.theme
    st.markdown(f"""
        <script>
        document.documentElement.setAttribute('data-theme', '{theme}');
        </script>
    """, unsafe_allow_html=True)


def append_transaction_to_csv(csv_path: str, transaction: dict) -> bool:
    """Append a new transaction to the CSV file."""
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=["Date", "Description", "Amount", "Type", "Category"])

        new_row = pd.DataFrame([transaction])
        df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving transaction: {e}")
        return False


@st.cache_data(show_spinner=False)
def run_pipeline(fp: str, _source_bytes: bytes):
    """Run the full data processing pipeline. Cached by file hash."""
    buf = BytesIO(_source_bytes)
    df, expenses_df, income_df = load_data.load_and_clean_data(buf)
    validation = load_data.validate_csv(df)
    summary = eda_numpy.perform_eda(df, expenses_df, income_df)
    anomalies = ml_anomaly.detect_anomalies(expenses_df)
    anomaly_metrics = ml_anomaly.get_anomaly_metrics(expenses_df)

    classifier_metrics = {}
    uncategorized = None
    try:
        model = ml_classifier.get_or_train_model(expenses_df)
        classifier_metrics = ml_classifier.evaluate_model(expenses_df, model)
        uncategorized = ml_classifier.categorize_dataframe(expenses_df, model)
    except Exception:
        pass

    chart_paths = plot_matplotlib.generate_plots(df, expenses_df, str(CHART_DIR))

    return (
        df, expenses_df, income_df, summary, anomalies,
        anomaly_metrics, classifier_metrics, uncategorized, validation, chart_paths,
    )


def run_face_recognition():
    """Run face recognition using Streamlit camera input."""
    if FACE_CASCADE is None:
        st.warning("Face detection model not loaded. Ensure `haarcascade_frontalface_default.xml` exists in `face_database/`.")
        return

    camera_image = st.camera_input("📸 Capture face for recognition", key="face_recog_tab")

    if camera_image is not None:
        file_bytes = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not process the captured image.")
            return

        result = detect_face(image)
        if result:
            face_roi, (x, y, w, h) = result
            display = image.copy()
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            st.image(display_rgb, caption="Face Detected", width="stretch")

            username = recognize_face(face_roi)
            if username:
                st.success(f"✅ Recognized: **{username}**")
            else:
                st.warning("Face detected but not recognized. The person may not be registered.")
        else:
            st.warning("No face detected. Try again with better lighting and positioning.")


def main():
    apply_theme()

    # Initialize authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = None
    if "username" not in st.session_state:
        st.session_state.username = None

    # Show authentication flow if not authenticated
    if not st.session_state.authenticated:
        if st.session_state.auth_page == "register":
            render_register_page()
            return
        elif st.session_state.auth_page == "login":
            render_login_page()
            return
        else:
            render_auth_selection()
            return

    # User is authenticated - show main app with logout option
    with st.sidebar:
        st.markdown("### 💎 BudgetBuddy AI")
        st.caption("Glass Edition · v4.0 · Pro UI + Biometric Auth")
        st.divider()

        if st.session_state.username:
            st.markdown(f"""
                <div class="glass" style="padding:.8rem 1rem;">
                    <div class="label">Logged in as</div>
                    <div style="color:var(--accent-light);font-weight:600;">{st.session_state.username}</div>
                </div>
            """, unsafe_allow_html=True)

        if st.button("🚪 Logout", width="stretch"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.auth_page = None
            st.rerun()

        st.divider()

        uploaded = st.file_uploader(
            "Upload bank statement (CSV)",
            type=["csv"],
            help="Required columns: Date, Description, Amount, Type",
        )
        privacy_mode = st.toggle("Privacy mode (LLM aggregates only)", value=True)

        st.divider()
        st.markdown("**CSV format**")
        st.caption("Date · Description · Amount · Type · Category")

        if uploaded:
            st.markdown(f"""
                <div class="glass" style="padding:.6rem .9rem;margin-top:.4rem;">
                    <div class="label">Active file</div>
                    <div style="color:var(--accent-light);font-size:.8rem;word-break:break-all;">{uploaded.name}</div>
                </div>
            """, unsafe_allow_html=True)

    if not uploaded:
        render_landing()
        return

    source_bytes = uploaded.read()
    source_name = uploaded.name
    fp = hashlib.md5(source_bytes).hexdigest()

    with st.spinner("✨ Analyzing your statement…"):
        try:
            result = run_pipeline(fp, source_bytes)
        except Exception as e:
            st.markdown("""
                <div class="hero">
                    <h1>💎 BudgetBuddy AI</h1>
                </div>
            """, unsafe_allow_html=True)
            st.error(f"Could not process file: {e}")
            st.info("Check that your CSV has Date, Description, Amount, and Type columns.")
            return

    (
        df, expenses_df, income_df, summary, anomalies,
        anomaly_metrics, classifier_metrics, uncategorized, validation, chart_paths,
    ) = result

    st.markdown("""
        <div class="hero">
            <h1>💎 BudgetBuddy AI</h1>
            <p>Financial Performance Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Data validation", expanded=not validation["valid"]):
        if validation["valid"]:
            st.success(f"✅ {validation['row_count']} rows loaded from **{source_name}**")
        else:
            for err in validation["errors"]:
                st.error(err)
        for warn in validation.get("warnings", []):
            st.warning(warn)

    tabs = st.tabs([
        "📊 Summary", "💰 Budget", "🔮 What-If", "📈 Charts",
        "🚨 Anomalies", "🏷️ Categorize", "📐 ML Metrics", "🤖 Advisor", "📄 Export",
        "➕ Add Transaction", "👤 Face Recognition",
    ])

    with tabs[0]:
        st.markdown('<div class="glass"><div class="label">Financial Performance</div></div>', unsafe_allow_html=True)
        render_summary_metrics(summary)
        st.markdown('<div class="glass"><div class="label">Spending by Category</div></div>', unsafe_allow_html=True)
        if summary.get("category_breakdown"):
            cat_df = pd.DataFrame(
                list(summary["category_breakdown"].items()),
                columns=["Category", "Amount (PKR)"],
            )
            st.bar_chart(cat_df.set_index("Category"))
        st.markdown('<div class="glass"><div class="label">Statement Preview</div></div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), width="stretch", height=350)

    with tabs[1]:
        st.markdown('<div class="glass"><div class="label">Budget vs Actual</div></div>', unsafe_allow_html=True)
        st.caption("Alerts at 80% (warning) and 100% (over budget).")

        categories = list(summary.get("category_breakdown", {}).keys())
        if "budgets" not in st.session_state:
            st.session_state.budgets = default_budgets(summary.get("category_breakdown", {}))

        with st.form("budget_form"):
            cols = st.columns(3)
            new_budgets = {}
            for i, cat in enumerate(categories):
                actual = summary["category_breakdown"][cat]
                with cols[i % 3]:
                    new_budgets[cat] = st.number_input(
                        f"{cat} (spent: {actual:,.0f})",
                        min_value=0.0,
                        value=float(st.session_state.budgets.get(cat, actual * 1.1)),
                        step=500.0,
                        key=f"budget_{cat}",
                    )
            if st.form_submit_button("Save budgets", type="primary"):
                st.session_state.budgets = new_budgets
                st.toast("Budgets saved!", icon="✅")

        budget_status = compute_budget_status(summary.get("category_breakdown", {}), st.session_state.budgets)
        for row in budget_status:
            cls = "red" if row["pct"] >= 100 else ("yellow" if row["pct"] >= 80 else "green")
            st.markdown(f"""
                <div class="glass" style="padding:1rem 1.2rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="color:var(--text-primary);font-weight:700;">{row['category']}</div>
                        <span class="badge {cls}">{row['status']}</span>
                    </div>
                    <div style="color:var(--text-secondary);font-size:.85rem;margin-top:.3rem;">
                        {_fmt(row['actual'])} / {_fmt(row['budget'])} · {row['pct']:.0f}% used
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(min(row["pct"] / 100, 1.0))

    with tabs[2]:
        st.markdown('<div class="glass"><div class="label">What-If Savings Simulator</div></div>', unsafe_allow_html=True)
        st.caption("Adjust sliders to model spending cuts.")

        reductions = {}
        for cat in summary.get("category_breakdown", {}):
            reductions[cat] = st.slider(f"Reduce {cat} by", 0, 50, 0, 5, format="%d%%", key=f"whatif_{cat}")

        whatif = compute_whatif(summary.get("category_breakdown", {}), reductions)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
                <div class="glass">
                    <div class="label">Current Expenses</div>
                    <div class="value" style="color:var(--red);">PKR {whatif['original_expenses']:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="glass">
                    <div class="label">After Cuts</div>
                    <div class="value">PKR {whatif['adjusted_expenses']:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="glass">
                    <div class="label">Monthly Savings</div>
                    <div class="value" style="color:var(--green);">PKR {whatif['total_saved']:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)

        net_after = summary.get("total_income", 0) - whatif["adjusted_expenses"]
        st.markdown(f"""
            <div class="glass">
                <div class="label">Projected Net Savings</div>
                <div class="value" style="color:#4ade80;">PKR {net_after:,.0f} / month</div>
            </div>
        """, unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="glass"><div class="label">Visual Analytics</div></div>', unsafe_allow_html=True)
        if chart_paths:
            col_a, col_b = st.columns(2)
            for idx, path in enumerate(chart_paths):
                with (col_a if idx % 2 == 0 else col_b):
                    st.markdown('<div class="glass" style="padding:.6rem;">', unsafe_allow_html=True)
                    st.image(path, width="stretch", caption=Path(path).name)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No charts generated — ensure your file has debit transactions.")

        if st.button("🔄 Regenerate charts", type="primary"):
            plot_matplotlib.generate_plots(df, expenses_df, CHART_DIR)
            st.toast("Charts updated!", icon="📈")
            st.rerun()

    with tabs[4]:
        st.markdown('<div class="glass"><div class="label">Flagged Transactions</div></div>', unsafe_allow_html=True)
        st.caption("Isolation Forest with per-transaction explanations.")
        if anomalies:
            st.warning(f"⚠️ {len(anomalies)} atypical transaction(s) detected.")
            for a in anomalies:
                st.markdown(f"""
                    <div class="glass">
                            <div style="color:var(--text-primary);font-weight:600;">
                            {a.get('Date')} · {a.get('Description')}
                        </div>
                            <div style="color:var(--red);font-weight:700;margin-top:.3rem;">
                            PKR {a.get('Amount', 0):,.0f} · {a.get('Category')}
                        </div>
                            <div style="color:var(--text-secondary);font-size:.85rem;margin-top:.4rem;">
                            {a.get('Explanation', '')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No irregular transactions flagged.")

    with tabs[5]:
        st.markdown('<div class="glass"><div class="label">Auto-Categorize</div></div>', unsafe_allow_html=True)
        if uncategorized is not None and len(uncategorized):
            st.dataframe(uncategorized, width="stretch", hide_index=True)
        else:
            st.success("All transactions have categories assigned.")
        st.markdown('<div class="glass"><div class="label">Try a merchant</div></div>', unsafe_allow_html=True)
        test_desc = st.text_input("Merchant description", "Foodpanda Order", label_visibility="collapsed")
        if st.button("Predict category", type="primary"):
            cat, conf = ml_classifier.predict_category(test_desc)
            badge_color = "green" if conf > 0.7 else ("yellow" if conf > 0.4 else "red")
            st.markdown(f"""
                <div class="glass">
                    <div class="label">Prediction</div>
                    <div style="color:var(--green);font-size:1.3rem;">{cat}</div>
                    <span class="badge {badge_color}">{conf:.0%} confidence</span>
                </div>
            """, unsafe_allow_html=True)

    with tabs[6]:
        st.markdown('<div class="glass"><div class="label">Model Evaluation</div></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🚨 Anomaly Detection")
            if anomaly_metrics.get("trained"):
                st.metric("Anomalies flagged", anomaly_metrics["anomaly_count"])
                st.metric("Normal avg", f"PKR {anomaly_metrics['normal_avg_amount']:,.0f}")
                st.metric("Anomaly avg", f"PKR {anomaly_metrics['anomaly_avg_amount']:,.0f}")
            else:
                st.info(anomaly_metrics.get("message", "Not enough data."))
        with col2:
            st.markdown("#### 🏷️ Category Classifier")
            if classifier_metrics:
                st.metric("Test accuracy", f"{classifier_metrics['accuracy'] * 100:.1f}%")
                st.metric("CV mean", f"{classifier_metrics['cv_mean'] * 100:.1f}%")
                report = classifier_metrics.get("classification_report", {})
                per_class = {
                    k: {"precision": v.get("precision"), "recall": v.get("recall"), "f1": v.get("f1-score")}
                    for k, v in report.items()
                    if k not in {"accuracy", "macro avg", "weighted avg"}
                }
                if per_class:
                    st.dataframe(pd.DataFrame(per_class).T, width="stretch")
            else:
                st.info("Classifier metrics unavailable.")

    with tabs[7]:
        st.markdown('<div class="glass"><div class="label">AI Financial Advisor</div></div>', unsafe_allow_html=True)
        if privacy_mode:
            st.caption("🔒 Privacy mode — only aggregates sent to Groq.")
        user_query = st.text_input(
            "Your question",
            placeholder="Where should I cut 10% to save more?",
            label_visibility="collapsed"
        )
        if st.button("✨ Get insights", type="primary"):
            if not user_query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking…"):
                    try:
                        response = call_groq(summary, anomalies, user_query, privacy_mode)
                    except Exception as ex:
                        response = f"Groq API error: {ex}"
                with st.chat_message("assistant", avatar="💎"):
                    st.markdown(response)

    with tabs[8]:
        st.markdown('<div class="glass"><div class="label">Export Report</div></div>', unsafe_allow_html=True)
        budget_status = compute_budget_status(
            summary.get("category_breakdown", {}),
            st.session_state.get("budgets", default_budgets(summary.get("category_breakdown", {}))),
        )
        html = report_export.build_html_report(summary, anomalies, budget_status, source_name)
        st.download_button("⬇️ Download HTML Report", html, file_name="budgetbuddy_report.html", mime="text/html")
        pdf_bytes = report_export.build_pdf_report(summary, anomalies, budget_status, source_name)
        if pdf_bytes:
            st.download_button("⬇️ Download PDF Report", pdf_bytes, file_name="budgetbuddy_report.pdf", mime="application/pdf")
        else:
            st.caption("Install `reportlab` for PDF export.")

    with tabs[9]:
        st.markdown('<div class="glass"><div class="label">Add New Transaction</div></div>', unsafe_allow_html=True)
        st.caption("Add a transaction that will be saved to your CSV file.")

        with st.form("add_transaction_form"):
            col1, col2 = st.columns(2)
            with col1:
                trans_date = st.date_input("Date", value=pd.Timestamp.now().date())
                description = st.text_input("Description", placeholder="e.g., Grocery Store, Uber Ride")
                amount = st.number_input("Amount (PKR)", min_value=0.0, step=100.0, format="%.2f")
            with col2:
                trans_type = st.selectbox("Type", ["Debit", "Credit"])
                category = st.text_input("Category (optional)", placeholder="e.g., Food, Travel, Shopping")
                csv_file = st.text_input("CSV File Path", value=source_name, help="Path to save the transaction")

            submitted = st.form_submit_button("💾 Save Transaction", type="primary")

            if submitted:
                if not description:
                    st.error("Please enter a description.")
                elif amount <= 0:
                    st.error("Please enter a valid amount.")
                else:
                    transaction = {
                        "Date": trans_date.strftime("%Y-%m-%d"),
                        "Description": description,
                        "Amount": amount,
                        "Type": trans_type,
                        "Category": category if category else "Others"
                    }
                    success = append_transaction_to_csv(csv_file, transaction)
                    if success:
                        st.success(f"✅ Transaction saved to {csv_file}!")
                        st.info("🔄 Reload the page or re-upload the CSV to see the updated data.")
                        st.json(transaction)

    with tabs[10]:
        st.markdown('<div class="glass"><div class="label">Face Recognition</div></div>', unsafe_allow_html=True)
        st.caption("Capture a photo to detect and recognize faces.")

        if cv2 is None:
            st.error("OpenCV not installed. Run: `pip install opencv-python`")
        else:
            run_face_recognition()


if __name__ == "__main__":
    main()