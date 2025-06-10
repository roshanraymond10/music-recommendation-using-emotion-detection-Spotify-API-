import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import mediapipe as mp
import math
import time
import speech_recognition as sr

# --- Spotify Auth ---
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='f6c613f376254e96a06b4dcba0c4eaed',
    client_secret='ee0d60fb5fea42a4b5ad0c8797490e40',
    redirect_uri='http://localhost:8888/callback',
    scope="user-read-playback-state user-modify-playback-state playlist-read-private"
))

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Music Player")
st.title("🎧 Emotion-Based Music Player")
frame_holder = st.image([])

# --- Session Variables ---
if "paused" not in st.session_state:
    st.session_state.paused = False
if "last_gesture_time" not in st.session_state:
    st.session_state.last_gesture_time = 0
if "emotion" not in st.session_state:
    st.session_state.emotion = None

# --- Webcam ---
cap = cv2.VideoCapture(0)

# --- Speech Command Handler ---
def handle_voice_commands():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Say 'play', 'pause', 'stop', or 'skip'")
        audio = recognizer.listen(source, phrase_time_limit=4)
        try:
            command = recognizer.recognize_google(audio).lower()
            st.success(f"Heard: **{command}**")
            if "play" in command:
                sp.start_playback()
            elif "pause" in command or "stop" in command:
                sp.pause_playback()
            elif "skip" in command:
                sp.next_track()
            else:
                st.warning("Unrecognized command.")
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError as e:
            st.error(f"Voice API error: {e}")

# --- Emotion Detection ---
def detect_emotion_and_play():
    st.info("Capturing emotion...")
    frames = []
    for _ in range(10):  # Capture a few frames for better emotion accuracy
        ret, f = cap.read()
        if ret:
            frames.append(cv2.flip(f, 1))
        time.sleep(0.1)

    avg_frame = np.median(frames, axis=0).astype(np.uint8)
    rgb = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2RGB)
    result = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion'].lower()
    st.session_state.emotion = emotion
    st.success(f"Detected Emotion: **{emotion.capitalize()}**")

    devices = sp.devices()
    if not devices['devices']:
        st.warning("⚠️ Open Spotify and start any music to activate a device.")
        return

    device_id = devices['devices'][0]['id']
    playlists = sp.current_user_playlists(limit=50)

    target_playlist = None
    for p in playlists['items']:
        if emotion in p['name'].lower():
            target_playlist = p
            break

    if target_playlist:
        sp.start_playback(device_id=device_id, context_uri=target_playlist['uri'])
        st.success(f"🎵 Now playing: **{target_playlist['name']}**")
        playback = sp.current_playback()
        if playback and playback['is_playing']:
            track = playback['item']
            st.markdown(f"**Current Song:** {track['name']} by {', '.join([artist['name'] for artist in track['artists']])}")
    else:
        st.warning(f"🚫 No playlist named '{emotion}' found in your account.")

# --- Main Frame Loop ---
ret, frame = cap.read()
if not ret:
    st.error("❌ Unable to access webcam.")
else:
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # --- Hand Gesture Detection ---
    results = hands.process(rgb)
    current_time = time.time()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            wrist = handLms.landmark[0]
            index_tip = handLms.landmark[8]
            middle_tip = handLms.landmark[12]

            x1, y1 = int(wrist.x * w), int(wrist.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
            x3, y3 = int(middle_tip.x * w), int(middle_tip.y * h)

            distance = math.hypot(x2 - x1, y2 - y1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if current_time - st.session_state.last_gesture_time > 1.0:
                try:
                    if distance > 180 and not st.session_state.paused:
                        sp.pause_playback()
                        st.session_state.paused = True
                        st.info("⏸️ Paused via gesture")
                        st.session_state.last_gesture_time = current_time

                    elif distance < 120 and st.session_state.paused:
                        sp.start_playback()
                        st.session_state.paused = False
                        st.info("▶️ Resumed via gesture")
                        st.session_state.last_gesture_time = current_time

                    elif (x2 < x3) and abs(y2 - y3) < 40:
                        sp.next_track()
                        st.info("➡️ Next track via gesture")
                        st.session_state.last_gesture_time = current_time

                    elif (x2 > x3) and abs(y2 - y3) < 40:
                        sp.previous_track()
                        st.info("⬅️ Previous track via gesture")
                        st.session_state.last_gesture_time = current_time
                except Exception as e:
                    st.warning(f"Spotify error: {e}")

    if st.session_state.emotion:
        cv2.putText(frame, f'Emotion: {st.session_state.emotion.capitalize()}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_holder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Live Webcam")

# --- Buttons ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🎭 Detect Emotion"):
        detect_emotion_and_play()

with col2:
    if st.button("🗣️ Voice Command"):
        handle_voice_commands()

# --- Currently Playing Track ---
try:
    current = sp.current_playback()
    if current and current['is_playing']:
        track = current['item']
        song_name = track['name']
        artist = ', '.join([a['name'] for a in track['artists']])
        st.markdown(f"🎶 **Now Playing:** *{song_name}* by *{artist}*")
    else:
        st.info("ℹ️ No song is currently playing.")
except Exception as e:
    st.warning(f"Couldn't fetch current track: {e}")

# --- Feedback ---
st.markdown("### 🧠 Feedback")
feedback = st.radio("Did this song improve your mood?", ["Yes", "No", "Not Sure"], key="feedback")
if st.button("Submit Feedback"):
    st.success(f"✅ Thank you for your feedback: *{feedback}*")
    # Here you could store feedback to a file or database
