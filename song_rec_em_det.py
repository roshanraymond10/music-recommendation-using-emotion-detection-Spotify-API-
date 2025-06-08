
import cv2
from deepface import DeepFace
import pyttsx3
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import mediapipe as mp
import math
import time

# === Spotify API credentials ===
SPOTIFY_CLIENT_ID = 'f6c613f376254e96a06b4dcba0c4eaed'
SPOTIFY_CLIENT_SECRET = 'ee0d60fb5fea42a4b5ad0c8797490e40'
REDIRECT_URI = 'http://127.0.0.1:8000/callback'

# === TTS setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# === Spotify Auth ===
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="user-read-playback-state user-modify-playback-state playlist-read-private"
))

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Webcam setup ===
cap = cv2.VideoCapture(0)
emotion_detected = False
emotion = None
paused = False
last_gesture_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # === Hand Gesture Detection ===
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Get coordinates
            wrist = handLms.landmark[0]
            index_tip = handLms.landmark[8]
            middle_tip = handLms.landmark[12]

            x1, y1 = int(wrist.x * w), int(wrist.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
            x3, y3 = int(middle_tip.x * w), int(middle_tip.y * h)

            # Distance between wrist and index
            distance = math.hypot(x2 - x1, y2 - y1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Dist: {int(distance)}px', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            current_time = time.time()
            if current_time - last_gesture_time > 1.0:

                # Pause
                if distance > 180 and not paused:
                    try:
                        sp.pause_playback()
                        paused = True
                        print("⏸️ Playback paused by hand gesture.")
                    except:
                        print("❌ Pause failed or no active device.")
                    last_gesture_time = current_time

                # Resume
                elif distance < 120 and paused:
                    try:
                        sp.start_playback()
                        paused = False
                        print("▶️ Playback resumed by hand gesture.")
                    except:
                        print("❌ Resume failed or no active device.")
                    last_gesture_time = current_time

                # Next track → if index + middle finger move to right
                elif (x2 < x3) and abs(y2 - y3) < 40:
                    try:
                        sp.next_track()
                        print("➡️ Skipped to next track.")
                    except:
                        print("❌ Next track failed.")
                    last_gesture_time = current_time

                # Previous track ← if index + middle finger move to left
                elif (x2 > x3) and abs(y2 - y3) < 40:
                    try:
                        sp.previous_track()
                        print("⬅️ Returned to previous track.")
                    except:
                        print("❌ Previous track failed.")
                    last_gesture_time = current_time

    # === Emotion Detection (Only once) ===
    if not emotion_detected:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion'].lower()
            print(f"😃 Detected emotion: {emotion}")
            engine.say(f"You look {emotion}")
            engine.runAndWait()

            # Get active Spotify device
            devices = sp.devices()
            if not devices['devices']:
                print("❗ No active Spotify devices found. Open Spotify and start playing something.")
            else:
                device_id = devices['devices'][0]['id']

                # Find playlist in user's library matching emotion
                user_playlists = sp.current_user_playlists(limit=50)
                target_playlist = None
                for playlist in user_playlists['items']:
                    if emotion in playlist['name'].lower():
                        target_playlist = playlist
                        break

                if target_playlist:
                    playlist_uri = target_playlist['uri']
                    sp.start_playback(device_id=device_id, context_uri=playlist_uri)
                    print(f"🎵 Now playing: {target_playlist['name']}")
                else:
                    print(f"🚫 No playlist named '{emotion}' found in your Spotify account.")

        except Exception as e:
            print(f"⚠️ Error during emotion detection or Spotify playback: {str(e)}")

        emotion_detected = True

    # === Show webcam ===
    if emotion:
        cv2.putText(frame, f'Emotion: {emotion}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

    cv2.imshow("Emotion + Gesture Music Player", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
