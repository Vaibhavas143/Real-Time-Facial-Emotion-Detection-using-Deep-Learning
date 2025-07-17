import cv2
from deepface import DeepFace
import datetime
import csv

# Start webcam
cap = cv2.VideoCapture(0)
emotion_log = []

print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        
        # Log emotion with timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        emotion_log.append((timestamp, dominant_emotion))

        # Show on screen
        cv2.putText(frame,
                    f'Emotion: {dominant_emotion}',
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)
    except Exception as e:
        print("Face not detected:", e)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save emotion log to CSV
with open("emotion_log.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Emotion"])
    writer.writerows(emotion_log)

print("Emotion log saved to emotion_log.csv")

# Cleanup
cap.release()
cv2.destroyAllWindows()
# Save emotion log to CSV
# Debug: Print how many emotions were collected
print(f"\n📊 Total emotions collected: {len(emotion_log)}")

# Debug: Show a few entries
if emotion_log:
    print("🧾 Sample log:", emotion_log[:3])  # show first 3 entries
else:
    print("⚠️ No emotions were logged!")

# Save emotion log to CSV only if there is data
if len(emotion_log) > 0:
    try:
        with open("emotion_log.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Emotion"])
            writer.writerows(emotion_log)
        print("✅ Emotion log saved to emotion_log.csv")
    except Exception as e:
        print("❌ Error saving CSV:", e)
else:
    print("❌ No data to save.")


