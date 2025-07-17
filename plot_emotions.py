import pandas as pd
import matplotlib.pyplot as plt

# Load the emotion log
try:
    df = pd.read_csv("emotion_log.csv")
    print("✅ CSV loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: File 'emotion_log.csv' not found.")
    exit()

# Debug print
print("📋 First few rows:")
print(df.head())

# Count frequency of each emotion
emotion_counts = df['Emotion'].value_counts()
print("\n📊 Emotion counts:")
print(emotion_counts)

# Plot
plt.figure(figsize=(8, 6))
emotion_counts.plot(kind='bar', color='skyblue')
plt.title("Emotion Frequency")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()

# Show the plot
print("\n📈 Displaying plot window...")
plt.show()
