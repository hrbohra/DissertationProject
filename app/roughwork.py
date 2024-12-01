import spacy
import time
from transformers import pipeline

# Load the emotion classification pipeline with the pre-trained RoBERTa model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Simulate a large dataset of text inputs
text_data = ["This is a great product!", "I am very sad today", "This is terrible", "I'm so happy!"] * 10000  # 40,000 inputs

# Start timer
start_time = time.time()

# Process text inputs in batches
for i in range(0, len(text_data), 1000):
    batch = text_data[i:i+1000]
    results = emotion_classifier(batch)

# End timer
end_time = time.time()

# Calculate throughput
total_time = end_time - start_time
total_inputs = len(text_data)
throughput = total_inputs / total_time

print(f"Processed {total_inputs} text inputs in {total_time:.2f} seconds.")
print(f"Throughput: {throughput:.2f} inputs per second")
