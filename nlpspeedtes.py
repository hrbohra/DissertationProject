import spacy
import time

# Load the large English model
nlp = spacy.load("en_core_web_lg", disable=["ner", "parser", "lemmatizer"])

# Use only tokenization and tagging

# Simulate a large dataset of text inputs
text_data = ["This is a great product!", "I am very sad today", "This is terrible", "I'm so happy!"] * 10000  # 40,000 inputs

# Start timer
start_time = time.time()

# Use spaCy's pipe to process text inputs in batches
docs = list(nlp.pipe(text_data, batch_size=1000))  # Adjust batch_size to tune performance

# End timer
end_time = time.time()

# Calculate throughput
total_time = end_time - start_time
total_inputs = len(text_data)
throughput = total_inputs / total_time

print(f"Processed {total_inputs} text inputs in {total_time:.2f} seconds.")
print(f"Throughput: {throughput:.2f} inputs per second")
