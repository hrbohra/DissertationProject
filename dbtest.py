import sqlite3
import time
# Connect to your SQLite database (adjust the path to your database file)
conn = sqlite3.connect('app.db')  # or 'test.db' for the test database
# Create a cursor object to execute queries
cursor = conn.cursor()
# Define the query you want to test
query = "SELECT * FROM user"
# Record the start time
start_time = time.time()
# Execute the query
cursor.execute(query)
# Fetch the results (optional, depending on whether you need them)
results = cursor.fetchall()
# Record the end time
end_time = time.time()
# Calculate the time taken for the query
execution_time = end_time - start_time
# Print the execution time
print(f"Query executed in {execution_time} seconds")
# Close the connection
conn.close()
