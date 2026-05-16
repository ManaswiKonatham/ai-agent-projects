# my_first_agent_groq.py
import os
from dotenv import load_dotenv
from groq import Groq

# --- Setup ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Your question ---
user_question = "whats the largest word in english?"

# --- Ask the AI ---
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": user_question}
    ]
)

# --- Print the answer ---
print("AI says:\n")
print(response.choices[0].message.content)
