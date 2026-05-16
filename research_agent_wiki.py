# research_agent.py
# An AI research agent — type any topic, get a live web summary!

import os
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# --- Setup ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")

def search_web(topic):
    """Search the web for a topic using wikipedia (free, no key needed)"""
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")

    headers = {
        "User-Agent": "ResearchAgent/1.0 (learning project; contact@example.com)"
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()

    results = []

    # Get main abstract
    if data.get("extract"):
        results.append(data["extract"])

    # Get related topics
    for item in data.get("RelatedTopics", [])[:5]:
        if isinstance(item, dict) and item.get("Text"):
            results.append(item["Text"])

    return "\n".join(results) if results else "No results found."

def summarize_with_ai(topic, search_results):
    """Use Groq AI to summarize the search results"""
    prompt = f"""
You are a research assistant. A user searched for: "{topic}"

Here are the raw search results:
{search_results}

Please write a clear, helpful 3-5 paragraph summary about this topic based on the search results.
Make it easy to read and informative.
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def save_report(topic, summary):
    """Save the summary to a text file"""
    filename = f"report_{topic.replace(' ', '_')[:30]}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"RESEARCH REPORT: {topic.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(summary)
    print(f"\n Report saved to: {filename}")

# --- Main agent loop ---
print("Welcome to your AI Research Agent!")
print("Type any topic and I'll search the web and summarize it for you.")
print("Type 'quit' to exit.\n")

while True:
    topic = input("What do you want to research? ")

    if topic.lower() == "quit":
        print("Goodbye!")
        break

    print(f"\nSearching the web for '{topic}'...")
    search_results = search_web(topic)

    print("Summarizing with AI...")
    summary = summarize_with_ai(topic, search_results)

    print("\n" + "=" * 50)
    print(f"RESEARCH SUMMARY: {topic.upper()}")
    print("=" * 50)
    print(summary)

    save_report(topic, summary)
    print("\n" + "-" * 50 + "\n")
