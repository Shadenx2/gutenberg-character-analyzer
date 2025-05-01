from flask import Flask, request, render_template
import requests
import json
import os
import re
from dotenv import load_dotenv

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)


def extract_characters_with_groq(text):
    """
    Send cleaned book text to Groq LLM and extract only the JSON characters block.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{
            "role": "user",
            "content": f"""
From the following book text, extract the main characters and their interactions.
Respond only with exactly the JSON object in this format (no explanation):

{{
  "characters": [
    {{"name": "CharacterName", "interacts_with": ["OtherName1", "OtherName2"]}},
    ...
  ]
}}

TEXT:
{text[:3000]}
"""
        }]
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    if resp.status_code != 200:
        return "{\"characters\": []}"

    raw = resp.json()['choices'][0]['message']['content'].strip()
    # Extract first JSON block between braces
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return "{\"characters\": []}"
    block = m.group(0)
    # Validate JSON
    try:
        parsed = json.loads(block)
        return json.dumps(parsed)
    except json.JSONDecodeError:
        return "{\"characters\": []}"


def clean_gutenberg_text(text):
    """
    Strip Gutenberg headers/footers, fallback to Acts/Chapters or skip first 100 lines.
    """
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1 and end != -1:
        return text[start+len("*** START OF"):end]
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "ACT I" in line.upper() or "CHAPTER I" in line.upper():
            return "\n".join(lines[i:])
    return "\n".join(lines[100:])


@app.route("/", methods=["GET", "POST"])
def index():
    book_content = ""
    llm_output = "{\"characters\": []}"
    if request.method == "POST":
        book_id = request.form.get("book_id", "").strip()
        # Attempt both URL patterns
        for suffix in [f"{book_id}-0.txt", f"{book_id}.txt"]:
            url = f"https://www.gutenberg.org/files/{book_id}/{suffix}"
            try:
                r = requests.get(url)
                r.raise_for_status()
                book_content = clean_gutenberg_text(r.text)
                break
            except Exception:
                continue
        # Call LLM if content fetched
        if book_content:
            llm_output = extract_characters_with_groq(book_content)
    return render_template("index.html", book_content=book_content, llm_output=llm_output)


if __name__ == "__main__":
    app.run(debug=True)
