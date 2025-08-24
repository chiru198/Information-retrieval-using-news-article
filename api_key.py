import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
 # ✅ updated model name
    response = model.generate_content("Hello Gemini, is my API key valid?")
    print("✅ Gemini Response:", response.text)
except Exception as e:
    print("❌ Error:", e)
