import os
import sqlite3
import requests
import base64
import io
import re
import json
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Robust Environment Loading
GOOGLE_API_KEY = None
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Method 1: Try python-dotenv
try:
    from dotenv import load_dotenv
    if os.path.exists(ENV_PATH):
        print(f"Loading .env from: {ENV_PATH}")
        load_dotenv(ENV_PATH)
    else:
        print("Warning: .env file not found at expected path.")
except ImportError:
    print("Warning: python-dotenv not installed.")

# Method 2: Manual Parse (Fallback)
GOOGLE_API_KEY = os.getenv("Google_api")

if not GOOGLE_API_KEY and os.path.exists(ENV_PATH):
    print("Attempting manual .env parsing...")
    try:
        with open(ENV_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Google_api="):
                    # Extract value and remove quotes if present
                    key_val = line.split("=", 1)[1].strip()
                    if (key_val.startswith('"') and key_val.endswith('"')) or \
                       (key_val.startswith("'") and key_val.endswith("'")):
                        key_val = key_val[1:-1]
                    GOOGLE_API_KEY = key_val
                    os.environ["Google_api"] = key_val # Set for other parts
                    print("Key found via manual parsing.")
                    break
    except Exception as e:
        print(f"Manual parsing failed: {e}")

if not GOOGLE_API_KEY:
    print("CRITICAL ERROR: Google_api key could not be loaded via dotenv or manual parsing.")
    # Stop execution or set a dummy to avoid 'None' in URL string which confuses users
    GOOGLE_API_KEY = "" 

# Print debug info
if GOOGLE_API_KEY:
    mask_key = f"{GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-5:]}"
    print(f"Active Google API Key: {mask_key} (Length: {len(GOOGLE_API_KEY)})")
else:
    print("Active Google API Key: NOT FOUND")

app = Flask(__name__)
CORS(app)

STATIC_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
DB_PATH = os.path.join(BASE_DIR, "diet.db")

import time
import random

# Use gemini-2.0-flash as primary, but fallback if needed
# Use gemini-2.0-flash as primary, but fallback if needed
# Aggressive model list to bypass regional/API-key/versioning related 404 errors
AVAILABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-1.0-pro"
]

def get_supported_models():
    """Fetch the actual available models for this API key and filter for text-capable ones."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_API_KEY}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            all_models = r.json().get('models', [])
            # Filter criteria:
            # 1. Must support 'generateContent'
            # 2. Must NOT be a TTS (Text-to-Speech) model
            # 3. Must NOT be an Embedding or AQA model
            models = [
                m['name'].split('/')[-1] for m in all_models 
                if 'generateContent' in m.get('supportedGenerationMethods', [])
                and 'tts' not in m['name'].lower()
                and 'embed' not in m['name'].lower()
                and 'aqa' not in m['name'].lower()
            ]
            print(f"Discovered {len(models)} text-capable models: {models}")
            return models
    except Exception as e:
        print(f"Discovery failed: {e}")
    return AVAILABLE_MODELS

def call_gemini(payload, timeout=30):
    global GOOGLE_API_KEY
    
    # Try discovered models first, then hardcoded ones
    models_to_try = get_supported_models()
    
    last_error = "No models attempted"
    
    for model in models_to_try:
        # Use v1beta as it's the most feature-complete for Gemini
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_API_KEY}"
        
        try:
            print(f"Trying {model}...")
            r = requests.post(url, json=payload, timeout=timeout)
            
            if r.status_code == 200:
                print(f"Success with {model}")
                return r.json()
            
            error_data = r.json() if r.headers.get('content-type') == 'application/json' else r.text
            last_error = f"API {r.status_code} ({model}): {error_data}"
            print(f"Failed {model}: {r.status_code}")
            
            # If it's a 403 (Permission) or 400 (Bad Request), trying other models won't help
            if r.status_code in [400, 403]:
                break
                
        except Exception as e:
            last_error = str(e)
            continue

    raise Exception(f"All AI attempts failed. Final error: {last_error}")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                gender TEXT,
                height REAL,
                weight REAL,
                bmi REAL,
                calories REAL,
                protein REAL,
                carbs REAL,
                fats REAL,
                goal TEXT,
                timestamp TEXT DEFAULT (datetime('now','localtime'))
            )
        """)

init_db()

def calculate_bmi(w, h):
    return w / ((h / 100) ** 2)

def image_to_base64(file):
    file.seek(0)
    return base64.b64encode(file.read()).decode()

# ---------- ROUTES ----------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    try:
        name = data.get("name")
        age = int(data.get("age"))
        gender = data.get("gender", "male")
        height = float(data.get("height"))
        weight = float(data.get("weight"))
        goal = data.get("goal")
        activity = data.get("activity", "sedentary")
        
        # BMR Calculation (Mifflin-St Jeor Equation)
        if gender == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        activity_map = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
        calories = bmr * activity_map.get(activity, 1.2)
        
        if goal == "loss": calories -= 500
        elif goal == "gain": calories += 400
            
        water = weight * 35
        bmi = calculate_bmi(weight, height)
        
        protein = weight * 2.0
        fats = weight * 0.8
        carbs = (calories - (protein * 4 + fats * 9)) / 4
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO users (name, age, gender, height, weight, bmi, calories, protein, carbs, fats, goal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, age, gender, height, weight, bmi, calories, protein, carbs, fats, goal))
        
        return jsonify({
            "BMI": round(bmi, 1),
            "Calories": round(calories),
            "Protein": round(protein),
            "Carbs": round(carbs),
            "Fats": round(fats),
            "Water": round(water)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/scan-food", methods=["POST"])
def scan_food():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    try:
        img_b64 = image_to_base64(file)
        
        prompt = 'Identify the food items in this image. For each item, provide its name, confidence score (0-1), and estimated calories per serving. Return strictly a JSON object in this format: {"foods": [{"name": "Food Name", "confidence": 0.95, "calories": 250}]}'
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": file.content_type or "image/jpeg", "data": img_b64}}
                ]
            }]
        }
        
        # Use robust call
        data = call_gemini(payload)
        
        if "candidates" not in data or not data["candidates"]:
            return jsonify({"error": "No results from AI. Safety filters might be active."}), 500
            
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Clean the text (remove markdown code blocks if present)
        clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        
        if match:
            try:
                json_data = json.loads(match.group())
                return jsonify(json_data)
            except json.JSONDecodeError:
                return jsonify({"error": "AI returned invalid JSON structure", "raw": text}), 500
                
        return jsonify({"error": "AI response did not contain JSON", "raw": text}), 500
    except Exception as e:
        print(f"Scan Food Error: {e}")
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

@app.route("/diet-plan", methods=["POST"])
def diet_plan():
    data = request.json
    try:
        prompt = f"""
        Generate a highly personalized diet plan for:
        Name: {data.get('name')} | Age: {data.get('age')} | BMI: {data.get('bmi')} | Goal: {data.get('goal')}
        Target: {data.get('calories')} kcal | Diet: {data.get('diet')} | Allergies: {data.get('allergies')} | Meals: {data.get('mealsPerDay')} per day
        
        Return STRICTLY a JSON object with this structure:
        {{
          "meals": [
            {{"time": "Breakfast", "item": "Food name", "calories": 400, "macros": "P:20g, C:50g, F:10g"}},
            ... (total {data.get('mealsPerDay')} meals)
          ],
          "health_insights": "Detailed advice on why these choices were made...",
          "tips": ["Tip 1", "Tip 2", "Tip 3"]
        }}
        """
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        # Use robust call
        res_data = call_gemini(payload)
        
        if "candidates" not in res_data or not res_data["candidates"]:
             return jsonify({"error": "AI could not generate a plan."}), 500
             
        text = res_data["candidates"][0]["content"]["parts"][0]["text"]
        # Clean the text (remove markdown code blocks if present)
        clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        
        if match:
            return jsonify(json.loads(match.group()))
        
        return jsonify({"error": "AI response did not contain JSON", "raw": text}), 500
    except Exception as e:
        return jsonify({"error": f"AI Plan failed: {str(e)}"}), 500

@app.route("/weekly-data")
def weekly():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT calories, protein, carbs, fats FROM users ORDER BY id DESC LIMIT 7").fetchall()
        
        if not rows:
            # Fallback for empty DB so the chart doesn't break
            return jsonify({
                "labels": ["Day 1"],
                "calories": [0],
                "protein": [0],
                "carbs": [0],
                "fats": [0]
            })

        return jsonify({
            "labels": [f"Day {i+1}" for i in range(len(rows))],
            "calories": [r[0] for r in rows][::-1],
            "protein": [r[1] for r in rows][::-1],
            "carbs": [r[2] for r in rows][::-1],
            "fats": [r[3] for r in rows][::-1]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-names")
def get_names():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Get unique names from the users table
            rows = conn.execute("SELECT DISTINCT name FROM users ORDER BY name").fetchall()
        return jsonify([r[0] for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-user-by-name/<string:name>")
def get_user_by_name(name):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM users WHERE name=? ORDER BY id DESC LIMIT 1", (name,)).fetchone()
        if row:
            return jsonify(dict(row))
        return jsonify({})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete-user-by-name/<string:name>", methods=["DELETE"])
def delete_user_by_name(name):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM users WHERE name=?", (name,))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-latest-user")
def get_latest_user():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM users ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            return jsonify(dict(row))
        return jsonify({})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download-report/<int:user_id>")
@app.route("/download-report/latest")
def download_pdf(user_id=None):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            if user_id and user_id != 'latest':
                cur.execute("SELECT name, bmi, calories, protein, carbs, fats FROM users WHERE id=?", (user_id,))
            else:
                # Get the most recent user entry
                cur.execute("SELECT name, bmi, calories, protein, carbs, fats FROM users ORDER BY id DESC LIMIT 1")
            
            user = cur.fetchone()
            if not user: return "User data not found.", 404
                
            name, bmi, target_calories, p, c, f = user
            
            # Fetch weekly history
            cur.execute("SELECT calories, protein, carbs, fats, timestamp FROM users WHERE name=? ORDER BY id DESC LIMIT 7", (name,))
            rows = cur.fetchall()

        buf = io.BytesIO()
        pdf = canvas.Canvas(buf, pagesize=A4)
        
        # Header
        pdf.setFont("Helvetica-Bold", 22)
        pdf.setFillColorRGB(0.42, 0.39, 1.0) # Purple accent
        pdf.drawString(50, 800, "HEAL.AI | Complete Health & Diet Report")
        
        pdf.setStrokeColorRGB(0.8, 0.8, 0.8)
        pdf.line(50, 790, 540, 790)
        
        # Profile Summary
        pdf.setFont("Helvetica-Bold", 14)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.drawString(50, 765, f"Patient Name: {name}")
        pdf.setFont("Helvetica", 11)
        pdf.drawString(50, 750, f"BMI: {bmi:.1f} | Daily Calorie Target: {target_calories:.0f} kcal")
        pdf.drawString(50, 735, f"Macros: Protein {p:.0f}g | Carbs {c:.0f}g | Fats {f:.0f}g")

        # Health History Table
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, 700, "Recent Health Trends (Last 7 Days)")
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(50, 680, "Date             Calories   Protein   Carbs   Fats")
        
        pdf.setFont("Helvetica", 10)
        y = 665
        for row in rows[::-1]:
            date = row[4][:10] if row[4] else "-"
            pdf.drawString(50, y, f"{date}   {row[0]:>8.0f}   {row[1]:>7.1f}   {row[2]:>6.1f}   {row[3]:>5.1f}")
            y -= 15
        
        # Diet Plan Section (If exists in session/passed, otherwise generic placeholder for now)
        # Note: In a real app, we'd store the 'last_diet_plan' in the DB to retrieve it here.
        # For this version, we'll leave a dedicated space or provide tips.
        y -= 30
        pdf.line(50, y+10, 540, y+10)
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "AI Personalized Diet Recommendations")
        
        pdf.setFont("Helvetica", 11)
        y -= 25
        recommendations = [
            "• Maintain consistent meal timings for metabolic stability.",
            "• Prioritize complex carbohydrates (oats, quinoa) over refined sugars.",
            "• Ensure total water intake matches your profile calculated target.",
            "• Review your macro distribution weekly to adjust for weight changes."
        ]
        for rec in recommendations:
            pdf.drawString(60, y, rec)
            y -= 20

        pdf.setFont("Helvetica-Oblique", 9)
        pdf.drawString(50, 40, "Disclaimer: This report is AI-generated for educational purposes. Consult a doctor for medical advice.")
        
        pdf.showPage()
        pdf.save()
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name=f"Full_Health_Report_{name}.pdf", mimetype="application/pdf")
    except Exception as e:
        return str(e), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    context = data.get("context", {})

    analysis = context.get("analysis", {})
    diet_plan = context.get("dietPlan", "")
    weekly_stats = context.get("weeklyStats", {})

    try:
        prompt = f"""
You are Bella, a friendly, intelligent AI Diet & Health Assistant.

USER HEALTH DATA (CURRENT):
- Name: {analysis.get('name', 'Friend')}
- BMI: {analysis.get('BMI', 'Unknown')}
- Daily Calories: {analysis.get('Calories', 'Unknown')} kcal
- Protein: {analysis.get('Protein', 'Unknown')} g
- Carbs: {analysis.get('Carbs', 'Unknown')} g
- Fats: {analysis.get('Fats', 'Unknown')} g

USER WEEKLY TRENDS (LAST 7 LOGS):
{weekly_stats if weekly_stats else "No history recorded yet."}

USER DIET PLAN:
{diet_plan if diet_plan else "No diet plan yet."}

USER QUESTION:
{message}

INSTRUCTIONS:
- If the user asks for a data interpretation (BMI, Calories, etc.), provide a concise, insightful summary.
- If the user asks to explain a meal, explain using the diet plan.
- If the user asks for a "weekly report" or "how was my week", query the database through your context (the app sends the recent 7 days of data). Summarize their calorie and macro trends.
- If the user asks about a "cheat day", explain how to incorporate it safely (e.g., 80/20 rule) and suggest adjustments for the rest of the week based on their specific goal ({analysis.get('goal', 'maintenance')}).
- Be friendly, motivating, and practical.
- ALWAYS address the user by their name ({analysis.get('name', 'Friend')}) at the start.
- Keep responses relatively brief (max 3-4 sentences for dashboard insights).
- Do NOT give medical diagnosis. If needed, advise consulting a doctor.
"""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        # Use robust call
        res_data = call_gemini(payload)

        if "candidates" not in res_data or not res_data["candidates"]:
            return jsonify({"error": "Bella is unable to respond right now."}), 500

        reply = res_data["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": f"Bella Error: {str(e)}"}), 500


@app.route("/<path:path>")
def send_static(path):
    return send_from_directory(STATIC_DIR, path)

if __name__=="__main__":
    print("Server starting at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
