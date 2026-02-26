"""
AI Teacher Backend — Complete v2.0
Features: Auth, Quiz, Ask, Search, Leaderboard, TeamCoin Bridge
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from gtts import gTTS
import os, io, base64, json, re, hashlib, sqlite3, random, string, requests
from datetime import datetime, timedelta, date
from functools import wraps
from bs4 import BeautifulSoup

try:
    import jwt
except ImportError:
    import PyJWT as jwt

try:
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
    TAVILY_OK = True
except:
    TAVILY_OK = False

app = Flask(__name__)
CORS(app, origins="*")

# ===== CONFIG =====
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
SECRET_KEY      = os.environ.get("SECRET_KEY",   "ai_teacher_secret_2025")
QUIZ_SECRET     = os.environ.get("QUIZ_SECRET",  "quiz_bridge_2025")
TEAMCOIN_URL    = os.environ.get("TEAMCOIN_URL",  "https://teamcoin-backend.onrender.com")
DB_PATH         = "ai_teacher.db"
SCORES_FILE     = "scores.json"

groq_client = Groq(api_key=GROQ_API_KEY)

# ===== DATABASE =====
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT NOT NULL,
        email      TEXT UNIQUE NOT NULL,
        password   TEXT NOT NULL,
        avatar     TEXT DEFAULT '👤',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        last_login TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_topics (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id  INTEGER NOT NULL,
        course   TEXT NOT NULL,
        topic    TEXT NOT NULL,
        added_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# ===== AUTH HELPERS =====
def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def make_token(uid):
    payload = {"uid": uid, "exp": datetime.utcnow() + timedelta(days=30)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization","").replace("Bearer ","")
        if not token:
            return jsonify({"error":"Login required"}), 401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.uid = data["uid"]
        except:
            return jsonify({"error":"Invalid or expired token"}), 401
        return f(*args, **kwargs)
    return decorated

# ===== GROQ CALL WITH FALLBACK =====
def groq_call(messages, temperature=0.4, max_tokens=6000):
    """Groq API call — returns text or raises exception."""
    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768"
    ]
    last_error = None
    for model in models:
        try:
            resp = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"Model {model} failed: {e}")
            continue
    raise Exception(f"All models failed: {last_error}")

# ===== SEARCH HELPERS =====
def search_tavily(query):
    if not TAVILY_OK: return []
    try:
        results = tavily_client.search(query=query, search_depth="basic", max_results=5)
        return [{"title": r.get("title",""), "snippet": r.get("content","")[:200]+"...", "url": r.get("url","")}
                for r in results.get("results", [])]
    except: return []

def search_duckduckgo(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}", headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        titles   = soup.select("a.result__a")[:5]
        snippets = soup.select("a.result__snippet")[:5]
        return [{"title": t.get_text(strip=True), "snippet": snippets[i].get_text(strip=True) if i < len(snippets) else "", "url": t.get("href","")}
                for i, t in enumerate(titles)]
    except: return []

def search_bing(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(f"https://www.bing.com/search?q={requests.utils.quote(query)}", headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select("li.b_algo")[:5]:
            t = item.select_one("h2 a")
            s = item.select_one("p")
            if t: results.append({"title": t.get_text(strip=True), "snippet": s.get_text(strip=True) if s else "", "url": t.get("href","")})
        return results
    except: return []

def smart_search(query):
    for fn, name in [(search_tavily, "Tavily"), (search_duckduckgo, "DuckDuckGo"), (search_bing, "Bing")]:
        r = fn(query)
        if r:
            print(f"✅ Search via {name}")
            return r, name
    return [], "None"

import re
import io
import base64
from gtts import gTTS
from pydub import AudioSegment

def text_to_speech(text):
    try:
        clean = re.sub(r'[*#_`]', '', text)

        chunk_size = 1000
        chunks = [clean[i:i+chunk_size] for i in range(0, len(clean), chunk_size)]

        final_audio = AudioSegment.empty()

        for chunk in chunks:
            tts = gTTS(text=chunk, lang="hi", slow=False)
            temp_buf = io.BytesIO()
            tts.write_to_fp(temp_buf)
            temp_buf.seek(0)

            segment = AudioSegment.from_file(temp_buf, format="mp3")
            final_audio += segment

        output_buf = io.BytesIO()
        final_audio.export(output_buf, format="mp3")
        output_buf.seek(0)

        return base64.b64encode(output_buf.read()).decode("utf-8")

    except Exception as e:
        print("Error:", e)
        return ""

# ===== SCORES =====
def load_scores():
    try:
        if os.path.exists(SCORES_FILE):
            with open(SCORES_FILE,"r",encoding="utf-8") as f: return json.load(f)
    except: pass
    return []

def save_scores(scores):
    try:
        with open(SCORES_FILE,"w",encoding="utf-8") as f: json.dump(scores, f, ensure_ascii=False)
    except: pass

# ========================================
# ROUTES
# ========================================

@app.route("/")
def home():
    return jsonify({"status": "AI Teacher v2.0 — Running! 🚀", "time": str(datetime.now())})

@app.route("/ping")
def ping():
    return jsonify({"status": "awake", "time": str(datetime.now())})

# ===== AUTH =====
@app.route("/register", methods=["POST"])
def register():
    try:
        d        = request.json
        name     = d.get("name","").strip()
        email    = d.get("email","").strip().lower()
        password = d.get("password","")
        avatar   = d.get("avatar","👤")

        if not name or not email or not password:
            return jsonify({"error":"सभी fields भरें"}), 400
        if len(password) < 6:
            return jsonify({"error":"Password कम से कम 6 characters का हो"}), 400
        if "@" not in email:
            return jsonify({"error":"सही Email डालो"}), 400

        conn = get_db()
        if conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
            conn.close()
            return jsonify({"error":"यह Email पहले से registered है"}), 400

        conn.execute("INSERT INTO users (name,email,password,avatar) VALUES (?,?,?,?)",
                     (name, email, hash_pw(password), avatar))
        uid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit(); conn.close()

        return jsonify({"token": make_token(uid), "name": name, "message": f"स्वागत है {name}! 🎉"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        d        = request.json
        email    = d.get("email","").strip().lower()
        password = d.get("password","")

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=? AND password=?",
                            (email, hash_pw(password))).fetchone()
        if not user:
            conn.close()
            return jsonify({"error":"Email या Password गलत है"}), 401

        today = str(date.today())
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (today, user["id"]))
        conn.commit(); conn.close()

        return jsonify({"token": make_token(user["id"]), "name": user["name"], "avatar": user["avatar"],
                        "message": f"वापस आए {user['name']}! 🙏"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/me", methods=["GET"])
@auth_required
def get_me():
    try:
        conn = get_db()
        user = conn.execute("SELECT id,name,email,avatar,created_at,last_login FROM users WHERE id=?",
                            (request.uid,)).fetchone()
        conn.close()
        return jsonify({"user": dict(user)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== SEARCH =====
@app.route("/search", methods=["POST"])
def search():
    try:
        query = request.json.get("query","").strip()
        if not query: return jsonify({"results":[]})

        results, source = smart_search(query)

        # Wikipedia pehle
        wiki = [r for r in results if "wikipedia.org" in r.get("url","")]
        rest = [r for r in results if "wikipedia.org" not in r.get("url","")]

        if not wiki:
            w2, _ = smart_search(f"{query} wikipedia")
            wiki  = [r for r in w2 if "wikipedia.org" in r.get("url","")][:1]

        if wiki: wiki[0]["is_wiki"] = True
        final = wiki + rest

        return jsonify({"results": final, "source": source})
    except Exception as e:
        return jsonify({"results":[], "error": str(e)}), 500

# ===== ASK =====
@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question","").strip()
        if not question: return jsonify({"error":"Question required"}), 400

        web, src = smart_search(question)
        web_text = "\n".join([f"{i+1}. {r['title']}: {r['snippet']}" for i,r in enumerate(web[:4])]) if web else ""

        system = f"""अगर कोई आपका नाम पूछे तो आप राजू राम हैं — एक दोस्ताना AI Teacher।
भारतीय गुरुजी की तरह प्यार से समझाएं। हिंदी में जवाब दें।
उदाहरण भारतीय context से दें (क्रिकेट, बॉलीवुड, त्योहार, राजनीति)।
जवाब 150-200 शब्दों में दें। Simple aur clear भाषा।
{f'Web Search Results ({src}): {web_text}' if web_text else ''}"""

        answer = groq_call(
            [{"role":"system","content":system}, {"role":"user","content":question}],
            temperature=0.5, max_tokens=800
        )
        # Audio alag endpoint pe — main response fast rakho
        return jsonify({"answer": answer, "source": src})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== TTS (alag endpoint — fast response ke liye) =====
@app.route("/tts", methods=["POST"])
def tts():
    try:
        text  = request.json.get("text","").strip()[:300]
        if not text: return jsonify({"audio":""})
        audio = text_to_speech(text)
        return jsonify({"audio": audio})
    except Exception as e:
        return jsonify({"audio":"", "error":str(e)})

# ===== TOPIC EXPLAIN =====
@app.route("/explain", methods=["POST"])
def explain():
    try:
        d     = request.json
        topic  = d.get("topic","").strip()
        course = d.get("course","").strip()
        if not topic: return jsonify({"error":"Topic required"}), 400

        system = f"""आप एक expert Indian teacher हैं।
'{course}' subject में '{topic}' topic को 500 से कम शब्दों में explain करो।
Structure:
1. परिभाषा और Introduction (100 words)
2. मुख्य Concepts (300 words)  
3. Real-life भारतीय Examples (200 words)
4. Important Facts & Dates (200 words)
5. परीक्षा के लिए Key Points (200 words)
6. explanation में * (तारांकन) का प्रयोग नहीं करना है।
हिंदी में लिखो। Technical terms English में रख सकते हो।
Engaging और आसान भाषा use करो।"""

        text = groq_call(
            [{"role":"system","content":system}, {"role":"user","content":f"{topic} के बारे में बताओ"}],
            temperature=0.4, max_tokens=2000
        )
        audio = text_to_speech(text[:800])
        return jsonify({"text": text, "audio": audio})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== QUIZ =====
@app.route("/quiz", methods=["POST"])
def generate_quiz():
    try:
        topic = request.json.get("topic","").strip()
        if not topic: return jsonify({"error":"Topic required"}), 400

        system = """You are a senior question paper setter for Indian competitive exams (UPSC, SSC, NEET, JEE, Railways).

BILINGUAL FORMAT (IMPORTANT):
- Write EVERY question in BOTH Hindi AND English
- Format: "हिंदी प्रश्न? (English question?)"
- Write EVERY option in BOTH languages: "हिंदी विकल्प (English option)"
- Write explanation in both languages

QUESTION QUALITY:
1. All 4 options same category (if answer is a year, all options are years)
2. No "उपरोक्त सभी / All of the above" options
3. Hard questions must be DETAILED and LENGTHY (2-3 lines)
4. Correct answer position randomly distributed (mix of 0,1,2,3)
5. Explanation must be specific with facts/years

DIFFICULTY:
- आसान/Easy (5 Qs): Class 8-10 level
- मध्यम/Medium (5 Qs): Class 11-12 / 1 year exam prep
- कठिन/Hard (5 Qs): UPSC/JEE/NEET serious aspirant level

Return ONLY valid JSON array. No other text."""

        prompt = f"""Topic: "{topic}"

Generate 15 bilingual MCQ questions.
First 5: level "आसान/Easy"
Next 5: level "मध्यम/Medium"  
Last 5: level "कठिन/Hard"

JSON format:
[{{"q":"हिंदी प्रश्न? (English question?)",
  "options":["विकल्प A (Option A)","विकल्प B (Option B)","विकल्प C (Option C)","विकल्प D (Option D)"],
  "ans":1,
  "level":"मध्यम/Medium",
  "explanation":"हिंदी explanation. English explanation."
}}]

Return ONLY the JSON array."""

        raw = groq_call(
            [{"role":"system","content":system}, {"role":"user","content":prompt}],
            temperature=0.4, max_tokens=6000
        )

        # Parse JSON
        raw = re.sub(r'```json\s*','',raw)
        raw = re.sub(r'```\s*','',raw).strip()
        s   = raw.find('['); e = raw.rfind(']')
        if s != -1 and e != -1: raw = raw[s:e+1]

        questions = json.loads(raw)
        valid = []
        for q in questions:
            if (isinstance(q,dict) and "q" in q and "options" in q and "ans" in q
                and isinstance(q["options"],list) and len(q["options"])==4
                and isinstance(q["ans"],int) and 0<=q["ans"]<=3):
                valid.append({
                    "q":           str(q["q"]).strip(),
                    "options":     [str(o).strip() for o in q["options"]],
                    "ans":         int(q["ans"]),
                    "level":       str(q.get("level","मध्यम/Medium")),
                    "explanation": str(q.get("explanation","")).strip()
                })

        if len(valid) < 5:
            return jsonify({"error":"Quiz generate नहीं हो पाई। फिर try करो।"}), 500

        return jsonify({"questions": valid, "topic": topic, "total": len(valid)})
    except json.JSONDecodeError:
        return jsonify({"error":"Quiz parse error। दोबारा try करो।"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== SAVE SCORE =====
@app.route("/save-score", methods=["POST"])
def save_score():
    try:
        d     = request.json
        name  = str(d.get("name","")).strip()
        score = int(d.get("score",0))
        total = int(d.get("total",15))
        topic = str(d.get("topic","")).strip()
        if not name: return jsonify({"error":"Name required"}), 400

        scores = load_scores()
        found  = False
        for e in scores:
            if e["name"].lower() == name.lower():
                if score > e["best_score"]:
                    e["best_score"]  = score
                    e["best_total"]  = total
                    e["best_topic"]  = topic
                    e["last_played"] = str(date.today())
                e["games_played"] = e.get("games_played",0) + 1
                found = True; break

        if not found:
            scores.append({"name":name,"best_score":score,"best_total":total,
                           "best_topic":topic,"games_played":1,"last_played":str(date.today())})
        save_scores(scores)
        return jsonify({"success":True})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ===== LEADERBOARD =====
@app.route("/leaderboard", methods=["GET"])
def leaderboard():
    try:
        scores = load_scores()
        scores.sort(key=lambda x: (-x["best_score"], x["name"]))
        top = scores[:20]
        for e in top:
            t = e.get("best_total",15)
            e["percentage"] = round(e["best_score"]/t*100) if t>0 else 0
        return jsonify({"leaderboard": top, "total_players": len(scores)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ===== CLAIM QUIZ COINS =====
@app.route("/claim-quiz-coins", methods=["POST"])
def claim_quiz_coins():
    try:
        d        = request.json
        username = str(d.get("username","")).strip()
        score    = int(d.get("score",0))
        total    = int(d.get("total",15))
        topic    = str(d.get("topic","")).strip()
        if not username: return jsonify({"error":"TeamCoin username डालो!"}), 400

        coins = score * 2 + (10 if total>0 and score/total>=0.8 else 0)
        if coins <= 0: return jsonify({"error":"Score 0 है — coins नहीं मिलेंगे"}), 400

        resp = requests.post(f"{TEAMCOIN_URL}/award-quiz-coins",
            json={"username":username,"coins":coins,"score":score,"total":total,"topic":topic,"secret":QUIZ_SECRET},
            timeout=20)
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.Timeout:
        return jsonify({"error":"TeamCoin backend respond नहीं कर रहा। थोड़ी देर बाद try करो।"}), 504
    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
