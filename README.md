# AI Teacher Backend v2.0

## Required Environment Variables (Render pe add karo)

| Variable | Value | Kahan se milega |
|----------|-------|----------------|
| `GROQ_API_KEY` | `gsk_...` | console.groq.com |
| `TAVILY_API_KEY` | `tvly-...` | tavily.com |
| `SECRET_KEY` | any random string | Khud banao |
| `QUIZ_SECRET` | `quiz_bridge_2025` | Same yahan se copy karo |
| `TEAMCOIN_URL` | `https://teamcoin-backend.onrender.com` | Tumhara TC backend URL |

## Deploy on Render
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --timeout 120 --workers 1`
- Runtime: Python 3
