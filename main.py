import os
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents

import httpx

# Optional: OpenAI for chat assistant
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Wellbeing MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utility helpers
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -----------------------------
# Schemas for requests
# -----------------------------

class RegisterRequest(BaseModel):
    role: str
    name: str
    email: Optional[str] = None
    parentId: Optional[str] = None
    pin: Optional[str] = None


class LoginRequest(BaseModel):
    role: str
    email: Optional[str] = None
    pin: Optional[str] = None
    userId: Optional[str] = None


class ChatRequest(BaseModel):
    userId: str
    text: str


class TimerSetRequest(BaseModel):
    childId: str
    dailyLimit: int
    sessionLimit: int


class SocialLinkRequest(BaseModel):
    childId: str
    provider: str
    handle: str


# -----------------------------
# Root and health
# -----------------------------

@app.get("/")
def read_root():
    return {"message": "Wellbeing MVP API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:100]}"
    return response


# -----------------------------
# Auth (very simple MVP)
# -----------------------------

@app.post("/auth/register")
def register(req: RegisterRequest):
    role = req.role.lower()
    if role not in ["parent", "child"]:
        raise HTTPException(status_code=400, detail="role must be 'parent' or 'child'")
    user_doc = {
        "role": role,
        "name": req.name,
        "email": req.email,
        "parentId": req.parentId,
        "children": [],
        "pin": req.pin,
        "created_at": now_iso(),
    }
    user_id = create_document("user", user_doc)
    # Link child to parent if provided
    if role == "child" and req.parentId:
        try:
            from bson import ObjectId
            db["user"].update_one({"_id": ObjectId(req.parentId)}, {"$addToSet": {"children": user_id}})
        except Exception:
            pass
    return {"userId": user_id, "role": role}


@app.post("/auth/login")
def login(req: LoginRequest):
    # Parent: by email, Child: by userId or PIN (MVP only)
    query: Dict[str, Any] = {}
    if req.userId:
        try:
            from bson import ObjectId
            query["_id"] = ObjectId(req.userId)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid userId")
    elif req.role == "parent" and req.email:
        query = {"role": "parent", "email": req.email}
    elif req.role == "child" and req.pin:
        query = {"role": "child", "pin": req.pin}
    else:
        raise HTTPException(status_code=400, detail="Provide userId, or email (parent), or pin (child)")

    user = db["user"].find_one(query)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"userId": str(user["_id"]), "role": user.get("role"), "name": user.get("name")}


# -----------------------------
# Positive content (static lists for MVP)
# -----------------------------

AFFIRMATIONS = [
    "You are brave and capable.",
    "Every day you learn and grow.",
    "You can do hard things.",
]
QUOTES = [
    "The secret of getting ahead is getting started. — Mark Twain",
    "It always seems impossible until it's done. — Nelson Mandela",
    "Believe you can and you're halfway there. — Theodore Roosevelt",
]
JOKES = [
    "Why did the kid bring a ladder to school? Because they wanted to go to high school!",
    "What do you call a bear with no teeth? A gummy bear!",
    "Why was the math book sad? It had too many problems!",
]

@app.get("/content/daily")
def daily_content():
    import random
    return {
        "affirmation": random.choice(AFFIRMATIONS),
        "quote": random.choice(QUOTES),
        "joke": random.choice(JOKES),
    }


# -----------------------------
# Simple emotion + risk analyzer (rule-based MVP)
# -----------------------------

NEGATIVE_WORDS = {
    "sad", "upset", "angry", "anxious", "lonely", "tired", "worried", "scared", "depressed",
}
POSITIVE_WORDS = {"happy", "excited", "proud", "grateful", "calm", "good", "love"}
SELF_HARM_WORDS = {
    "suicide", "kill myself", "end it", "self harm", "cut myself", "die", "hurt myself",
}
EMOTION_MAP = {
    "joy": POSITIVE_WORDS,
    "sadness": {"sad", "depressed", "cry", "lonely"},
    "anger": {"angry", "mad", "hate"},
    "fear": {"scared", "afraid", "worried", "anxious"},
}


def analyze_text(text: str) -> Dict[str, Any]:
    t = text.lower()
    emotions: List[str] = []
    for label, words in EMOTION_MAP.items():
        if any(w in t for w in words):
            emotions.append(label)
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    sentiment = "positive" if pos > neg else ("negative" if neg > pos else "neutral")

    risk = 0.0
    if any(w in t for w in SELF_HARM_WORDS):
        risk += 80
    risk += min(20, neg * 5)
    risk -= min(10, pos * 3)
    risk = clamp(risk, 0, 100)

    return {"emotions": emotions, "sentiment": sentiment, "risk": risk}


# -----------------------------
# OpenAI-powered chat (upgrade path)
# -----------------------------

async def llm_reply(messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        # fallback to template if no key configured
        last = messages[-1]["content"] if messages else ""
        return generate_supportive_reply(last, analyze_text(last))

    # Using OpenAI responses API via httpx for portability
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "You are a kid-safe wellbeing assistant. Be brief, supportive, and avoid medical advice. Encourage reaching out to a trusted adult for serious issues."},
            *messages,
        ],
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


# -----------------------------
# Chat endpoint with analysis and risk scoring
# -----------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    # Analyze message
    analysis = analyze_text(req.text)

    # Save user message
    msg_doc = {
        "userId": req.userId,
        "role": "user",
        "text": req.text,
        "emotions": analysis["emotions"],
        "sentiment": analysis["sentiment"],
        "risk": analysis["risk"],
        "created_at": now_iso(),
    }
    _ = create_document("message", msg_doc)

    # Update rolling risk
    score = update_risk(req.userId, analysis["risk"])

    # Generate a supportive assistant reply (OpenAI if configured)
    reply_text = await llm_reply([{ "role": "user", "content": req.text }])
    reply_doc = {
        "userId": req.userId,
        "role": "assistant",
        "text": reply_text,
        "emotions": ["support"],
        "sentiment": "positive",
        "risk": 0.0,
        "created_at": now_iso(),
    }
    _ = create_document("message", reply_doc)

    # Check alert thresholds
    alert_level = maybe_create_alert(req.userId, score)

    return {
        "reply": reply_text,
        "analysis": analysis,
        "riskScore": score,
        "alertLevel": alert_level,
    }


def generate_supportive_reply(text: str, analysis: Dict[str, Any]) -> str:
    if analysis["risk"] >= 80:
        return (
            "I'm really sorry you're feeling this way. You are not alone and you deserve help. "
            "I'm alerting your parent/guardian so they can support you. If you are in immediate danger, "
            "please call your local emergency number right now."
        )
    if analysis["sentiment"] == "negative":
        return (
            "Thanks for sharing how you feel. Let's try a 30-second breathing exercise together. "
            "Inhale for 4, hold for 4, exhale for 6. You are safe here. Would you like a joke or an affirmation?"
        )
    if analysis["sentiment"] == "neutral":
        return (
            "Got it! I'm here to help with anything. Want to do a quick focus session or hear a quote?"
        )
    return "That's awesome! I'm proud of you. Want to keep the good vibes going with a fun joke or a new goal?"


# -----------------------------
# Risk state management and alerts
# -----------------------------

def update_risk(user_id: str, current_risk: float) -> float:
    existing = db["riskstate"].find_one({"userId": user_id})
    decay = 0.85  # keep some memory
    new_score = current_risk if not existing else clamp(existing.get("score", 0) * decay + current_risk * (1 - decay) + (10 if current_risk >= 60 else 0), 0, 100)
    db["riskstate"].update_one(
        {"userId": user_id},
        {"$set": {"userId": user_id, "score": new_score, "updated_at": now_iso()}, "$push": {"history": {"$each": [new_score], "$slice": -50}}},
        upsert=True,
    )
    return float(new_score)


@app.get("/risk/{user_id}")
def get_risk(user_id: str):
    state = db["riskstate"].find_one({"userId": user_id})
    return {"userId": user_id, "score": float(state.get("score", 0)) if state else 0.0}


def maybe_create_alert(user_id: str, score: float) -> Optional[str]:
    # Thresholds
    level: Optional[str] = None
    if score >= 80:
        level = "critical"
    elif score >= 60:
        level = "concern"
    elif score >= 30:
        level = "info"

    if not level:
        return None

    # Check cooldown (12h)
    rs = db["riskstate"].find_one({"userId": user_id})
    last = None
    if rs and rs.get("lastAlertAt"):
        try:
            last = datetime.fromisoformat(rs["lastAlertAt"])
        except Exception:
            last = None
    if last and datetime.now(timezone.utc) - last < timedelta(hours=12) and level != "critical":
        return level

    alert = {
        "userId": user_id,
        "level": level,
        "summary": f"Risk level {level.upper()} detected at {now_iso()} with score {int(score)}.",
        "channels": ["log"],
        "sentAt": now_iso(),
        "acknowledged": False,
    }
    _ = create_document("alert", alert)
    db["riskstate"].update_one({"userId": user_id}, {"$set": {"lastAlertAt": now_iso()}})
    return level


# -----------------------------
# Timer endpoints (in-app limits)
# -----------------------------

@app.post("/timers/set")
def set_timer(req: TimerSetRequest):
    # Upsert timer document for child
    doc = {
        "childId": req.childId,
        "dailyLimit": req.dailyLimit,
        "sessionLimit": req.sessionLimit,
        "updated_at": now_iso(),
    }
    db["timer"].update_one({"childId": req.childId}, {"$set": doc}, upsert=True)
    return {"ok": True}


@app.get("/timers/{child_id}")
def get_timer(child_id: str):
    t = db["timer"].find_one({"childId": child_id})
    if not t:
        return {"childId": child_id, "dailyLimit": 60, "sessionLimit": 20}
    return {
        "childId": child_id,
        "dailyLimit": int(t.get("dailyLimit", 60)),
        "sessionLimit": int(t.get("sessionLimit", 20)),
    }


# -----------------------------
# Social links and activity
# -----------------------------

@app.post("/social/link")
def link_social(req: SocialLinkRequest):
    # Upsert a social link document per child+provider
    doc = {
        "childId": req.childId,
        "provider": req.provider.lower(),
        "handle": req.handle,
        "updated_at": now_iso(),
    }
    db["sociallink"].update_one(
        {"childId": req.childId, "provider": doc["provider"]},
        {"$set": doc},
        upsert=True,
    )
    return {"ok": True}


@app.get("/social/{child_id}/links")
def get_social_links(child_id: str):
    links = list(db["sociallink"].find({"childId": child_id}))
    return [
        {"childId": l.get("childId"), "provider": l.get("provider"), "handle": l.get("handle")}
        for l in links
    ]


# MVP mock scan (existing)
@app.post("/social/{child_id}/scan")
def scan_social(child_id: str):
    links = list(db["sociallink"].find({"childId": child_id}))
    if not links:
        raise HTTPException(status_code=404, detail="No social links configured for this child")

    samples = [
        "Had a great day at school, proud of my project!",
        "Feeling lonely today...",
        "That test made me so anxious",
        "I love my new bike",
        "Sometimes I just want to disappear",
        "Hanging out with friends made me happy",
    ]
    import random
    activities: List[Dict[str, Any]] = []
    for l in links:
        for _ in range(3):
            text = random.choice(samples)
            a = analyze_text(text)
            activities.append({
                "childId": child_id,
                "provider": l.get("provider"),
                "handle": l.get("handle"),
                "text": text,
                "analysis": a,
                "risk": a["risk"],
                "created_at": now_iso(),
            })
    if activities:
        db["socialactivity"].insert_many(activities)

    avg_risk = sum(act["risk"] for act in activities) / max(1, len(activities))
    _ = update_risk(child_id, avg_risk)
    return {"count": len(activities), "avgRisk": avg_risk}


# Instagram scraping placeholder: requires official API in production
@app.post("/social/{child_id}/scan/instagram")
def scan_instagram(child_id: str, handle: str = Form(...)):
    # This endpoint documents the legit path: use Meta Graph API with user consent.
    # Here we store an intent record and return a message instructing to connect OAuth in future phase.
    create_document("socialscanrequest", {"childId": child_id, "provider": "instagram", "handle": handle, "created_at": now_iso()})
    return {"ok": True, "message": "Instagram scan requires OAuth (Meta Graph API). Connect account to enable real scans."}


@app.get("/social/{child_id}/activity")
def get_social_activity(child_id: str):
    cur = db["socialactivity"].find({"childId": child_id}).sort("_id", -1).limit(15)
    items = list(cur)
    return [
        {
            "provider": it.get("provider"),
            "handle": it.get("handle"),
            "text": it.get("text"),
            "analysis": it.get("analysis", {}),
            "risk": float(it.get("risk", 0)),
            "created_at": it.get("created_at"),
        }
        for it in items
    ]


# -----------------------------
# Safety: Deepfake detection (placeholder)
# -----------------------------

@app.post("/safety/deepfake/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    # Placeholder: basic metadata-based heuristic. In production, integrate a vision model/API.
    content = await file.read()
    size_kb = len(content) / 1024
    ext = (file.filename or "").split(".")[-1].lower()
    score = 0.0
    # naive heuristics
    if ext in {"mp4", "mov", "avi", "mkv"}:
        score += 15
    if ext in {"jpg", "jpeg", "png", "webp"}:
        score += 5
    if size_kb > 5120:  # 5MB
        score += 10
    score = clamp(score, 0, 100)
    result = {"filename": file.filename, "sizeKB": round(size_kb, 1), "suspicion": score, "label": "likely" if score>=60 else ("uncertain" if score>=30 else "unlikely")}
    create_document("deepfakedetect", {**result, "created_at": now_iso()})
    return result


# -----------------------------
# Reporting
# -----------------------------

class ReportRequest(BaseModel):
    reporterId: str
    targetUserId: Optional[str] = None
    reason: str
    note: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@app.post("/report")
def submit_report(req: ReportRequest):
    doc = {
        "reporterId": req.reporterId,
        "targetUserId": req.targetUserId,
        "reason": req.reason,
        "note": req.note,
        "context": req.context or {},
        "created_at": now_iso(),
        "status": "open",
    }
    report_id = create_document("report", doc)
    return {"ok": True, "reportId": report_id}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
