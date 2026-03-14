from flask import Flask, render_template, request, redirect, session
from datetime import datetime
from textblob import TextBlob
import sqlite3
import joblib
import re


app = Flask(__name__)
app.secret_key = "cyberbullying_secret"


# -----------------------------
# DATABASE INITIALIZATION
# -----------------------------

def init_db():

    conn = sqlite3.connect("comments.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        time TEXT,
        message TEXT,
        result TEXT,
        score INTEGER,
        sentiment TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    # CHAT TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        message TEXT,
        time TEXT
    )
    """)

    cursor.execute("SELECT * FROM users WHERE username='admin'")

    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username,password,role) VALUES (?,?,?)",
            ("admin","admin123","admin")
        )

    conn.commit()
    conn.close()


init_db()


# -----------------------------
# LOAD ML MODEL
# -----------------------------

try:
    model = joblib.load("model/bullying_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    pca = joblib.load("model/pca_transform.pkl")
    ml_enabled = True
except:
    ml_enabled = False


# -----------------------------
# TOXIC WORD DICTIONARY
# -----------------------------

toxic_words = {
    "stupid": "Insult",
    "idiot": "Insult",
    "loser": "Harassment",
    "hate": "Hate Speech",
    "useless": "Harassment",
    "dumb": "Insult",
    "trash": "Harassment",
    "pathetic": "Harassment",
    "worst": "Harassment",
    "fool": "Insult",
    "moron": "Insult"
}


# -----------------------------
# LOGIN ROUTE
# -----------------------------

@app.route("/login", methods=["GET","POST"])
def login():

    if "user" in session:
        return redirect("/")

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("comments.db")
        cursor = conn.cursor()

        cursor.execute(
        "SELECT username,role FROM users WHERE username=? AND password=?",
        (username,password)
        )

        user = cursor.fetchone()

        conn.close()

        if user:

            session["user"] = user[0]
            session["role"] = user[1]

            return redirect("/")

        else:
            return render_template("login.html", error="Invalid login")

    return render_template("login.html")

# -----------------------------
# ML PREDICTION
# -----------------------------

if ml_enabled:

    cleaned = preprocess(message)

    vector = vectorizer.transform([cleaned])

    pred = model.predict(vector)

    predicted_class = label_encoder.inverse_transform(pred)[0]

    reasons.append(f"ML Prediction: {predicted_class}")

    # Simple toxicity score
    if predicted_class != "not_cyberbullying":

        result = "Cyberbullying Detected"
        category = predicted_class
        score = 85
        confidence = 90

    else:

        result = "Safe Message"
        category = "None"
        score = 5
        confidence = 95
# -----------------------------
# REGISTER ROUTE
# -----------------------------

@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("comments.db")
        cursor = conn.cursor()

        try:

            cursor.execute(
                "INSERT INTO users (username,password,role) VALUES (?,?,?)",
                (username,password,"user")
            )

            conn.commit()

            return redirect("/login")

        except:

            return render_template("register.html", error="Username already exists")

        finally:
            conn.close()

    return render_template("register.html")


# -----------------------------
# LOGOUT
# -----------------------------

@app.route("/logout")
def logout():

    session.pop("user", None)
    return redirect("/login")


# -----------------------------
# MAIN PAGE
# -----------------------------

@app.route("/", methods=["GET","POST"])
def home():

    if "user" not in session:
        return redirect("/login")

    result = ""
    category = ""
    sentiment = ""
    score = 0
    confidence = 0
    risk_level = ""
    recommendation = ""
    bar_class = "low"
    reasons = []

    if request.method == "POST":

        message = request.form["message"]
        lower_message = message.lower()

        analysis = TextBlob(message)
        polarity = analysis.sentiment.polarity

        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        words = lower_message.split()
        detected_words = []

        for word in words:
            if word in toxic_words:
                detected_words.append(word)
                reasons.append(f"Toxic word detected: {word}")

        if detected_words:

            result = "Cyberbullying Detected"
            category = toxic_words[detected_words[0]]

            score = min(95, 60 + len(detected_words) * 10)
            confidence = min(98, 70 + len(detected_words) * 8)

        else:

            result = "Safe Message"
            category = "None"
            score = 5
            confidence = 95

        if score <= 30:
            risk_level = "Low"
            bar_class = "low"
            recommendation = "Allow message"

        elif score <= 70:
            risk_level = "Medium"
            bar_class = "medium"
            recommendation = "Send warning to user"

        else:
            risk_level = "High"
            bar_class = "high"
            recommendation = "Hide comment and alert moderator"

        conn = sqlite3.connect("comments.db")
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO comments (username,time,message,result,score,sentiment) VALUES (?,?,?,?,?,?)",
            (
                session["user"],
                datetime.now().strftime("%H:%M:%S"),
                message,
                result,
                score,
                sentiment
            )
        )

        conn.commit()
        conn.close()

    # LOAD CHAT MESSAGES
    conn = sqlite3.connect("comments.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT username,message,time
    FROM chat
    ORDER BY id DESC
    LIMIT 20
    """)

    chat_messages = cursor.fetchall()

    conn.close()

    return render_template(
        "index.html",
        result=result,
        category=category,
        sentiment=sentiment,
        score=score,
        confidence=confidence,
        risk_level=risk_level,
        recommendation=recommendation,
        bar_class=bar_class,
        reasons=reasons,
        chat_messages=chat_messages
    )


# -----------------------------
# CHAT ROUTE
# -----------------------------

@app.route("/chat", methods=["POST"])
def chat():

    if "user" not in session:
        return redirect("/login")

    message = request.form["chat_message"]

    conn = sqlite3.connect("comments.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO chat (username,message,time) VALUES (?,?,?)",
        (session["user"], message, datetime.now().strftime("%H:%M:%S"))
    )

    conn.commit()
    conn.close()

    return redirect("/")


# -----------------------------
# USER HISTORY
# -----------------------------

@app.route("/my_history")
def my_history():

    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("comments.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT time,message,result,score
    FROM comments
    WHERE username=?
    ORDER BY id DESC
    """,(session["user"],))

    history = cursor.fetchall()

    conn.close()

    return render_template("history.html", history=history)


# -----------------------------
# DASHBOARD
# -----------------------------

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return "Access Denied"

    conn = sqlite3.connect("comments.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM comments")
    total_comments = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM comments WHERE result='Cyberbullying Detected'")
    bullying_detected = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM comments WHERE result='Safe Message'")
    safe_messages = cursor.fetchone()[0]

    cursor.execute("""
    SELECT username,time,message,result,score
    FROM comments
    ORDER BY id DESC
    """)

    activities = cursor.fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total_comments=total_comments,
        bullying_detected=bullying_detected,
        safe_messages=safe_messages,
        activities=activities
    )


# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)