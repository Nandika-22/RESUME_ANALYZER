from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import PyPDF2

app = Flask(__name__)

# -------------------------
# DATASET
# -------------------------

data = {
"text":[
"python machine learning pandas numpy statistics data analysis",
"python deep learning tensorflow pytorch neural networks ai",
"java spring backend api microservices system design",
"html css javascript react frontend web ui design",
"sql excel power bi tableau business analytics",
"digital marketing seo content strategy branding",
"python sql statistics machine learning pandas",
"c++ algorithms operating systems data structures",
"figma ui ux user research prototyping"
],

"role":[
"Data Scientist",
"AI Engineer",
"Backend Developer",
"Frontend Developer",
"Data Analyst",
"Digital Marketer",
"Data Scientist",
"Software Engineer",
"UI UX Designer"
]
}

df = pd.DataFrame(data)

# -------------------------
# TRAIN MODEL
# -------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
model = MultinomialNB()
model.fit(X, df["role"])

# -------------------------
# SKILL DATABASE
# -------------------------

skills_db = [
"python","machine learning","deep learning","tensorflow",
"pytorch","sql","excel","power bi","tableau","statistics",
"java","spring","html","css","javascript","react",
"seo","marketing","figma","ui","ux","c++","pandas","numpy"
]

# -------------------------
# ROLE SKILLS
# -------------------------

role_skills = {

"Data Scientist":[
"python","machine learning","statistics","pandas","numpy","sql"
],

"AI Engineer":[
"python","deep learning","tensorflow","pytorch"
],

"Data Analyst":[
"sql","excel","power bi","tableau"
],

"Frontend Developer":[
"html","css","javascript","react"
],

"Backend Developer":[
"java","spring","api"
],

"UI UX Designer":[
"figma","ui","ux"
]
}

# -------------------------
# PDF TEXT EXTRACT
# -------------------------

def extract_pdf(file):

    reader = PyPDF2.PdfReader(file)
    text=""

    for page in reader.pages:
        text += page.extract_text()

    return text

# -------------------------
# SKILL EXTRACTION
# -------------------------

def extract_skills(text):

    text = text.lower()
    found=[]

    for skill in skills_db:
        if skill in text:
            found.append(skill)

    return found

# -------------------------
# SKILL GAP
# -------------------------

def skill_gap(role,skills):

    required = role_skills.get(role,[])
    missing=[]

    for s in required:
        if s not in skills:
            missing.append(s)

    return missing

# -------------------------
# CHATBOT RESPONSES
# -------------------------

def chatbot_response(message):

    message=message.lower()

    if "data scientist" in message:
        return "Data Scientists need Python, Machine Learning, Statistics, SQL, Pandas and Data Visualization."

    if "ai engineer" in message:
        return "AI Engineers require Deep Learning, TensorFlow, PyTorch, Python and Neural Networks."

    if "frontend" in message:
        return "Frontend developers use HTML, CSS, JavaScript, React and UI design."

    if "backend" in message:
        return "Backend developers use Java, Python, APIs, Databases and System Design."

    return "I recommend learning Python, Data Analysis, SQL and Machine Learning for strong tech careers."

# -------------------------
# MAIN PAGE
# -------------------------

HTML_PAGE = """

<html>
<head>

<title>Resume Analyzer 2.0</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

body{
font-family:Arial;
background:#0f172a;
color:white;
text-align:center;
}

.container{
width:80%;
margin:auto;
}

textarea{
width:80%;
height:120px;
}

button{
padding:10px 20px;
background:#22c55e;
border:none;
color:white;
font-size:16px;
cursor:pointer;
}

.dashboard{
display:flex;
justify-content:space-around;
margin-top:30px;
}

.card{
background:#1e293b;
padding:20px;
border-radius:10px;
width:30%;
}

.chatbox{
margin-top:40px;
}

</style>

</head>

<body>

<div class="container">

<h1>AI Resume Analyzer 2.0</h1>

<form method="POST" enctype="multipart/form-data">

<textarea name="resume_text" placeholder="Paste resume text"></textarea>

<br><br>

Upload PDF Resume
<input type="file" name="resume_file">

<br><br>

<button type="submit">Analyze Resume</button>

</form>

{% if role %}

<div class="dashboard">

<div class="card">
<h2>Predicted Role</h2>
<h3>{{role}}</h3>
</div>

<div class="card">
<h2>Confidence</h2>
<h3>{{confidence}}%</h3>
</div>

<div class="card">
<h2>Skills Found</h2>
<p>{{skills}}</p>
</div>

</div>

<br>

<h2>Missing Skills</h2>
<p>{{missing}}</p>

<br>

<canvas id="skillChart" width="400" height="200"></canvas>

<script>

var ctx = document.getElementById('skillChart').getContext('2d');

var chart = new Chart(ctx,{
type:'bar',
data:{
labels:{{skills|safe}},
datasets:[{
label:'Skill Score',
data:{{scores|safe}}
}]
}
});

</script>

{% endif %}

<div class="chatbox">

<h2>AI Career Chatbot</h2>

<input type="text" id="msg" placeholder="Ask career question">

<button onclick="sendMsg()">Ask</button>

<p id="reply"></p>

</div>

</div>

<script>

function sendMsg(){

let msg=document.getElementById("msg").value;

fetch("/chat",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({message:msg})
})

.then(res=>res.json())
.then(data=>{
document.getElementById("reply").innerText=data.reply
})

}

</script>

</body>
</html>

"""

# -------------------------
# ROUTE
# -------------------------

@app.route("/",methods=["GET","POST"])
def home():

    role=None
    skills=[]
    missing=[]
    confidence=0
    scores=[]

    if request.method=="POST":

        text=request.form["resume_text"]

        file=request.files["resume_file"]

        if file and file.filename!="":
            text=extract_pdf(file)

        vector=vectorizer.transform([text])

        prediction=model.predict(vector)[0]

        prob=model.predict_proba(vector).max()*100

        skills=extract_skills(text)

        missing=skill_gap(prediction,skills)

        confidence=round(prob,2)

        scores=[10]*len(skills)

        role=prediction

    return render_template_string(
    HTML_PAGE,
    role=role,
    skills=skills,
    missing=missing,
    confidence=confidence,
    scores=scores
    )

# -------------------------
# CHATBOT API
# -------------------------

@app.route("/chat",methods=["POST"])
def chat():

    data=request.get_json()

    message=data["message"]

    reply=chatbot_response(message)

    return jsonify({"reply":reply})

# -------------------------
# RUN APP
# -------------------------

if __name__=="__main__":
    app.run(debug=True)