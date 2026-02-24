import os
import mariadb
from flask import Flask, request, jsonify, render_template
from google import genai 
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Initialize Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={'api_version': 'v1'}
)

DB_CONFIG = {
    "host": "localhost",
    "user": "admin",
    "password": "Omi9321",
    "database": "analytics"
}

def execute_safe_sql(query):
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER"]
    if any(word in query.upper() for word in forbidden):
        return "Error: Security violation - Read-only allowed."
    try:
        conn = mariadb.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)
        cur.execute(query)
        result = cur.fetchall()
        conn.close()
        return result
    except Exception as e:
        return f"DB Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('question')
    
    try:
        # Step 1: SQL Generation
        # FIXED: Remove 'models/' prefix from model name
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f"You are a MariaDB expert. Translate this to SQL: {user_question}. Table: sales (product_name, revenue, country). Return ONLY SQL."
        )
        generated_sql = response.text.strip().replace('```sql', '').replace('```', '')

        # Step 2: DB Query
        db_results = execute_safe_sql(generated_sql)
        
        # Step 3: Human Summary
        summary_resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"The user asked '{user_question}' and the database returned {db_results}. Explain this to them briefly."
        )
        
        return jsonify({
            "sql": generated_sql, 
            "results": db_results, 
            "summary": summary_resp.text
        })
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)