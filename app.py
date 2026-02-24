import os
import mysql.connector  
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

# FIX: Added auth_plugin to resolve 'caching_sha2_password' error on MySQL 8+
DB_CONFIG = {
    "host": "localhost", 
    "user": "admin",
    "password": "sky1234",
    "database": "company_metrics",
    "auth_plugin": "mysql_native_password"   # <-- fix for MySQL 8+
}

def execute_safe_sql(query):
    # Basic security check for read-only access
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER"]
    if any(word in query.upper() for word in forbidden):
        return "Error: Security violation - Read-only allowed."
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)
        cur.execute(query)
        result = cur.fetchall()
        cur.close()
        conn.close()
        return result
    except Exception as e:
        return f"Database Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('question')
    
    try:
        # Step 1: SQL Generation via Gemini
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f"You are a MySQL expert. Translate this to a SQL query: '{user_question}'. "
                     f"Table: sales (product_name, revenue, country). "
                     f"Return ONLY SQL."
        )
        
        # Clean the response to get pure SQL
        generated_sql = response.text.strip().replace('```sql', '').replace('```', '').split(';')[0]

        # Step 2: DB Query
        db_results = execute_safe_sql(generated_sql)
        
        # Step 3: Human-friendly Summary
        summary_resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"The user asked: '{user_question}'. The database returned: {db_results}. "
                     f"Explain this result to the user naturally."
        )
        
        return jsonify({
            "sql": generated_sql, 
            "results": db_results, 
            "summary": summary_resp.text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)