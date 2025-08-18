from flask import Flask, request
from api.rag import app as rag_app
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    incoming_msg = data.get('Body')
    language = data.get('Language', '')
    input_message = {"question":incoming_msg,"language":language}
    final_state = rag_app.invoke(input_message)
    return final_state["answer"]