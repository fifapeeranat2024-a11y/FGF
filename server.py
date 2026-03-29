from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)
# ตั้งค่าสมอง 1.4T / Context 30K
llm = Llama(model_path="./brain_model.gguf", n_ctx=30000)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    output = llm(data['prompt'], max_tokens=512)
    return jsonify({"answer": output['choices'][0]['text']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
  
