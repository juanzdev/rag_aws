from flask import Flask, request, jsonify, render_template
from chunks import Chunks
from model_gen import ModelGenerator
from vector import Vector
import os
app = Flask(__name__)

model_gen = ModelGenerator()
vector_store = Vector(
	api_key=os.getenv('PINECONE_API_KEY'),
	environment="gcp-starter",
	index_name="rag-aws-poc-index"
)

print("VS KEY")
print(os.getenv('PINECONE_API_KEY'))
chunk_manager = Chunks()

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	data = request.get_json(force=True)
	question = data.get('question', '')
	if not question:
		return jsonify({"error": "No question provided"})
	out = model_gen.predict_with_enriched_context(question)
	return out

@app.route('/regenerate_index', methods=['POST'])
def regenerate_index():
	chunk_manager.consolidate_all_chunks_to_csv_and_upload_to_s3()
	chunks_csv = chunk_manager.download_all_csv_chunks_from_s3()
	vector_store.chunks_to_vector_store(chunks_csv)
	return "Index updated"

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80)

# debug utility commands
#ps aux | grep flask
#ps aux | grep app.py
#kill pid
#sudo tail -f /var/log/cloud-init-output.log
#source /myenv/bin/activate
#cd /var/www
#nohup python3.9 app.py &