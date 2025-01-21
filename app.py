import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone
from flask import Flask, request, jsonify

def query_index(query):
    # 1. Cargar variables de entorno
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not all([openai_api_key, pinecone_api_key]):
        raise ValueError("Asegúrate de que las variables de entorno estén configuradas correctamente")
    
    # 2. Inicializar Pinecone
    pc = Pinecone(
        api_key=pinecone_api_key
    )
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

    # 3. Conectar al índice existente
    index_name = "promptier-index"
    index = pc.Index(index_name)
    
    # 4. Inicializar el modelo de embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # 5. Generar el embedding de la consulta usando OpenAI
    query_embedding = embeddings.embed_query(query)
    
    # 6. Realizar la búsqueda en el índice
    results = index.query(
        namespace="promptier-namespace",
        vector=query_embedding,
        top_k=3,
        include_values=False,
        include_metadata=True
    )

    # 7. Preparar el contexto para GPT
    context_texts = []
    sources = []
    
    for match in results.matches:
        context_texts.append(match.metadata['text'])
        if match.metadata['source'] not in sources:
            sources.append(match.metadata['source'])
    
    # 9. Crear el prompt para GPT
    prompt = f"""Basándote en la siguiente información, responde a la pregunta: "{query}"
    
    Información relevante:
    {' '.join(context_texts)}

    Por favor, proporciona una respuesta clara y concisa, utilizando solo la información proporcionada.
    """

    # 10. Generar la respuesta
    response = llm.invoke(prompt)
    
    # 11. Mostrar resultados
    print(response.content)
    
    return {
        "result": response.content,
        "source_documents": context_texts
    }
  
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")

    result = query_index(query)
    answer = result["result"]
    sources = result["source_documents"]

    return jsonify({
        "answer": answer,
        "sources": sources,
    })

@app.route("/")
def index():
    return "Servidor de chatbot de Promptier"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)