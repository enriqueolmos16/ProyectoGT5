from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Cargar el modelo T5
nombre_modelo = "t5-large"
convertir_vectores = AutoTokenizer.from_pretrained(nombre_modelo)
modelo = T5ForConditionalGeneration.from_pretrained(nombre_modelo)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    data = request.json
    eleccion = int(data['eleccion'])
    texto = data['texto']
    
    tipo_tarea = ""
    
    if eleccion == 1:
        tipo_tarea = "summarize: "
    elif eleccion == 2:
        tipo_tarea = "translate English to French: "
    elif eleccion == 3:
        contexto = data.get('contexto', '')
        tipo_tarea = "question: "
        texto = f"{tipo_tarea} {texto}. context: {contexto}"
    elif eleccion == 4:
        tipo_tarea = "generate questions: "
    
    texto = f"{tipo_tarea} {texto}"
    
    vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids
    num_preguntas = 5 if eleccion == 4 else 1
    vectores_salida = modelo.generate(vectores_entrada, max_length=512, num_return_sequences=num_preguntas)

    texto_salida = [convertir_vectores.decode(salida, skip_special_tokens=True) for salida in vectores_salida]
    
    return jsonify({'resultado': texto_salida})

if __name__ == '__main__':
    app.run(debug=True)


