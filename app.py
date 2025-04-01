from flask import Flask, request, jsonify
from transformers import AutoTokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Cargar el modelo y el tokenizador en versión TensorFlow
nombre_modelo = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
model = T5ForConditionalGeneration.from_pretrained(nombre_modelo)

@app.route('/procesar', methods=['POST'])
def procesar_texto():
    datos = request.json
    eleccion = datos.get("eleccion")  # 1: Resumir, 2: Traducir, 3: Preguntar, 4: Generar Preguntas
    texto = datos.get("texto", "")

    if not texto:
        return jsonify({"error": "Texto no proporcionado"}), 400

    if eleccion == 1:
        texto = f"summarize: {texto}"
    elif eleccion == 2:
        texto = f"translate English to French: {texto}"
    elif eleccion == 3:
        contexto = datos.get("contexto", "")
        texto = f"question: {texto}. context: {contexto}"
    elif eleccion == 4:
        texto = f"generate questions: {texto}"
    else:
        return jsonify({"error": "Opción no válida"}), 400

    # Tokenizamos el texto y generamos la respuesta usando tensores de TensorFlow
    input_ids = tokenizer(texto, return_tensors="tf").input_ids
    output_ids = model.generate(input_ids, max_length=512)
    resultado = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({"resultado": resultado})

if __name__ == '__main__':
    app.run(debug=True)

