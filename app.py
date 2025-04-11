from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

app = Flask(__name__)

# ========== CARGA DE MODELOS ==========
modelo_general_nombre = "t5-large"
tokenizer_general = AutoTokenizer.from_pretrained(modelo_general_nombre)
modelo_general = T5ForConditionalGeneration.from_pretrained(modelo_general_nombre)

modelo_preguntas_nombre = "mrm8488/t5-base-finetuned-question-generation-ap"
tokenizer_preguntas = AutoTokenizer.from_pretrained(modelo_preguntas_nombre, use_fast=False)
modelo_preguntas = T5ForConditionalGeneration.from_pretrained(modelo_preguntas_nombre)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_general.to(device)
modelo_preguntas.to(device)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe = pipe.to(device)

# ========== FUNCIONES DE PROCESAMIENTO ==========
def generar_texto(modelo, tokenizer, prompt, max_length=128):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = modelo.generate(inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def traducir_texto(texto, origen="Spanish", destino="English"):
    prompt = f"translate {origen} to {destino}: {texto}"
    return generar_texto(modelo_general, tokenizer_general, prompt, max_length=128)

def generar_preguntas(texto_es):
    # 1. Traducir al inglés
    texto_en = traducir_texto(texto_es, "Spanish", "English")

    # 2. Generar preguntas en inglés
    prompt_qg = f"generate 5 questions based on the following text: {texto_en}"
    inputs = tokenizer_preguntas(prompt_qg, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = modelo_preguntas.generate(
            inputs,
            max_length=64,
            num_return_sequences=5,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    preguntas_en = [tokenizer_preguntas.decode(out, skip_special_tokens=True) for out in outputs]

    # 3. Traducir preguntas al español
    preguntas_es = [traducir_texto(pregunta, "English", "Spanish") for pregunta in preguntas_en]
    return preguntas_es

def generar_imagen(prompt):
    image = pipe(prompt).images[0]
    path = "static/generated_image.png"
    image.save(path)
    return "/" + path

# ========== RUTAS ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    data = request.json
    eleccion = int(data.get('eleccion', 0))
    texto = data.get('texto', '')

    try:
        if eleccion == 5:
            return jsonify({'resultado': generar_imagen(texto)})

        elif eleccion == 1:
            prompt = f"summarize: {texto}"
            return jsonify({'resultado': [generar_texto(modelo_general, tokenizer_general, prompt)]})

        elif eleccion == 2:
            return jsonify({'resultado': [traducir_texto(texto, "English", "French")]})

        elif eleccion == 3:
            contexto = data.get('contexto', '')
            prompt = f"question: {texto} context: {contexto}"
            return jsonify({'resultado': [generar_texto(modelo_general, tokenizer_general, prompt)]})

        elif eleccion == 4:
            preguntas = generar_preguntas(texto)
            return jsonify({'resultado': preguntas})

        else:
            return jsonify({'error': 'Elección no válida'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== EJECUCIÓN ==========
if __name__ == '__main__':
    app.run(debug=True)
