from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, T5ForConditionalGeneration, MarianMTModel, MarianTokenizer
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

app = Flask(__name__)

# Imagen
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Modelos
modelo_general = T5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer_general = AutoTokenizer.from_pretrained("t5-large")

modelo_qg = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")
tokenizer_qg = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")

traductor_modelo = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
traductor_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

def traducir_a_espanol(textos):
    inputs = traductor_tokenizer(textos, return_tensors="pt", padding=True, truncation=True)
    translated = traductor_modelo.generate(**inputs)
    return [traductor_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    data = request.json
    eleccion = int(data['eleccion'])
    texto = data['texto']

    if eleccion == 5:
        try:
            image = pipe(texto).images[0]
            path = "static/generated_image.png"
            image.save(path)
            return jsonify({'resultado': "/" + path})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Modelo por defecto
    tokenizer = tokenizer_general
    modelo = modelo_general
    prompt = ""

    if eleccion == 1:
        prompt = f"summarize: {texto}"
    elif eleccion == 2:
        prompt = f"translate English to French: {texto}"
    elif eleccion == 3:
        # Extrae la pregunta y el contexto desde el textarea
        pregunta = ""
        contexto = ""

        lineas = texto.strip().split("\n")
        for linea in lineas:
            if linea.lower().startswith("pregunta:"):
                pregunta = linea[len("pregunta:"):].strip()
            elif linea.lower().startswith("contexto:"):
                contexto = linea[len("contexto:"):].strip()

        if not pregunta or not contexto:
            return jsonify({'error': 'Por favor escribe en el formato:\nPregunta: ¿...? \nContexto: ...'}), 400

        prompt = f"question: {pregunta} context: {contexto}"
    elif eleccion == 4:
        # Detecta una frase del texto y la resalta
        frase = texto.strip().split(".")[0]
        contexto_hl = texto.replace(frase, f"<hl>{frase}<hl>")
        prompt = f"generate question: {contexto_hl}"
        tokenizer = tokenizer_qg
        modelo = modelo_qg

    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    num_salidas = 5 if eleccion == 4 else 1

    outputs = modelo.generate(inputs, max_length=128, num_return_sequences=num_salidas, num_beams=num_salidas)
    respuestas = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    if eleccion == 4:
        respuestas = traducir_a_espanol(respuestas)

    return jsonify({'resultado': respuestas})

if __name__ == '__main__':
    app.run(debug=True)
