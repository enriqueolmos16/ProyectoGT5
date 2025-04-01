from transformers import AutoTokenizer, T5ForConditionalGeneration

nombre_modelo = "t5-large"

# Importar los módulos necesarios del modelo T5
convertir_vectores = AutoTokenizer.from_pretrained(nombre_modelo)
modelo = T5ForConditionalGeneration.from_pretrained(nombre_modelo)

# Mostrar un menú para que el usuario elija la opción que desea realizar
print("Qué tarea quieres realizar")
print("1. Resumir")
print("2. Traducir")
print("3. Hacer una pregunta")
print("4. Generar 5 preguntas a partir de un texto")

# Leer la elección del usuario
eleccion = int(input())

# Variable para evaluar qué orden se le dará al modelo
tipo_tarea = ""

# Solicitar el texto que se quiere procesar
texto = input("Ingresa el texto: ")

# Determinar la tarea según la opción elegida
if eleccion == 1:
    tipo_tarea = "summarize: "
    texto = f"{tipo_tarea} {texto}"
elif eleccion == 2:
    tipo_tarea = "translate English to French: "
    texto = f"{tipo_tarea} {texto}"
elif eleccion == 3:
    contexto = input("Ingresa el contexto: ")
    tipo_tarea = "question: "
    texto = f"{tipo_tarea} {texto}. context: {contexto}"
elif eleccion == 4:
    tipo_tarea = "generate questions: "
    texto = f"{tipo_tarea} {texto}"

# Calcular los vectores de entrada
vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids

# Generar salida (para generar 5 preguntas, ajustamos max_length y num_return_sequences)
num_preguntas = 5 if eleccion == 4 else 1
vectores_salida = modelo.generate(vectores_entrada, max_length=512, num_return_sequences=num_preguntas)

# Decodificar los vectores de salida para generar el resultado
texto_salida = [convertir_vectores.decode(salida, skip_special_tokens=True) for salida in vectores_salida]

# Mostrar el resultado
print("\n ========= RESULTADO =========")
if eleccion == 4:
    for i, pregunta in enumerate(texto_salida, 1):
        print(f"{i}. {pregunta}")
else:
    print(texto_salida[0])
