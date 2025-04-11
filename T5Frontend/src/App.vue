<script setup>
import { ref } from "vue";
import axios from "axios";

const texto = ref("");
const eleccion = ref(1);
const resultado = ref("");

const procesarTexto = async () => {
  try {
    const response = await axios.post("http://127.0.0.1:5000/procesar", {
      texto: texto.value,
      eleccion: eleccion.value,
    });
    resultado.value = response.data.resultado;
  } catch (error) {
    console.error("Error al procesar el texto:", error);
    resultado.value = "Hubo un error. Intenta de nuevo.";
  }
};
</script>

<template>
  <div class="container">
    <h1>Procesador de Texto con T5</h1>
    <textarea v-model="texto" placeholder="Ingresa tu texto"></textarea>

    <select v-model="eleccion">
      <option value="1">Resumir</option>
      <option value="2">Traducir (Inglés a Francés)</option>
      <option value="3">Hacer una pregunta</option>
      <option value="4">Generar 5 preguntas</option>
    </select>

    <button @click="procesarTexto">Procesar</button>

    <div v-if="resultado">
      <h2>Resultado:</h2>
      <p>{{ resultado }}</p>
    </div>
  </div>
</template>

<style>
.container {
  max-width: 600px;
  margin: auto;
  padding: 20px;
  text-align: center;
}
textarea {
  width: 100%;
  height: 100px;
}
button {
  margin-top: 10px;
}
</style>
