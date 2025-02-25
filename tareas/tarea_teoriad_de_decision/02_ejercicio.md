# Guía para diseñar, mapear y resolver tu propio problema de decisión estática bajo incertidumbre (con ejemplo y código en Python)


## Instrucciones
Este documento describe un proceso **paso a paso** para que los estudiantes **formulen** y **resuelvan** un **problema de decisión estática** bajo incertidumbre, incorporando información parcial (proxy), heurísticas y un pequeño ejemplo programado en Python. 

La idea es que el lector aplique los conceptos en un **problema original** (inventado por sí mismo), definiendo:

1. **Qué se decide**
2. **Cuál es la incertidumbre** y el espacio de estados
3. **Qué información parcial** (proxy) se dispone
4. **Cómo se calcula** la utilidad o el costo
5. **Qué criterios** o heurísticas se usan para la decisión con pocos datos

Para ilustrar, desarrollaremos un **ejemplo lúdico**: tomar la decisión de **enviar (o no) un mensaje** a la persona que nos gusta, bajo incertidumbre acerca de sus intenciones y sentimientos, basándonos en ciertas **"señales"** (proxy) que podemos observar en redes sociales. Se añadirán elementos anecdóticos (gaslighting, etc.) para que sirvan de inspiración. 

Al final, presentaremos **código en Python** (en forma de jupyter_notebook, scripts, txt y markdown) que demuestra cómo **definir el modelo**, **implementarlo** y **visualizar** el proceso de decisión.

### Entregable
**Entregable**: Piensa en un problema de tu dia a dia, y utilizando lo que sabes formulalo como un problema de teoria de desicion estatica (con incertidumbre), y agrega codigo para entender mejor el problema y simularlo. Pueden usar cuantos scripts.py, jupyters y .md o .txt requierran (dentro de lo sensato) siempre y cuando este disponible online (ya sea github, drive, onedrive, colab, etc..). Tota tu informacion tiene que ser accesible y online para que la pueda calificar. Al ser tu proyecto tu decides la estructura, visualizaciones, heuristicas, el problema, etc.
+ Definicion matematica y clara del problema (puedes usar latex, mardown o subir una foto clara y pegarla en un archivo markdown o directo en un jupyter notebook). Desde la idea general, cuales son los estados, probabilidades, funciones de perdida, etc. La idea es que des una explicacion muy detallada y formal del problema que tu quieras.
+ Codigo donde se define el problema, se visualiza su definicion.
+ Por lo menos dos heuristicas/solucione para el problema. Tienes que describir por que las tomaste (o como se inspiraron). Compara las heuristicas usando varios ejemplos e interpreta.
+ **Importante** La descripcion del problema, como se mapea a un problema de decision, las heuristicas, metricas y comparaciones tienen que ser muy detalldas pues es lo mas importante.


---

## 1. Planeación y mapeo del problema

### 1.1 Preguntas para iniciar

Antes de plantear un problema de decisión bajo incertidumbre, conviene hacerse preguntas clave (puedes usar esto como lista de _prompts_ para ti mismo):

1. **Objetivo**: Ir a jugar tenis al ITAM
2. **Contexto**: ITAM, 
   - Factores externos:

      - Clima: Lluvia (imposibilita jugar), sol (ideal pero calor intenso podría ser un problema).

      - Hora del día: Mañana (menos tráfico, posible rocío), tarde (tráfico pesado, calor), noche (iluminación, seguridad).

      - Tráfico: Afecta tiempo de traslado (ej: tarde = hora pico).

      - Estado personal: Cansancio físico (riesgo de lesión o bajo rendimiento).
3. **Incertidumbre**: 
   - Clima: ¿Lloverá durante el horario planeado?

   - Disponibilidad: ¿Estará la cancha ocupada?

   - Tráfico: ¿Cambiará según la hora?

   - Energía futura: ¿El cansancio actual empeorará al llegar?
4. **Métricas de éxito**:
   - Beneficios:

      - Ejercicio satisfactorio (1-2 horas de juego).


   - Costos:

      - Tiempo perdido si no se juega (ej: viaje innecesario).

      - Fatiga física excesiva.

      - Riesgo de mojarse o accidentes por clima.
5. **Información parcial (proxy)**
   - Clima: Pronóstico del tiempo (ej: app con 80% de precisión).

   - Tráfico: Apps como Google Maps para estimar tiempo de traslado.

   - Cansancio: Autoevaluación física (ej: escala del 1 al 10; si ≥7, reconsiderar).

   - Canchas: Historial de ocupación (ej: las tardes suelen estar llenas).
6. **Recursos o restricciones**: 
   - Tiempo: Horario disponible (ej: 2 horas libres).

   - Físicas: Energía suficiente para jugar sin lesionarte.

   - Logísticas: Acceso a transporte, raqueta, equipo adecuado.
7. **Posibles heurísticas**: 
   - Regla climática:

      - "Si hay >30% probabilidad de lluvia o está lloviendo, no voy".

   - Regla horaria:

      - "Juego en la mañana si el pronóstico es bueno; evito tardes con tráfico alto".

   - Regla de energía:

      - "Si mi cansancio está ≥7/10, pospongo para otro día".

   - Regla de eficiencia:

      - "Si el viaje toma >30% del tiempo total planeado, priorizo otra actividad".

---

### 1.2 Definición conceptual de un problema genérico

Siguiendo el documento teórico previo, un **problema de decisión estática bajo incertidumbre** incluye:

1. **Decisión**: 

   - \( d_1 \): Ir a jugar tenis al ITAM.  
   - \( d_2 \): No ir y quedarse en casa.  
 
2. **Estados de la naturaleza** $\Omega$: Combinaciones de factores externos y personales:  
   - **Clima**: Lluvia (\( \omega_1 \)).  
   - **Tráfico**: Alto (\( \omega_2 \)).  
   - **Energía personal**: x/10 (\( \omega_3 \)).  
   - **Disponibilidad de cancha**: Ocupada (\( \omega_4 \)). 
3. **Información proxy** $Z$: Observaciones ruidosas correlacionadas con $\Omega$
   Señales observables (imperfectas) relacionadas con los estados:  
   - \( Z_1 \): Pronóstico del tiempo (ej: "10% probabilidad de lluvia").  
   - \( Z_2 \): Tiempo estimado de traslado en Google Maps.  
   - \( Z_3 \): Autoevaluación de energía (ej: escala 1-10).  
   - \( Z_4 \): Historial de ocupación de canchas.  
4. **Función de utilidad** $U(\omega, d)$ o costo \[
U(\omega, d) = 
\begin{cases} 
\text{Beneficio físico} - \text{riesgo}, & \text{si } d = d_1 \text{ y se puede jugar}, \\
0, & \text{si } d = d_2, \\
\end{cases}
\]
5. **Modelo probabilístico**: $p(\omega)$ o $p(\omega \mid Z)$

   Usando el criterio de **máxima utilidad esperada**:  
   \[
   d^* = \arg\max_{d \in D} \; \mathbb{E}[U(\omega, d)] = \arg\max_{d \in D} \sum_{\omega \in \Omega} U(\omega, d) \cdot p(\omega \mid Z).
   \]
   - Supongamos que \( Z \) indica:  
      - Probabilidad de lluvia (\( \omega_1 \)) = 10%,  
      - Probabilidad de tráfico alto (\( \omega_2 \)) = 30%,  
      - Energía = 6/10 (\( \omega_3 \)),  
      - Cancha ocupada (\( \omega_4 \)) = 40%.
   - **Cálculo para \( d_1 \)** (simplificando estados):  
   \[
   \mathbb{E}[U(d_1)] = (0.9 \cdot 0.6 \cdot 8) + (0.1 \cdot 0.6 \cdot -3) = 4.32 - 0.18 = 4.14.
   \]  
   - **Cálculo para \( d_2 \)** (no ir):  
   \[
   \mathbb{E}[U(d_2)] = 0.
   \]  
   - **Conclusión**: \( d^* = d_1 \) (ir a jugar).

