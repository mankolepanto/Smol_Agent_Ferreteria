from typing import Literal
import gradio as gr
from pydantic import BaseModel, Field

import os
import json
import pprint
from smolagents import CodeAgent, DuckDuckGoSearchTool
from smolagents import LiteLLMModel, OpenAIServerModel, CodeAgent, tool

last_output = '' # Solución temporal para almacenar el out del último step
contexto = ''

from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
'''
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)
'''
model = LiteLLMModel(
    model_id="groq/deepseek-r1-distill-llama-70b",
    api_key=os.environ.get("GROQ_API_KEY"),
    # custom_role_conversions=custom_roles,
#    flatten_messages_as_text=True
)
model._flatten_messages_as_text = False


# Función que se ejecutará en cada paso del agente
def callback_step(step_info):
    global last_output
    """
    Esta función se ejecuta en cada paso del agente.
    
    Args:
        step_info: Información sobre el paso actual del agente
    """

    if 1 == step_info.step_number:
        last_output = step_info.action_output

    print(f"DEBUG - Paso #{step_info.step_number} del agente")


# Modelo Pydantic para la respuesta estructurada
class RespuestaFerreteria(BaseModel):
    tipo_respuesta: Literal["final_answer", "additional_question", "out_of_scope"] = Field(
        description="Tipo de respuesta: final_answer si es la respuesta definitiva, additional_question si se necesita más información, out_of_scope si está fuera del ámbito de ferretería"
    )
    respuesta: str = Field(
        description="Texto de la respuesta según el tipo"
    )
    razonamiento: str = Field(
        description="Explicación del proceso de resolución, actualizado con las preguntas y respuestas hasta el momento"
    )


from pydantic import ValidationError

@tool
def answer_agent(response: dict) -> RespuestaFerreteria:
    """
    Herramienta que finaliza el proceso del agente con una respuesta estructurada.

    Args:
        response: Diccionario que debe cumplir con la estructura de RespuestaFerreteria.

    Returns:
        Un objeto RespuestaFerreteria validado.
    """
    try:
        print('tool answer_agent, respuesta:\n')
        pprint.pprint(response)
        return RespuestaFerreteria(**response)
    except ValidationError as e:
        print('tool answer_agent, ValidationError:\n')
        pprint.pprint(e.json())
        return RespuestaFerreteria(
            tipo_respuesta="out_of_scope",
            respuesta="Error al procesar la respuesta. Intenta reformular tu pregunta.",
            razonamiento=f"Error de validación: {e.json()}"
        )


system_prompt = """\nEres un asistente especializado EXCLUSIVAMENTE en aconsejar sobre temas relacionados con el mundo de la ferretería. 
Tienes conocimientos sobre herramientas, procedimientos, materiales de construcción, bricolaje, reparaciones domésticas, 
pintura, cerrajería, electricidad básica, fontanería y productos relacionados con el hogar y la construcción.

INSTRUCCIONES DE FUNCIONAMIENTO:

1. SOLO responderás consultas relacionadas con ferretería.

2. Para cada nueva consulta, deberás:
   - Analizar si está dentro del ámbito de la ferretería
   - Determinar qué información necesitas para responder completamente
   - Planificar las preguntas necesarias si falta información

3. Después de CADA interacción con el usuario, debes llamar a la herramienta "answer_agent" con una respuesta estructurada.
   La llamada a "answer_agent" SIEMPRE finaliza el proceso actual - no puedes ejecutar más acciones después.

4. Tu respuesta debe estar estructurada con Pydantic (RespuestaFerreteria) con estos tres campos obligatorios:
   - "tipo_respuesta": Uno de estos valores específicos:
      * "final_answer" - Cuando tienes toda la información para dar una respuesta completa
      * "additional_question" - Cuando necesitas más información del usuario
      * "out_of_scope" - Cuando la consulta no está relacionada con ferretería
   
   - "respuesta": El texto que se mostrará al usuario:
      * Si es final_answer: Tu respuesta completa y detallada
      * Si es additional_question: La siguiente pregunta necesaria para completar la información
      * Si es out_of_scope: Una explicación educada de que solo puedes ayudar con temas de ferretería
   
   - "razonamiento": Tu análisis interno detallado que debe incluir:
      * Evaluación de la consulta
      * Información que ya tienes
      * Información que aún necesitas
      * Plan de preguntas (si aplica)
      * Historial de preguntas y respuestas hasta el momento

5. REGLAS IMPORTANTES:
   - NUNCA respondas a temas no relacionados con ferretería
   - Después de cada llamada a "answer_agent" se reiniciará el contexto - no asumas que el usuario recuerda conversaciones previas
   - Solo muestras al usuario el contenido del campo "respuesta"
   - El "razonamiento" es para tu uso interno y no lo ve el usuario
   - Mantén las respuestas claras, concisas y profesionales
   - Si la consulta es ambigua pero podría estar relacionada con ferretería, usa "additional_question" para aclarar
   - Siempre responde utilizando la estructura definida por el modelo Pydantic `RespuestaFerreteria`.  
   - Devuelve únicamente un JSON válido, sin incluir delimitadores de bloque de código como ```json o ```.  
   - No agregues ninguna explicación antes o después de la respuesta.  



6. EJEMPLOS DE TEMAS VÁLIDOS:
   - Herramientas eléctricas y manuales
   - Materiales de construcción
   - Sistemas de fijación
   - Pinturas y barnices
   - Cerrajería
   - Electricidad básica
   - Fontanería
   - Jardinería (herramientas y materiales)
   - Seguridad en el hogar
   - Accesorios y consumibles relacionados
"""


agent = CodeAgent(
    tools=[answer_agent], 
    model=model, 
    add_base_tools=False,
    step_callbacks=[callback_step],
    # prompt_templates=custom_templates
)

agent.prompt_templates['system_prompt'] += system_prompt
agent.max_steps = 1  # Limitamos a 1 paso para forzar respuesta en un solo paso


def procesar_consulta(history):
    global last_output, contexto

    if not history:
        return history  # Si no hay historial, no hacer nada

    user_message = history[-1]["content"]  # Último mensaje del usuario

    try:
        query = f'''Utiliza el siguiente contexto para responder la pregunta del usuario.

        ## Contexto:
        {contexto}

        ## Pregunta del usuario:
        {user_message}
        '''

        print(f'procesar_consulta, query:\n{query}')
        result = agent.run(query)
        result = last_output # Despreciar la respuesta del agente y usar el out del último step

        # Si el resultado ya es un diccionario, lo usamos directamente
        if isinstance(result, dict):
            json_obj = result
        elif isinstance(result, str):
            json_obj = {"respuesta": result}
        elif isinstance(result, RespuestaFerreteria):
            json_obj = json.loads(result.model_dump_json())
        else:
            json_obj = json.loads(result)  # Convertir a diccionario si es JSON en texto

        # Verificar que el JSON tiene la estructura esperada
        if "respuesta" not in json_obj:
            raise KeyError("La respuesta del agente no tiene la clave 'respuesta'")

        contexto = json_obj['razonamiento']
        history.append({"role": "assistant", "content": json_obj['respuesta']})  # Agregar respuesta

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"\n\nDEBUG - Error procesando la respuesta: {e}")
        history.append({"role": "assistant", "content": "Hubo un error procesando la respuesta."})

    return history



# Interfaz Gradio
with gr.Blocks() as app:
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox(label="Introduzca una consulta", placeholder="Pregunta algo sobre ferretería...")

    def show_msg_user(user_message, history):
        print(f'user_message: {user_message}')
        history.append({"role": "user", "content": user_message})
        return "", history
   
    msg.submit(show_msg_user, [msg, chatbot], [msg, chatbot], queue=False).then(procesar_consulta, chatbot, chatbot)


# Ejecución de la aplicación
if __name__ == "__main__":
    app.launch()