import gradio as gr
from llama2_inference import doctorOutput
from language_en_to_other import translate
import json

# Path to your JSON file
file_path = '/home/mediboina.v/Vikash/medicalBot/language.json'

# Read data from the JSON file
with open(file_path, 'r') as file:
    key_value = json.load(file)

chatbot=[]
input_texts=[]
previous_input=f'''<<SYS>>Imagine you're ahealthcare professional offering guidance. When proposing treatments, prescriptions, and remedies, organize the details in a concise format using subheadings. It's crucial to suggest only one medication per specific issue to prevent confusion for the patient. Gather thorough information by asking follow-up questions before making any recommendations, and avoid immediate prescriptions if uncertainties exist. Instead, recommend relevant tests that should be conducted. Post-testing, inquire about the results and then prescribe medications based on the findings. In cases of urgency, offer basic first aid advice while emphasizing the importance of seeking immediate medical attention. Additionally, maintain a consistent and helpful approach by considering the context of prior responses.<</SYS>>'''

def generate_response(history,input_text,drp):
    global previous_input
    print(previous_input,"generating.......")
    src_lan=drp
    en_text=translate(history[-1][0],src_lan,'eng_Latn')
    print(en_text)
    previous_inputs=doctorOutput(en_text,previous_input)
    chat_bot_text=previous_inputs.split('[/INST]')[-1]
    print(chat_bot_text,previous_inputs)
    translated_text=translate(chat_bot_text,'eng_Latn',src_lan)
    print(translated_text)
    previous_input=previous_inputs
    # previous_input=previous_inputs
    # chatbot.append(chat_bot_text)
    # input_texts.append(input_text)
    # chat_history.append((input_text, chat_bot_text))
#     chat_display = '<div style="display: flex; flex-direction: column-reverse;">'
#     for i,text in enumerate(input_texts):
#         chat_display += f'<div style="text-align: right;">{text}</div>'
#         chat_display += f'<div style="text-align: left;">{chatbot[i]}</div>'
# #     chat_display += '</div>'

   
    return translated_text



def add_text(history, text,drp):
    history = history + [(text, None)]
    chat_bot_text= generate_response(history, text,drp)
    print(text,history)
    tupl= history[-1]
    history[-1]=(tupl[0],chat_bot_text)
    return history
    # return history, gr.Textbox(value="", interactive=False)




   


with gr.Blocks() as demo:
    drp=gr.Dropdown(
            key_value.keys(), label="Language", info="select language"
        )
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )

    txt_msg = txt.submit(add_text, [chatbot, txt,drp], [chatbot], queue=False)
    print("uhuhuhuhu")
    # txt_msg.then(
    #     generate_response, [chatbot,txt,drp], [chatbot], api_name="bot_response"
    # )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    


demo.queue()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",share=True)