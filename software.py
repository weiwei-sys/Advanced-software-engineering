import gradio as gr

from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")



# def translate(text):
#     return pipe(text)[0]["translation_text"]

def translate(text):
    return pipe(text, src_lang="en", tgt_lang="zh")[0]["translation_text"]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            english = gr.Textbox(label="English text")
            translate_btn = gr.Button(value="Translate")
        with gr.Column():
            chinese = gr.Textbox(label="Chinese Text")

    translate_btn.click(translate, inputs=english, outputs=chinese, api_name="translate-to-chinese")
    examples = gr.Examples(examples=["I went to the supermarket yesterday.", "Helen is a good swimmer."],
                           inputs=[english])

demo.launch(share=True)