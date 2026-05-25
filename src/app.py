import gradio as gr
import pickle
import pandas as pd

# Загрузка модели
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_rating(num_pages, ratings_count, text_reviews_count):
    input_data = pd.DataFrame([[num_pages, ratings_count, text_reviews_count]], 
                              columns=['num_pages', 'ratings_count', 'text_reviews_count'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return f"{round(prediction[0], 2)}"

# CSS для компактности: убираем лишние margin и padding
css = """
.container { max-width: 450px !important; margin: auto !important; }
/* Прячем футер Gradio и ссылку на API */
footer { display: none !important; }
.gradio-container .built-with { display: none !important; }
"""

with gr.Blocks() as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("### 📚 Goodreads Rating Predictor")
        
        num_pages = gr.Number(label="Страниц", value=300)
        ratings_count = gr.Number(label="Оценок", value=1000)
        text_reviews_count = gr.Number(label="Отзывов", value=50)
        
        btn = gr.Button("Рассчитать рейтинг", variant="primary")
        output = gr.Textbox(label="Прогноз", interactive=False)
        
        # Примеры теперь тоже компактные
        gr.Examples(
            examples=[[300, 1000, 50], [500, 5000, 200], [150, 200, 10]],
            inputs=[num_pages, ratings_count, text_reviews_count],
            label="Примеры для теста:"
        )

    btn.click(fn=predict_rating, inputs=[num_pages, ratings_count, text_reviews_count], outputs=output)

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Monochrome(primary_hue="indigo"), css=css)