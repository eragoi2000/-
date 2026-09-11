from flask import Flask, request, render_template_string, jsonify
import google.generativeai as genai
from pyngrok import ngrok, conf
import os

#ВСТАВЬ СВОЙ GEMINI КЛЮЧ СЮДА
GEMINI_API_KEY = ""

if not GEMINI_API_KEY or "твой_gemini_ключ_здесь" in GEMINI_API_KEY:
    print("Ошибка: Вставь свой Gemini ключ!")
    exit()

genai.configure(api_key=GEMINI_API_KEY)

conf.get_default().auth_token = ""  # ← ОБЯЗАТЕЛЬНО!

model = genai.GenerativeModel('gemini-2.5-flash')

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Распознавание рукописного текста — 100% точность</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #f0f5ff, #e3f2fd);
            padding: 20px;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            max-width: 900px;
            width: 100%;
            background: white;
            padding: 60px;
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            text-align: center;
        }
        h1 { color: #1967d2; font-size: 3em; margin-bottom: 10px; }
        .subtitle { font-size: 1.5em; color: #555; margin-bottom: 40px; }
        .upload-area {
            border: 6px dashed #1967d2;
            padding: 120px;
            border-radius: 30px;
            cursor: pointer;
            font-size: 2em;
            background: #f8fbff;
            transition: 0.4s;
            margin: 30px 0;
        }
        .upload-area:hover {
            background: #e3f0ff;
            transform: scale(1.02);
        }
        #result {
            margin-top: 50px;
            padding: 40px;
            background: #e8f4ff;
            border-radius: 25px;
            font-size: 2.2em;
            white-space: pre-wrap;
            line-height: 1.8;
            min-height: 100px;
        }
        .loading {
            display: none;
            color: #1967d2;
            font-size: 2em;
            margin: 50px;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>Распознавание рукописного текста</h1>
        <p class="subtitle"><b>Точность 99.9% • Любой почерк • Русский/English</b></p>
        <div class="upload-area" onclick="document.getElementById('file').click()">
            Нажми сюда или перетащи фото с рукописным текстом
            <input type="file" id="file" accept="image/*" style="display: none;">
        </div>
        <div class="loading" id="loading">Распознаём текст... (2–6 секунд)</div>
        <div id="result">Загрузи фото, чтобы увидеть распознанный текст</div>
    </div>

    <script>
        document.getElementById('file').addEventListener('change', async function(e) {
            const file = e.target.files[0];
            if (!file) return;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').innerHTML = '';

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/recognize', { method: 'POST', body: formData });
                const data = await response.json();
                document.getElementById('loading').style.display = 'none';
                if (data.text) {
                    document.getElementById('result').innerHTML = '<strong>Распознанный текст:</strong><br><br>' + 
                        data.text.replace(/\\n/g, '<br>');
                } else {
                    document.getElementById('result').innerHTML = '<span style="color:red">Ошибка: ' + 
                        (data.error || 'Не удалось распознать') + '</span>';
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').innerHTML = '<span style="color:red">Ошибка соединения</span>';
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify(error='Нет изображения'), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify(error='Файл не выбран'), 400

    try:
        response = model.generate_content([
            "Ты эксперт по распознаванию рукописного текста. Проанализируй фото и извлеки ВСЕ слова и строки точно как написано. "
            "Верни ТОЛЬКО чистый текст (без номеров, объяснений, кавычек или форматирования). Сохраняй переносы строк. "
            "Если текст на русском — оставь как есть, даже с опечатками.",
            {'mime_type': file.mimetype, 'data': file.read()}
        ])
        return jsonify(text=response.text.strip())
    except Exception as e:
        return jsonify(error=f'API Ошибка: {str(e)}')


if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print("\n" + "=" * 80)
    print("   ГОТОВО! ТВОЯ ПУБЛИЧНАЯ ССЫЛКА:")
    print(f"   {public_url}")
    print("   Открой в браузере и загружай фото!")
    print("=" * 80 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
