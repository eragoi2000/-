import io
import base64
import requests
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from flask import Flask, request, jsonify

GOOGLE_API_KEY = "ВСТАВЬТЕ_ВАШ_КЛЮЧ_СЮДА"

def preprocess(pil_image):
    img = np.array(pil_image.convert('RGB'))
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    result = Image.fromarray(img)
    result = ImageEnhance.Contrast(result).enhance(1.4)
    return result


def image_to_base64(pil_image):
    """Конвертируем PIL изображение в base64 для Google API"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def recognize_text(pil_image):
    processed = preprocess(pil_image)
    img_b64 = image_to_base64(processed)

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"

    payload = {
        "requests": [{
            "image": {
                "content": img_b64
            },
            "features": [{
                "type": "DOCUMENT_TEXT_DETECTION",  # лучше для рукописного текста
                "maxResults": 1
            }],
            "imageContext": {
                "languageHints": ["kk", "ru", "en"]  # казахский, русский, английский
            }
        }]
    }

    response = requests.post(url, json=payload, timeout=30)
    result = response.json()

    # Проверка ошибок
    if 'error' in result:
        raise Exception(result['error']['message'])

    resp = result['responses'][0]

    if 'error' in resp:
        raise Exception(resp['error']['message'])

    if 'fullTextAnnotation' not in resp:
        return '', []

    full_annotation = resp['fullTextAnnotation']
    full_text = full_annotation.get('text', '').strip()

    # Разбиваем на строки с уверенностью
    lines = []
    for page in full_annotation.get('pages', []):
        for block in page.get('blocks', []):
            block_conf = round(block.get('confidence', 0) * 100, 1)
            for paragraph in block.get('paragraphs', []):
                para_text = ''
                para_conf_list = []
                for word in paragraph.get('words', []):
                    word_conf = word.get('confidence', 0)
                    para_conf_list.append(word_conf)
                    symbols = ''.join(
                        s.get('text', '') for s in word.get('symbols', [])
                    )
                    para_text += symbols + ' '
                para_text = para_text.strip()
                if para_text:
                    avg_conf = round(
                        sum(para_conf_list) / len(para_conf_list) * 100, 1
                    ) if para_conf_list else block_conf
                    lines.append({
                        'text': para_text,
                        'confidence': avg_conf
                    })

    return full_text, lines

HTML_PAGE = """<!DOCTYPE html>
<html lang="kk">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Қолжазбаны тану · Google Vision OCR</title>
  <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:      #f7f3ed;
      --surface: #ffffff;
      --ink:     #1e1810;
      --blue:    #1a56a0;
      --gold:    #c8922a;
      --green:   #2e7d44;
      --red:     #b03030;
      --muted:   #8a7a65;
      --line:    #e0d8cc;
      --shadow:  rgba(30,24,16,0.10);
    }
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      background: var(--bg);
      color: var(--ink);
      font-family: 'Nunito', sans-serif;
      min-height: 100vh;
    }

    /* ── Шапка ── */
    .header {
      background: linear-gradient(135deg, #1a3a6a 0%, #1a56a0 100%);
      color: white;
      padding: 18px 36px;
      display: flex;
      align-items: center;
      gap: 14px;
      box-shadow: 0 3px 20px rgba(26,86,160,0.3);
    }
    .flag { font-size: 2rem; }
    .header h1 { font-size: 1.25rem; font-weight: 800; }
    .header p  { font-size: 0.7rem; opacity: 0.65; font-family: 'JetBrains Mono',monospace; letter-spacing: 2px; margin-top: 2px; }
    .powered {
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.72rem;
      opacity: 0.75;
      font-family: 'JetBrains Mono', monospace;
    }
    .g-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: #fbbc04;
      box-shadow: 12px 0 0 #ea4335, 6px 10px 0 #34a853;
    }

    /* ── Сетка ── */
    .container {
      max-width: 960px;
      margin: 0 auto;
      padding: 32px 20px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    @media(max-width:640px){
      .container { grid-template-columns: 1fr; }
      .header { padding: 14px 18px; }
      .powered { display: none; }
    }

    /* ── Панель ── */
    .panel {
      background: var(--surface);
      border-radius: 14px;
      padding: 22px;
      box-shadow: 0 2px 16px var(--shadow);
      border: 1px solid var(--line);
    }
    .panel-title {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.65rem;
      color: var(--muted);
      letter-spacing: 3px;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .panel-title::after { content:''; flex:1; height:1px; background:var(--line); }

    /* ── Загрузка ── */
    .upload-zone {
      border: 2px dashed var(--line);
      border-radius: 10px;
      padding: 28px 16px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s;
      position: relative;
      background: var(--bg);
    }
    .upload-zone:hover, .upload-zone.drag {
      border-color: var(--blue);
      background: rgba(26,86,160,0.04);
    }
    .upload-zone input { position:absolute; inset:0; opacity:0; cursor:pointer; }
    .upload-icon { font-size: 2.4rem; margin-bottom: 8px; }
    .upload-title { font-weight: 800; font-size: 0.95rem; color: var(--blue); }
    .upload-sub   { font-size: 0.75rem; color: var(--muted); margin-top: 4px; }

    #preview-wrap { display:none; margin-top: 14px; }
    #preview-img  {
      width: 100%;
      max-height: 300px;
      object-fit: contain;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fafafa;
    }

    /* ── Кнопки ── */
    .btn-row { display:flex; gap:10px; margin-top:14px; }
    button {
      font-family: 'Nunito', sans-serif;
      font-weight: 700;
      font-size: 0.88rem;
      padding: 10px 20px;
      border-radius: 8px;
      cursor: pointer;
      border: none;
      transition: all 0.2s;
    }
    .btn-main {
      background: var(--blue);
      color: white;
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
    .btn-main:hover:not(:disabled) { background: #134080; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(26,86,160,0.3); }
    .btn-main:disabled { background: var(--line); color: var(--muted); cursor: not-allowed; transform: none; box-shadow: none; }
    .btn-reset { background: transparent; border: 1px solid var(--line); color: var(--muted); padding: 10px 14px; }
    .btn-reset:hover { border-color: var(--muted); }

    /* ── Статус ── */
    #status {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      color: var(--muted);
      margin-top: 10px;
      min-height: 18px;
      display: flex;
      align-items: center;
      gap: 7px;
      letter-spacing: 0.5px;
    }
    #status.ok   { color: var(--green); }
    #status.err  { color: var(--red); }
    #status.wait { color: var(--gold); }

    .spin {
      width:13px; height:13px;
      border: 2px solid var(--line);
      border-top-color: var(--gold);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      flex-shrink: 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Результат ── */
    .result-section { display:none; }
    .result-section.show { display:block; animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }

    #result-text {
      font-size: 1.05rem;
      line-height: 2;
      white-space: pre-wrap;
      color: var(--ink);
      min-height: 80px;
      padding: 8px 0 12px;
      border-bottom: 1px solid var(--line);
    }

    .result-actions { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }

    .btn-sm {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.68rem;
      font-weight: 500;
      letter-spacing: 1px;
      padding: 6px 14px;
      background: transparent;
      border: 1px solid var(--line);
      color: var(--muted);
      border-radius: 5px;
      cursor: pointer;
      transition: all 0.18s;
    }
    .btn-sm:hover { border-color: var(--blue); color: var(--blue); }

    /* ── Таблица строк ── */
    .lines-wrap { display:none; margin-top:16px; }
    .lines-wrap.show { display:block; }
    .line-row {
      display:flex;
      align-items:center;
      gap:10px;
      padding: 7px 0;
      border-bottom: 1px solid var(--line);
      font-size: 0.87rem;
    }
    .line-text { flex:1; line-height:1.5; }
    .conf-col  { min-width:56px; text-align:right; }
    .conf-num  { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:var(--muted); }
    .conf-bar  { height:3px; border-radius:2px; background:var(--line); margin-top:3px; overflow:hidden; }
    .conf-fill { height:100%; border-radius:2px; transition: width 0.5s ease; }
    .hi { background: var(--green); }
    .md { background: var(--gold); }
    .lo { background: var(--red);  }

    /* ── Пустое состояние ── */
    .empty {
      text-align: center;
      padding: 50px 0;
      color: var(--muted);
    }
    .empty-icon { font-size:2.8rem; opacity:0.25; margin-bottom:10px; }
    .empty-text { font-size:0.85rem; line-height:1.7; }

    /* ── Советы ── */
    .tips {
      grid-column: 1/-1;
      background: rgba(26,86,160,0.05);
      border: 1px solid rgba(26,86,160,0.13);
      border-radius: 10px;
      padding: 14px 20px;
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
    }
    .tip { display:flex; align-items:flex-start; gap:8px; font-size:0.8rem; color:var(--muted); flex:1; min-width:160px; }
    .tip b { color: var(--ink); }
  </style>
</head>
<body>

<div class="header">
  <div class="flag">🇰🇿</div>
  <div>
    <h1>Қолжазбаны тану · OCR</h1>
    <p>KAZAKH · RUSSIAN · ENGLISH HANDWRITING RECOGNITION</p>
  </div>
  <div class="powered">
    <div class="g-dot"></div>
    &nbsp;Google Vision API
  </div>
</div>

<div class="container">

  <!-- Левая: загрузка -->
  <div class="panel">
    <div class="panel-title">СУРЕТ · ИЗОБРАЖЕНИЕ</div>

    <div class="upload-zone" id="dropzone">
      <input type="file" id="file-input" accept="image/*">
      <div class="upload-icon">🖼️</div>
      <div class="upload-title">Фото таңдаңыз · Выберите фото</div>
      <div class="upload-sub">Перетащите или нажмите<br>JPG · PNG · WEBP · BMP</div>
    </div>

    <div id="preview-wrap">
      <img id="preview-img" src="" alt="preview">
    </div>

    <div class="btn-row">
      <button class="btn-main" id="btn-rec" disabled onclick="recognize()">
        🔍 Тануды бастау
      </button>
      <button class="btn-reset" onclick="reset()">✕</button>
    </div>

    <div id="status">// фото жүктеңіз · загрузите фото</div>
  </div>

  <!-- Правая: результат -->
  <div class="panel">
    <div class="panel-title">НӘТИЖЕ · РЕЗУЛЬТАТ</div>

    <div class="result-section" id="result-section">
      <div id="result-text"></div>
      <div class="result-actions">
        <button class="btn-sm" id="btn-copy" onclick="copyText()">📋 КӨШІРУ</button>
        <button class="btn-sm" id="btn-detail" onclick="toggleDetail()">📊 ЖОЛДАР</button>
      </div>
      <div class="lines-wrap" id="lines-wrap"></div>
    </div>

    <div class="empty" id="empty">
      <div class="empty-icon">📝</div>
      <div class="empty-text">
        Нәтиже осы жерде шығады<br>
        <span style="font-size:0.75rem;opacity:0.6">Результат появится здесь</span>
      </div>
    </div>
  </div>

  <!-- Советы -->
  <div class="tips">
    <div class="tip">💡 <div><b>Жарық / Освещение</b><br>Жақсы жарық, көлеңке жоқ</div></div>
    <div class="tip">📐 <div><b>Тік / Ровно</b><br>Мәтін көлденең болуы керек</div></div>
    <div class="tip">🔤 <div><b>Айқын / Чётко</b><br>Буквы чёткие, не размыты</div></div>
    <div class="tip">📷 <div><b>Сапа / Качество</b><br>Минимум 300×300 пикселей</div></div>
  </div>

</div>

<script>
  let file = null, detailOn = false;

  const dropzone    = document.getElementById('dropzone');
  const fileInput   = document.getElementById('file-input');
  const previewWrap = document.getElementById('preview-wrap');
  const previewImg  = document.getElementById('preview-img');
  const btnRec      = document.getElementById('btn-rec');
  const statusEl    = document.getElementById('status');
  const resultSec   = document.getElementById('result-section');
  const emptyEl     = document.getElementById('empty');
  const resultText  = document.getElementById('result-text');
  const linesWrap   = document.getElementById('lines-wrap');

  fileInput.addEventListener('change', e => { if(e.target.files[0]) loadFile(e.target.files[0]); });
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('drag');
    if(e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });

  function loadFile(f) {
    file = f;
    previewImg.src = URL.createObjectURL(f);
    previewWrap.style.display = 'block';
    btnRec.disabled = false;
    setStatus('// фото жүктелді · загружено', '');
  }

  function setStatus(text, cls) {
    statusEl.innerHTML = cls === 'wait'
      ? `<div class="spin"></div>${text}` : text;
    statusEl.className = cls || '';
  }

  async function recognize() {
    if (!file) return;
    btnRec.disabled = true;
    setStatus('ТАНЫЛУДА... · РАСПОЗНАВАНИЕ...', 'wait');
    resultSec.classList.remove('show');
    emptyEl.style.display = 'none';

    const base64 = await new Promise(res => {
      const r = new FileReader();
      r.onload = e => res(e.target.result);
      r.readAsDataURL(file);
    });

    try {
      const resp = await fetch('/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64 })
      });
      const data = await resp.json();

      if (data.error) {
        setStatus('✗ ' + data.error, 'err');
        emptyEl.style.display = 'block';
      } else {
        resultText.textContent = data.text || '(мәтін табылмады · текст не найден)';
        renderLines(data.lines_detail || []);
        resultSec.classList.add('show');
        setStatus(`✓ ${data.line_count} жол · ${data.char_count} таңба`, 'ok');
      }
    } catch(e) {
      setStatus('✗ Сервер қатесі · Ошибка сервера', 'err');
      emptyEl.style.display = 'block';
    }
    btnRec.disabled = false;
  }

  function renderLines(lines) {
    linesWrap.innerHTML = lines.map(l => {
      const cls = l.confidence >= 70 ? 'hi' : l.confidence >= 40 ? 'md' : 'lo';
      return `<div class="line-row">
        <div class="line-text">${l.text}</div>
        <div class="conf-col">
          <div class="conf-num">${l.confidence}%</div>
          <div class="conf-bar"><div class="conf-fill ${cls}" style="width:${l.confidence}%"></div></div>
        </div>
      </div>`;
    }).join('');
  }

  function toggleDetail() {
    detailOn = !detailOn;
    linesWrap.classList.toggle('show', detailOn);
    document.getElementById('btn-detail').textContent = detailOn ? '▲ ЖАСЫРУ' : '📊 ЖОЛДАР';
  }

  function copyText() {
    navigator.clipboard.writeText(resultText.textContent).then(() => {
      const btn = document.getElementById('btn-copy');
      btn.textContent = '✓ КӨШІРІЛДІ';
      setTimeout(() => btn.textContent = '📋 КӨШІРУ', 2000);
    });
  }

  function reset() {
    file = null; fileInput.value = '';
    previewWrap.style.display = 'none';
    previewImg.src = '';
    btnRec.disabled = true;
    resultSec.classList.remove('show');
    linesWrap.classList.remove('show');
    detailOn = false;
    emptyEl.style.display = 'block';
    setStatus('// фото жүктеңіз · загрузите фото', '');
  }
</script>
</body>
</html>"""

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json.get('image', '')
        if not data:
            return jsonify({'error': 'Изображение не получено'})

        header, encoded = data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Ограничение размера
        w, h = image.size
        if max(w, h) > 2400:
            scale = 2400 / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        full_text, lines = recognize_text(image)

        return jsonify({
            'text': full_text,
            'line_count': len(lines),
            'char_count': len(full_text.replace('\n', '')),
            'lines_detail': lines
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    if GOOGLE_API_KEY == "ВСТАВЬТЕ_ВАШ_КЛЮЧ_СЮДА":
        print("\n⚠️  ВНИМАНИЕ: Вставьте Google API ключ в переменную GOOGLE_API_KEY!")
        print("   Получить: https://console.cloud.google.com\n")
    else:
        print("\n✅ API ключ найден")

    print("=" * 55)
    print("  🇰🇿 КАЗАХСКИЙ OCR — СЕРВЕР ЗАПУЩЕН")
    print("  Открой: http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000)