# 🚀 Moshrif Embedding Service

خدمة Embedding متكاملة للمحتوى العربي باستخدام نموذج BGE-M3، مع نظام استرجاع هرمي ذكي (Hierarchical Retrieval) لقاعدة بيانات Qdrant.

---

## 📋 المميزات

- ✅ **Embedding API** - واجهة FastAPI لتوليد embeddings للنصوص العربية
- ✅ **Arabic Text Normalization** - تطبيع النصوص العربية (إزالة التشكيل، توحيد الألف، إلخ)
- ✅ **Hierarchical Retrieval** - نظام استرجاع ثلاثي الطبقات (Filename → Title → Content)
- ✅ **Qdrant Integration** - تخزين واسترجاع فعّال باستخدام Qdrant Vector Database
- ✅ **BGE-M3 Model** - نموذج متعدد اللغات عالي الجودة (1024-dim vectors)

---

## 🏗️ بنية المشروع

```
Embedding-Service/
├── main.py                    # FastAPI Embedding API
├── model_loader.py            # تحميل نموذج BGE-M3
├── config.py                  # إعدادات النموذج
├── requirements.txt           # المتطلبات
├── model/
│   └── bge-m3/               # ملفات النموذج
├── qdrant_db/                # قاعدة بيانات Qdrant الأساسية
└── hierarchical_retrieval/   # نظام الاسترجاع الهرمي
    ├── search_hierarchical.py        # البحث الهرمي
    ├── build_hierarchical_index.py   # بناء الفهرس
    ├── normalize_arabic.py           # تطبيع النصوص
    ├── Moshrif-knowledge-chunks.json # البيانات
    └── qdrant_db_hierarchical/       # قاعدة البيانات الهرمية
```

---

## 🚀 التثبيت

### 1. استنساخ المستودع
```bash
git clone https://github.com/Loay-Wael1/embedding-service-moshrif.git
cd embedding-service-moshrif
```

### 2. إنشاء بيئة افتراضية
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 4. تحميل النموذج
قم بتحميل نموذج [BGE-M3](https://huggingface.co/BAAI/bge-m3) ووضعه في مجلد `model/bge-m3/`

---

## 💻 الاستخدام

### تشغيل خدمة الـ Embedding API

```bash
uvicorn main:app --reload
```

الخدمة ستعمل على: `http://127.0.0.1:8000`

### الـ Endpoints

| Endpoint | Method | الوصف |
|----------|--------|-------|
| `/health` | GET | فحص حالة الخدمة |
| `/embed` | POST | توليد embedding لنص |

### مثال استخدام الـ API

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/embed",
    json={"text": "كيف أربي طفلي؟"}
)

embedding = response.json()["embedding"]
print(f"Vector size: {len(embedding)}")  # 1024
```

---

## 🔍 نظام الاسترجاع الهرمي (Hierarchical Retrieval)

### الفكرة
نظام ذكي يبحث على 3 مستويات بأولويات مختلفة:

| الطبقة | Threshold | السلوك |
|--------|-----------|--------|
| **Filename** | 0.60 | إذا طابق اسم الملف، يرجع كل chunks الفيديو بترتيبها الطبيعي |
| **Title** | 0.65 | إذا طابق العنوان، يرجع أفضل 5 chunks من كل الفيديوهات |
| **Content** | 0.55 | إذا طابق المحتوى، يرجع الـ chunk المطابق فقط |

### بناء الفهرس

```bash
cd hierarchical_retrieval
python build_hierarchical_index.py
```

### البحث

```python
from hierarchical_retrieval.search_hierarchical import search_query

result = search_query("هل الذكاء الاصطناعي هيستبدل المبرمجين؟", top_k=5)

print(f"Mode: {result['retrieval_mode']}")
print(f"Scores: {result['scores']}")
for chunk in result['results']:
    print(f"- {chunk['chunk_title']}")
```

### مثال الـ Output

```json
{
  "query": "هل الذكاء الاصطناعي هيستبدل المبرمجين؟",
  "retrieval_mode": "by_filename",
  "scores": {
    "title": 0.5234,
    "filename": 0.7821,
    "content": 0.6123
  },
  "results": [
    {
      "video_id": 15,
      "filename": "الذكاء الاصطناعي والبرمجة",
      "chunk_id": 1,
      "chunk_title": "مقدمة",
      "chunk_content": "..."
    }
  ]
}
```

---

## ⚙️ الإعدادات

### config.py
```python
MODEL_NAME = "./model/bge-m3"  # مسار النموذج
DEVICE = "cpu"                  # أو "cuda" للـ GPU
```

### Thresholds (في search_hierarchical.py)
```python
TITLE_THRESHOLD = 0.65
FILENAME_THRESHOLD = 0.60
CONTENT_THRESHOLD = 0.55
```

---

## 📊 تطبيع النصوص العربية

الخدمة تقوم تلقائياً بتطبيع النصوص العربية قبل توليد الـ embedding:

1. **إزالة التشكيل** - حذف الحركات (فتحة، ضمة، كسرة، إلخ)
2. **توحيد الألف** - تحويل (إ، أ، آ) إلى (ا)
3. **توحيد الياء** - تحويل (ى) إلى (ي)
4. **تنظيف المسافات** - إزالة الشرطات والمسافات الزائدة

---

## 🛠️ المتطلبات التقنية

- Python 3.10+
- PyTorch 2.0+
- ~2GB RAM للنموذج
- GPU اختياري (يعمل على CPU)

---

## 📝 License

MIT License

---

## 👨‍💻 المطور

[Loay Wael](https://github.com/Loay-Wael1)
