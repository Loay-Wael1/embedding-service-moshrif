# Egyptian Laws Embedding + Retrieval Service

هذا المشروع أصبح نظام استرجاع قانوني للمحتوى التشريعي المصري، بعد إزالة تصميمه القديم المبني على فيديوهات وترانسكريبتات. النظام الحالي مبني حول:

- `FastAPI` لخدمة embeddings
- `BGE-M3` للمتحولات dense
- `Qdrant` كقاعدة vector
- نموذج بيانات قانوني: `law` و`article` و`article_chunk`

## ما الذي تغير عن النسخة القديمة

تمت إزالة أو استبدال الآتي:

- منطق `filename/title/content` الخاص بالفيديو
- payload القديم المبني على `video_id`, `filename`, `chunk_title`, `chunk_content`
- collection names القديمة الخاصة بـ Moshrif
- استرجاع الفيديو كاملًا بترتيبه الطبيعي
- عتبات `threshold logic` الخاصة بمطابقة اسم الملف والعنوان والمحتوى
- سكربتات `hierarchical_retrieval` القديمة

تم استبدال ذلك بـ:

- تطبيع عربي قانوني محافظ وقابل للاختبار
- indexer قانوني يقرأ `jsonl` أو `json`
- collection جديدة: `egyptian_laws_v2_legal`
- استرجاع يعتمد على `article_chunk` للاستدعاء الأولي ثم `article` للسياق النهائي
- down-ranking / filtering للحالات الملغاة أو النصوص المريبة
- rerank hook نظيف وقابل للاستبدال

## بنية المشروع

```text
Embedding-Service/
├── app/
│   ├── api/                  # FastAPI app + schemas
│   ├── embeddings/           # embedding service and model loading
│   ├── indexing/             # dataset parsing and Qdrant indexing
│   ├── models/               # legal dataclasses
│   ├── preprocessing/        # legal Arabic normalization and quality checks
│   ├── retrieval/            # legal retrieval and reranking
│   └── settings.py           # environment-driven settings
├── scripts/
│   ├── build_legal_index.py
│   ├── demo_legal_queries.py
│   └── build_and_query_demo.py
├── tests/
├── env.example
├── main.py                   # uvicorn entrypoint
├── model_loader.py           # backward-compatible embedding shim
└── requirements.txt
```

## نموذج البيانات القانوني

الملف الأساسي الجديد هو:

- `egypt_laws_civil_labor_penal_dataset_v2_expanded.jsonl`

السجلات الأصلية في الملف تحتوي أساسًا على:

- `article`
- `article_chunk`

وأثناء الفهرسة يتم توليد سجلات synthetic من النوع:

- `law`

### الحقول الأساسية في Qdrant payload

- `id`
- `record_kind`
- `parent_id`
- `legal_domain`
- `law_name`
- `law_number`
- `law_year`
- `article_number`
- `title`
- `content`
- `summary`
- `source_url`
- `status`
- `status_normalized`
- `quality_flags`
- `quality_warnings`
- `quality_score`
- `noise_score`
- `section_level`
- `document_level`
- `retrieval_text`
- `embedding_text`
- `is_repealed_candidate`

## ملاحظات الجودة على الداتا

الملف الحالي مناسب جدًا لنسخة MVP / demo، لكن توجد احتياطات مضافة في الكود:

- بعض السجلات المدنية والعمالية ما زالت تحمل آثار OCR أو تكسيرات مسافات وترقيم.
- سجلات supplement الخاصة بقانون العقوبات أنظف عمومًا.
- `status_normalized` في العينة الحالية قد لا يعكس دائمًا كل إشارات الإلغاء النصية، لذلك يوجد فحص إضافي داخل النص نفسه مثل `ملغاة` و`منسوخ`.
- هناك `noise_score` و`quality_warnings` محسوبان أثناء الفهرسة ويؤثران على الاسترجاع.

## تطبيع العربية القانونية

التطبيع الجديد يحاول تحسين البحث بدون تدمير الصياغة القانونية:

- إزالة التشكيل
- توحيد الألف: `إ/أ/آ/ٱ -> ا`
- تحويل `ى -> ي`
- تحويل الأرقام العربية والهندية إلى ASCII للحفاظ على مرجعية المواد
- تنظيف المسافات وعلامات الترقيم بشكل محافظ
- إزالة `tatweel`
- اكتشاف مؤشرات OCR بدل تنفيذ إصلاحات عدوانية قد تغيّر النص القانوني

## تشغيل خدمة الـ Embedding

### التثبيت

```bash
pip install -r requirements.txt
```

### تشغيل الخدمة

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Endpoints

#### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

#### `GET /info`

```bash
curl http://127.0.0.1:8000/info
```

#### `POST /embed`

```bash
curl -X POST http://127.0.0.1:8000/embed ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"ما هي أحكام عقد العمل؟\",\"mode\":\"query\",\"normalize\":true,\"return_dense\":true,\"return_sparse\":false}"
```

#### `POST /embed/batch`

```bash
curl -X POST http://127.0.0.1:8000/embed/batch ^
  -H "Content-Type: application/json" ^
  -d "{\"texts\":[\"عقد العمل\",\"انعقاد العقد\"],\"mode\":\"document\",\"normalize\":true,\"return_dense\":true,\"return_sparse\":false}"
```

### شكل الاستجابة

```json
{
  "model": "./model/bge-m3",
  "dim": 1024,
  "mode": "document",
  "normalized": true,
  "sparse_available": false,
  "warnings": [],
  "results": [
    {
      "text": "عقد العمل",
      "normalized_text": "عقد العمل",
      "dense": [0.1, 0.2],
      "sparse": null,
      "metadata": {
        "mode": "document"
      }
    }
  ]
}
```

## بناء الفهرس القانوني

### الإعدادات

انسخ `env.example` أو اضبط المتغيرات يدويًا. أهمها:

- `LEGAL_DATASET_PATH`
- `QDRANT_PATH`
- `QDRANT_COLLECTION`

### البناء

```bash
python scripts/build_legal_index.py
```

هذا السكربت:

- يقرأ `jsonl` أو `json`
- يبني سجلات `article` و`article_chunk`
- يضيف سجلات `law` synthetic
- ينشئ collection جديدة
- يضيف payload indexes للفلترة
- يخزن vectors dense في Qdrant

## كيف يعمل الاسترجاع الآن

خطوات الاسترجاع القانونية أصبحت:

1. تحويل الاستعلام إلى embedding في وضع `query`
2. البحث أولًا في `article_chunk` لتحسين recall
3. البحث أيضًا في `article` و`law`
4. توسيع نتائج الـ chunks إلى `parent article`
5. دمج النتائج وإزالة التكرار على مستوى المادة
6. تطبيق reranking heuristic
7. إرجاع المادة القانونية النهائية مع `supporting_chunks`

### الفلاتر المدعومة

- `legal_domain`
- `law_number`
- `law_year`
- `status_normalized`
- `exclude_repealed`

### التعامل مع المواد الملغاة أو النصوص المريبة

- إذا كانت `status_normalized != current` يتم وسم السجل على أنه `is_repealed_candidate`
- إذا ظهرت مؤشرات نصية مثل `ملغاة` أو `منسوخ` يتم وسمه أيضًا
- يمكن إما استبعاده أو تنزيل ترتيبه
- إذا ظهرت ضوضاء OCR، يتم احتساب `noise_score` وإنقاص الوزن أثناء الترتيب

## سكربتات الديمو

### بناء الفهرس فقط

```bash
python scripts/build_legal_index.py
```

### تشغيل استعلامات ديمو على فهرس موجود

```bash
python scripts/demo_legal_queries.py
```

### بناء الفهرس ثم تشغيل ديمو كامل

```bash
python scripts/build_and_query_demo.py
```

## التشغيل في Hugging Face / Docker / VPS

### Hugging Face Spaces

- استخدم `uvicorn main:app --host 0.0.0.0 --port 7860`
- خزن dataset path في Secret أو mount خارجي
- اجعل `EMBEDDING_DEVICE=cpu` إذا لم يتوفر GPU

### Docker

- مرر `LEGAL_DATASET_PATH` و`QDRANT_PATH` عبر environment variables
- اعمل volume mount لـ dataset وQdrant path
- لو كان الموديل محليًا، mount مجلد `model/bge-m3`

### VPS منخفض الحركة

- شغل خدمة واحدة لـ FastAPI
- استخدم Qdrant local path للـ demo أو low traffic
- خزن الفهرس على disk path ثابت
- راقب استهلاك الذاكرة عند أول تحميل للموديل

## ملاحظات الهجرة من النظام القديم

- `video_id` لم يعد موجودًا في الـ payload الجديد
- `filename` لم يعد يستخدم كإشارة استرجاع
- `chunk_title/chunk_content` أصبحا `title/content`
- `retrieval_mode=by_filename/by_title/by_content` أزيل بالكامل
- الاسترجاع النهائي الآن article-centric وليس video-centric
- `hierarchical_retrieval/*` لم يعد جزءًا من مسار التشغيل

## الاختبارات

```bash
pytest
```

الاختبارات الحالية تغطي:

- تطبيع العربية
- عقد `/embed` و`/embed/batch`
- بناء الفهرس القانوني
- parent-child expansion
- فلترة السجلات الملغاة
- بنية مخرجات الاسترجاع
