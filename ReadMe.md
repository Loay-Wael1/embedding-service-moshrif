# 🧠 Embedding Service — Super Simple Setup (BGE-M3 + FastAPI)

This project gives you a ready-to-use **Embedding API** (1024-dim vectors) using **FastAPI + BGE-M3**.  
You don’t need to know Python. Just follow the steps exactly, and the server will run successfully.

---

# ✅ What You Will Do

1. Install Python  
2. Clone the project  
3. Install the requirements  
4. Run the API  
5. Test it  

That's all.

---

# 📌 1) Install Python 3.10

Download Python 3.10 from this link:
https://www.python.org/downloads/release/python-31011/

During installation:
- ✔ Check **Add Python to PATH**  
- ✔ Enable all optional features  
- ✔ Install normally (Next → Next)

Then confirm installation:

```
python --version
```

It should show:

```
Python 3.10.x
```

---

# 📌 2) Enable Developer Mode (Important for Windows)

Press **Windows + R**, type:

```
start ms-settings:developers
```

Enable:

✔ **Developer Mode**

Close the window.

---

# 📌 3) Clone the Project

Open Command Prompt (CMD) and run:

```
git clone https://github.com/Mohamed-ALQarram/Embedding-Service.git
cd Embedding-Service
```

# 📌 4) Create a Virtual Environment (venv)

This step is **only required if you have multiple Python versions installed**,  
or if you want to keep this project separate from your system packages.  
If you only have **one** Python version installed, you can still do this step — but it's optional.

To create a virtual environment:

```
python -m venv venv
```

Then activate it:

**Windows:**
```
venv\Scripts\activate
```

**Mac/Linux:**
```
source venv/bin/activate
```

When activated, your terminal will show:

```
(venv)
```

This means everything you install now will stay inside this project only.
---

# 📌 5) Install Dependencies

Run:

```
pip install -r requirements.txt
```

Wait until it finishes.

---

# 📌 6) Run the API Server

Start the server:

```
uvicorn main:app
```

⚠️ VERY IMPORTANT  
**Do NOT use `--reload`**  
It will break the model by loading it twice.

If everything is correct, you will see:

```
Uvicorn running on http://127.0.0.1:8000
```

---

# 📌 7) Test the API (Super Easy)

Open your browser and go to:

👉 http://127.0.0.1:8000/docs

Click on `/embed` → **Try it out**

Enter a text:

```
hello world
```

Click **Execute**

You will get a long list of numbers = the embedding vector.

---

# 📌 8) Offline Mode (Optional — Use Only If Online Download Fails)

You **ONLY** need Offline Mode if the model fails to download automatically.  
This method lets you download the model **manually** and run the API without internet.

---

## ✅ Step 1 — Create the Model Folder

Create the following directory:

```
model/bge-m3/
```

Make sure the folder names are **exactly** the same.

---

## ✅ Step 2 — Download the Required Files Manually

Go to the model page on HuggingFace:

👉 https://huggingface.co/BAAI/bge-m3/tree/main
Download **these exact files**:

```
config.json
pytorch_model.bin
tokenizer.json
tokenizer_config.json
special_tokens_map.json
sentencepiece.bpe.model
```

⚠️ **Important Notes**  
- Make sure the **file names AND extensions** are exactly correct.  
- Do NOT rename any file.  
- Do NOT put them inside subfolders.

---

## ✅ Step 3 — Place All Files in the Folder

Put all downloaded files directly inside:

```
model/bge-m3/
```

Your folder structure should now look like this:

```
Embedding-Service/
│
├── main.py
├── model_loader.py
├── config.py
├── requirements.txt
└── model/
    └── bge-m3/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        └── sentencepiece.bpe.model
```

---

## ✅ Step 4 — Update `config.py`

Open `config.py` and replace the model path with the **local folder path**:

```python
MODEL_NAME = "./model/bge-m3"
DEVICE = "cpu"
```

Explanation:

- `MODEL_NAME` → tells the program **not to download** the model and instead load it from the folder.
- `DEVICE = "cpu"` → ensures the model runs on any Windows machine without GPU problems.

---

## 🎉 Done!

Now you can run the API normally:

```
uvicorn main:app
```

Even if you have **no internet connection**, the model will load successfully.