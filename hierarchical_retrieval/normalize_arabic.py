"""
Arabic Text Normalization Script
==================================
Normalizes Arabic text in JSON file:
  - Remove diacritics (تشكيل)
  - Normalize Alef variants (أ، إ، آ، ء → ا)
  - Remove underscores and hyphens
  - Clean extra whitespace
"""

import json
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

INPUT_FILE = Path("Moshrif-knowledge-chunks.json")
OUTPUT_FILE = Path("Moshrif-knowledge-chunks-normalized.json")
BACKUP_FILE = Path("Moshrif-knowledge-chunks-backup.json")


# ──────────────────────────────────────────────────────────────────────────────
# Arabic Normalization Functions
# ──────────────────────────────────────────────────────────────────────────────

def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (تشكيل).
    
    Diacritics include:
    - َ (Fatha)
    - ُ (Damma)
    - ِ (Kasra)
    - ّ (Shadda)
    - ْ (Sukun)
    - ً (Tanween Fath)
    - ٌ (Tanween Damm)
    - ٍ (Tanween Kasr)
    """
    # Arabic diacritics Unicode range
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670]')
    return arabic_diacritics.sub('', text)


def normalize_alef(text: str) -> str:
    """
    Normalize all Alef variants to simple Alef (ا).
    
    Variants:
    - أ (Alef with Hamza above)
    - إ (Alef with Hamza below)
    - آ (Alef with Madda)
    - ء (Hamza alone) - keep this for now, only normalize when on Alef
    """
    # Normalize Alef variants
    text = re.sub('[إأآ]', 'ا', text)
    return text


def normalize_yaa(text: str) -> str:
    """
    Normalize Yaa variants.
    """
    # ى (Alef Maksura) → ي (Yaa)
    text = text.replace('ى', 'ي')
    return text


def normalize_taa(text: str) -> str:
    """
    Normalize Taa Marbuta.
    """
    # ة (Taa Marbuta) → ه (Haa) - optional, comment out if not needed
    # text = text.replace('ة', 'ه')
    return text


def remove_special_chars(text: str) -> str:
    """
    Remove underscores, hyphens, and other special characters.
    """
    # Remove _ and -
    text = text.replace('_', ' ')
    text = text.replace('-', ' ')
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def normalize_arabic_text(text: str) -> str:
    """
    Apply all normalization steps.
    """
    if not text or not isinstance(text, str):
        return text
    
    # 1. Remove diacritics
    text = remove_diacritics(text)
    
    # 2. Normalize Alef
    text = normalize_alef(text)
    
    # 3. Normalize Yaa
    text = normalize_yaa(text)
    
    # 4. Normalize Taa (optional)
    text = normalize_taa(text)
    
    # 5. Remove special characters
    text = remove_special_chars(text)
    
    return text


# ──────────────────────────────────────────────────────────────────────────────
# JSON Processing
# ──────────────────────────────────────────────────────────────────────────────

def normalize_video(video: dict) -> dict:
    """
    Normalize all text fields in a video object.
    """
    # Normalize filename
    if 'filename' in video:
        video['filename'] = normalize_arabic_text(video['filename'])
    
    # Normalize telegram_url (keep as is, no normalization)
    # telegram_url stays the same
    
    # Normalize chunks
    if 'chunks' in video:
        for chunk in video['chunks']:
            if 'topicTitle' in chunk:
                chunk['topicTitle'] = normalize_arabic_text(chunk['topicTitle'])
            
            if 'topicContent' in chunk:
                chunk['topicContent'] = normalize_arabic_text(chunk['topicContent'])
    
    return video


def normalize_json_file():
    """
    Main normalization pipeline.
    """
    print("="*80)
    print("Arabic Text Normalization")
    print("="*80)
    
    # Check if file exists
    if not INPUT_FILE.exists():
        print(f"❌ Error: File not found: {INPUT_FILE}")
        return
    
    # Create backup
    print(f"\n📦 Creating backup: {BACKUP_FILE}")
    with INPUT_FILE.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    with BACKUP_FILE.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Backup created")
    
    # Process videos
    print(f"\n🔄 Normalizing {len(data)} videos...")
    
    total_chunks = 0
    for i, video in enumerate(data, 1):
        video = normalize_video(video)
        total_chunks += len(video.get('chunks', []))
        
        if i % 50 == 0:
            print(f"   Processed {i}/{len(data)} videos...")
    
    print(f"✅ Normalized {len(data)} videos, {total_chunks} chunks")
    
    # Save normalized data
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    with OUTPUT_FILE.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved successfully")
    
    # Show examples
    print("\n" + "="*80)
    print("📊 Normalization Examples:")
    print("="*80)
    
    if data and 'chunks' in data[0] and data[0]['chunks']:
        first_chunk = data[0]['chunks'][0]
        print(f"\nFilename: {data[0].get('filename', 'N/A')}")
        print(f"Title: {first_chunk.get('topicTitle', 'N/A')[:100]}...")
        print(f"Content: {first_chunk.get('topicContent', 'N/A')[:150]}...")
    
    print("\n" + "="*80)
    print("✅ Normalization complete!")
    print("="*80)
    print(f"\n📁 Files created:")
    print(f"   - Backup: {BACKUP_FILE}")
    print(f"   - Normalized: {OUTPUT_FILE}")
    print(f"\n💡 To use normalized file, rename it to:")
    print(f"   mv {OUTPUT_FILE} {INPUT_FILE}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    normalize_json_file()
