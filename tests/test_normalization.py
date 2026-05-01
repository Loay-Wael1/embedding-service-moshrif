from app.preprocessing.legal_arabic import assess_text_quality, normalize_legal_arabic


def test_legal_normalization_preserves_article_numbers():
    text = "المــادةُ ١  -  إبرامُ العقدِ"
    assert normalize_legal_arabic(text) == "المادة 1 - ابرام العقد"


def test_quality_detection_flags_ocr_like_noise():
    assessment = assess_text_quality("ذه المدة: . أ فى حالة إخفاء بيانات . .")
    assert assessment.noise_score > 0
    assert "broken_punctuation" in assessment.warnings
