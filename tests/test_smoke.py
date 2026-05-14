"""Smoke test — verifica recommender e chatbot sem chamar modelos HuggingFace."""
import json
import tempfile
from pathlib import Path

import pytest

from src.recommender import MusicRecommender, SINONIMOS, SENTIMENTOS_VALIDOS


@pytest.fixture
def sample_data(tmp_path):
    """Cria dataset minimo para testes."""
    songs = [
        {"titulo": "Alegria", "artista": "A", "letra": "la la", "estrelas": 5, "sentimento": "energetico", "score": 0.9},
        {"titulo": "Saudade", "artista": "B", "letra": "ai ai", "estrelas": 2, "sentimento": "melancolico", "score": 0.8},
        {"titulo": "Calma", "artista": "C", "letra": "om om", "estrelas": 3, "sentimento": "neutro", "score": 0.7},
    ]
    path = tmp_path / "test_songs.json"
    path.write_text(json.dumps(songs, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_recommender_loads(sample_data):
    rec = MusicRecommender(sample_data)
    assert len(rec.songs) == 3


def test_recommend_by_mood(sample_data):
    rec = MusicRecommender(sample_data)
    recs = rec.recommend("energetico", n=5)
    assert len(recs) == 1
    assert recs[0]["titulo"] == "Alegria"


def test_recommend_empty_mood(sample_data):
    rec = MusicRecommender(sample_data)
    recs = rec.recommend("inexistente", n=5)
    assert recs == []


def test_sinonimos_mapping():
    assert SINONIMOS["feliz"] == "energetico"
    assert SINONIMOS["chateado"] == "triste"
    assert SINONIMOS["calmo"] == "neutro"


def test_feedback_affects_order(sample_data):
    rec = MusicRecommender(sample_data)
    rec.register_feedback("Calma", "C", "like")
    rec.register_feedback("Calma", "C", "like")
    # Feedback nao muda mood filter, mas affects ordering within same mood
    recs = rec.recommend("neutro")
    assert recs[0]["titulo"] == "Calma"


def test_normalize_sentiment(sample_data):
    rec = MusicRecommender(sample_data)
    assert rec.normalize_sentiment("alegre") == "energetico"
    assert rec.normalize_sentiment("TRISTE") == "triste"


def test_get_song_by_title(sample_data):
    rec = MusicRecommender(sample_data)
    song = rec.get_song_by_title("alegria")
    assert song is not None
    assert song["artista"] == "A"
    assert rec.get_song_by_title("nao existe") is None


def test_chatbot_respond_greeting(sample_data):
    from src.chatbot import PlaycatchChatbot
    rec = MusicRecommender(sample_data)
    bot = PlaycatchChatbot(rec)
    resp = bot.respond("oi")
    assert "DJ" in resp or "humor" in resp.lower()


def test_chatbot_respond_empty(sample_data):
    from src.chatbot import PlaycatchChatbot
    rec = MusicRecommender(sample_data)
    bot = PlaycatchChatbot(rec)
    resp = bot.respond("")
    assert len(resp) > 0


def test_chatbot_farewell(sample_data):
    from src.chatbot import PlaycatchChatbot
    rec = MusicRecommender(sample_data)
    bot = PlaycatchChatbot(rec)
    resp = bot.respond("tchau")
    assert "mais" in resp.lower() or "ate" in resp.lower()


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        MusicRecommender("/tmp/nao_existe_xyz.json")
