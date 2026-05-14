"""
Etapa 2 — Recomendacao de Musicas
Playcatch: plataforma de streaming musical inteligente

Carrega resultados da analise de sentimentos e recomenda musicas
com base no humor desejado. Inclui sistema de feedback para
ajustar recomendacoes ao longo do tempo.
"""

import json
import random
from pathlib import Path


# Sentimentos disponiveis no sistema
SENTIMENTOS_VALIDOS = ["triste", "melancolico", "neutro", "feliz", "energetico"]

# Mapeamento de sinonimos para facilitar input do usuario
SINONIMOS = {
    "alegre": "feliz",
    "animado": "energetico",
    "empolgado": "energetico",
    "calmo": "neutro",
    "tranquilo": "neutro",
    "relaxado": "neutro",
    "triste": "triste",
    "chateado": "triste",
    "deprimido": "triste",
    "melancolico": "melancolico",
    "saudade": "melancolico",
    "nostalgico": "melancolico",
    "feliz": "feliz",
    "contente": "feliz",
    "neutro": "neutro",
    "energetico": "energetico",
}


class MusicRecommender:
    """Sistema de recomendacao musical baseado em sentimentos."""

    def __init__(self, data_path: str):
        self.songs = self._load_data(data_path)
        self.feedback: dict[str, dict[str, int]] = {}
        self._init_feedback()

    def _load_data(self, path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _init_feedback(self):
        """Inicializa contadores de feedback para cada musica."""
        for song in self.songs:
            key = f"{song['titulo']}|{song['artista']}"
            self.feedback[key] = {"likes": 0, "skips": 0}

    def normalize_sentiment(self, sentiment: str) -> str:
        """Normaliza input do usuario para sentimento padrao."""
        s = sentiment.lower().strip()
        return SINONIMOS.get(s, s)

    def recommend(self, sentiment: str, n: int = 5, exclude: list[str] | None = None) -> list[dict]:
        """
        Recomenda ate N musicas com base no sentimento desejado.
        Ordena por score de confianca + feedback positivo.
        """
        target = self.normalize_sentiment(sentiment)
        if target not in SENTIMENTOS_VALIDOS:
            return []

        exclude = exclude or []
        candidates = [
            s for s in self.songs
            if s["sentimento"] == target and s["titulo"] not in exclude
        ]

        # Ordena por score + bonus de likes - penalidade de skips
        def sort_key(song):
            key = f"{song['titulo']}|{song['artista']}"
            fb = self.feedback.get(key, {"likes": 0, "skips": 0})
            return song["score"] + (fb["likes"] * 0.1) - (fb["skips"] * 0.05)

        candidates.sort(key=sort_key, reverse=True)
        return candidates[:n]

    def register_feedback(self, titulo: str, artista: str, action: str) -> bool:
        """Registra feedback do usuario (like ou skip)."""
        key = f"{titulo}|{artista}"
        if key not in self.feedback:
            return False
        if action == "like":
            self.feedback[key]["likes"] += 1
        elif action == "skip":
            self.feedback[key]["skips"] += 1
        else:
            return False
        return True

    def get_available_moods(self) -> list[str]:
        """Retorna sentimentos que possuem musicas na base."""
        moods = set(s["sentimento"] for s in self.songs)
        return sorted(moods)

    def get_song_by_title(self, titulo: str) -> dict | None:
        """Busca musica por titulo."""
        for s in self.songs:
            if s["titulo"].lower() == titulo.lower():
                return s
        return None


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "lyrics_sentiment.json"

    print("=" * 60)
    print("PLAYCATCH — Recomendador de Musicas")
    print("=" * 60)

    rec = MusicRecommender(str(data_path))
    moods = rec.get_available_moods()
    print(f"\nHumores disponiveis: {', '.join(moods)}")

    # Teste: recomendacoes para cada humor
    for mood in moods:
        recs = rec.recommend(mood, n=3)
        print(f"\n{'='*40}")
        print(f"Recomendacoes para humor: {mood.upper()}")
        print(f"{'='*40}")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['titulo']} — {r['artista']} (score: {r['score']})")

    # Teste: feedback
    print(f"\n{'='*40}")
    print("TESTE DE FEEDBACK")
    print(f"{'='*40}")
    if recs:
        song = recs[0]
        rec.register_feedback(song["titulo"], song["artista"], "like")
        rec.register_feedback(song["titulo"], song["artista"], "like")
        print(f"  2 likes registrados para: {song['titulo']}")
        new_recs = rec.recommend(mood, n=3)
        print(f"  Reordenacao apos feedback:")
        for i, r in enumerate(new_recs, 1):
            print(f"    {i}. {r['titulo']} — {r['artista']}")

    # Teste: sinonimos
    print(f"\n{'='*40}")
    print("TESTE DE SINONIMOS")
    print(f"{'='*40}")
    for sin in ["alegre", "chateado", "animado", "calmo"]:
        norm = rec.normalize_sentiment(sin)
        count = len(rec.recommend(sin, n=99))
        print(f"  '{sin}' -> '{norm}' ({count} musicas)")


if __name__ == "__main__":
    main()
