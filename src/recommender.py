"""
Etapa 2 — Recomendacao de Musicas
Playcatch: plataforma de streaming musical inteligente

Carrega resultados da analise de sentimentos e recomenda musicas
com base no humor desejado. Inclui sistema de feedback para
ajustar recomendacoes ao longo do tempo.
"""

import json
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)


# Sentimentos presentes na base (gerados pelo sentiment_analyzer)
SENTIMENTOS_VALIDOS = ["triste", "melancolico", "neutro", "energetico"]

# Mapeamento de sinonimos para facilitar input do usuario
# "feliz" e mapeado para "energetico" pois o modelo BERT nao gerou essa categoria na base atual
SINONIMOS = {
    "alegre": "energetico",
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
    "feliz": "energetico",
    "contente": "energetico",
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
        if not Path(path).exists():
            raise FileNotFoundError(f"Arquivo de sentimentos nao encontrado: {path}")
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

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("=" * 60)
    log.info("PLAYCATCH — Recomendador de Musicas")
    log.info("=" * 60)

    rec = MusicRecommender(str(data_path))
    moods = rec.get_available_moods()
    log.info("Humores disponiveis: %s", ", ".join(moods))

    for mood in moods:
        recs = rec.recommend(mood, n=3)
        log.info("=" * 40)
        log.info("Recomendacoes para humor: %s", mood.upper())
        log.info("=" * 40)
        for i, r in enumerate(recs, 1):
            log.info("  %d. %s — %s (score: %s)", i, r["titulo"], r["artista"], r["score"])

    log.info("=" * 40)
    log.info("TESTE DE FEEDBACK")
    log.info("=" * 40)
    if recs:
        song = recs[0]
        rec.register_feedback(song["titulo"], song["artista"], "like")
        rec.register_feedback(song["titulo"], song["artista"], "like")
        log.info("  2 likes registrados para: %s", song["titulo"])
        new_recs = rec.recommend(mood, n=3)
        log.info("  Reordenacao apos feedback:")
        for i, r in enumerate(new_recs, 1):
            log.info("    %d. %s — %s", i, r["titulo"], r["artista"])

    log.info("=" * 40)
    log.info("TESTE DE SINONIMOS")
    log.info("=" * 40)
    for sin in ["alegre", "chateado", "animado", "calmo"]:
        norm = rec.normalize_sentiment(sin)
        count = len(rec.recommend(sin, n=99))
        log.info("  '%s' -> '%s' (%d musicas)", sin, norm, count)


if __name__ == "__main__":
    main()
