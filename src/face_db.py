"""Persistent face embedding storage and retrieval via FAISS + SQLite.

The index stores *L2-normalised* 512-D embeddings. A separate SQLite database
keeps a mapping from a UUIDv7 (string) to the person's *display name*.

Embeddings are stored in an ``embeddings.npy`` NumPy file so that the FAISS
index can be rebuilt on start-up - this keeps the on-disk format simple and
portable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss  # type: ignore
import numpy as np

from . import config
from .schema import Person, db

# ---------------------------------------------------------------------------
# Face DB - FAISS + SQLite
# ---------------------------------------------------------------------------


class FaceDatabase:
    """Store & retrieve face embeddings via FAISS (similarity search).

    The FAISS index uses *inner-product* similarity (a.k.a. cosine similarity
    given normalised vectors) which is equivalent to the dot-product between
    two L2-normalised vectors. That is why every embedding *must* be
    pre-normalised before being added or queried.
    """

    def __init__(
        self,
        db_path: str | Path = config.DB_PATH,
        embeddings_path: str | Path = config.EMBEDDINGS_PATH,
        dim: int = config.EMBEDDING_DIM,
    ) -> None:
        self.db_path = Path(db_path)
        self.embeddings_path = Path(embeddings_path)
        self.dim = dim

        # ------------- SQLite ----------------
        db.connect(reuse_if_open=True)

        # ------------- Embeddings -------------
        if self.embeddings_path.exists():
            self._embeddings = np.load(self.embeddings_path)
        else:
            self._embeddings = np.empty((0, dim), dtype=np.float32)

        # Sanity-check: SQLite rows must match #embeddings
        row_cnt = Person.select().count()
        if row_cnt != len(self._embeddings):
            raise RuntimeError(
                "Mismatch between stored embeddings and SQLite rows - "
                "please ensure both files are in sync."
            )

        # ------------- FAISS index ------------
        self._index = faiss.IndexFlatIP(dim)
        if len(self._embeddings):
            self._index.add(self._embeddings)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def add_person(
        self,
        embedding: np.ndarray,
        person_id: str,
        name: str,
        person_type: str,
        admission_number: str | None = None,
        room_id: str | None = None,
        picture_filename: str | None = None,
    ) -> str:
        """Add a *single* embedding → returns generated UUID."""
        emb = self._prepare_vec(embedding)

        # Persist - SQLite first so we fail atomically before mutating others
        Person.create(
            uniqueId=person_id,
            name=name,
            personType=person_type,
            admissionNumber=admission_number,
            roomId=room_id,
            pictureFileName=picture_filename or "placeholder.jpg",
        )

        # Persist - embeddings (append-save to disk)
        self._embeddings = np.vstack([self._embeddings, emb])
        np.save(self.embeddings_path, self._embeddings)
        # Update FAISS in-memory index
        self._index.add(emb[np.newaxis, :])

        return person_id

    def search(
        self, embedding: np.ndarray, threshold: float = config.DB_SEARCH_THRESHOLD
    ) -> Optional[tuple[str, float]]:
        """Return the *name* and *similarity* that best matches the given
        embedding or ``None``.

        ``threshold`` is applied to the cosine similarity (1.0 == perfect
        match). 0.35 is a conservative default - tweak to your needs.
        """
        if self._index.ntotal == 0:
            return None, 0.0

        vec = self._prepare_vec(embedding)
        sims, idxs = self._index.search(vec[np.newaxis, :], 1)
        sim = float(sims[0, 0])
        best_idx = int(idxs[0, 0])
        if best_idx == -1 or sim < threshold:
            return None, sim

        # Fetch the name corresponding to *rowid* = best_idx + 1
        person = Person.select(Person.name).order_by(Person.uniqueId).offset(best_idx).scalar()
        return person, sim

    # ------------------------------------------------------------------
    # Helper utils
    # ------------------------------------------------------------------

    def _prepare_vec(self, v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32)
        # Ensure L2-normalised
        norm = np.linalg.norm(v)
        if norm == 0.0:
            raise ValueError("Zero-length embedding cannot be added/searched.")
        return v / norm

    def close(self) -> None:
        """Close the database connection."""
        if not db.is_closed():
            db.close()
