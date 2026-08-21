"""SQLite index for the P: drive pipeline.

Lives on C:, never on P: — a drive dropout must not cost the index.
"""
import sqlite3

STATES = frozenset({
    "walked", "exif_read", "hashed",
    "planned_dupe", "quarantined", "evacuated",
    "stamped", "organised", "undated", "error",
})

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id             INTEGER PRIMARY KEY,
    path           TEXT UNIQUE NOT NULL,
    top_folder     TEXT NOT NULL,
    rel_dir        TEXT NOT NULL,
    filename       TEXT NOT NULL,
    ext            TEXT NOT NULL,
    size           INTEGER NOT NULL,
    mtime          REAL NOT NULL,
    kind           TEXT NOT NULL,
    partial_hash   TEXT,
    sha256         TEXT,
    pixel_hash     TEXT,
    percept_hash   TEXT,
    exif_dt        TEXT,
    exif_dt_source TEXT,
    exif_create_dt TEXT,
    camera_model   TEXT,
    sidecar_of     INTEGER REFERENCES files(id),
    state          TEXT NOT NULL DEFAULT 'walked'
);
CREATE INDEX IF NOT EXISTS idx_size  ON files(size);
CREATE INDEX IF NOT EXISTS idx_sha   ON files(sha256);
CREATE INDEX IF NOT EXISTS idx_dt    ON files(exif_dt);
CREATE INDEX IF NOT EXISTS idx_state ON files(state);
CREATE INDEX IF NOT EXISTS idx_part  ON files(partial_hash);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # CREATE TABLE IF NOT EXISTS will not add a column to an index built by an
    # earlier version, so bring older databases forward rather than failing on
    # an unknown column halfway through a multi-hour run.
    existing = {r[1] for r in conn.execute("PRAGMA table_info(files)")}
    for col in ("exif_create_dt",):
        if col not in existing:
            conn.execute(f"ALTER TABLE files ADD COLUMN {col} TEXT")
    conn.commit()


def upsert_file(conn: sqlite3.Connection, **fields) -> int:
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "path")
    conn.execute(
        f"INSERT INTO files ({cols}) VALUES ({marks}) "
        f"ON CONFLICT(path) DO UPDATE SET {updates}",
        tuple(fields.values()),
    )
    return conn.execute(
        "SELECT id FROM files WHERE path = ?", (fields["path"],)
    ).fetchone()[0]


def set_state(conn: sqlite3.Connection, file_id: int, state: str) -> None:
    if state not in STATES:
        raise ValueError(f"unknown state: {state}")
    conn.execute("UPDATE files SET state = ? WHERE id = ?", (state, file_id))
