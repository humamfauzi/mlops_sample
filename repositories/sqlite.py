import sqlite3
import uuid
import json
import io
import pickle
from typing import List

from repositories.struct import ModelObject, TransformationObject
from .struct import TransformationInstruction, TransformationObject, ModelObject

class SQLiteRepository:
    """
    Communication layer with SQLite database for storing and retrieving experiments, runs, tags, metrics, objects, properties, and audit logs.
    """
    def __init__(self, name='example.db', migrate=False):
        self.name = name
        if migrate:
            self.migrate()
    def new_experiment(self, name: str):
        with sqlite3.connect(self.name) as conn:
            id = str(uuid.uuid4())
            c = conn.cursor()
            c.execute('INSERT INTO experiments (id, name) VALUES (?, ?)', (id, name))
            conn.commit()
            return id

    def ensure_experiment(self, experiment_id: str):
        """Register `experiment_id` if it is not already known.

        Runs reference an experiment_id from configuration, but nothing ever
        created the corresponding row, so the experiments table stayed empty
        while 91 runs pointed at 'experiment_001'. A typo in EXPERIMENT_ID
        would silently start a parallel universe of runs.
        """
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute(
                'INSERT OR IGNORE INTO experiments (id, name) VALUES (?, ?)',
                (experiment_id, experiment_id),
            )
            conn.commit()
        return experiment_id

    def new_run(self, name: str, experiment_id: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            v = (name, experiment_id)
            c.execute('INSERT INTO runs (name, experiment_id) VALUES (?, ?)', v)
            conn.commit()
            return c.lastrowid

    def new_child_run(self, name: str, parent_id: int, experiment_id: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            v = (name, parent_id, experiment_id)
            c.execute('INSERT INTO runs (name, parent_id, experiment_id) VALUES (?, ?, ?)', v)
            conn.commit()
            return c.lastrowid

    def new_metric(self, run_id: int, key: str, value: float):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            try:
                value = round(float(value), 3)
            except (TypeError, ValueError):
                # leave value as-is if it can't be converted to float
                pass
            c.execute('INSERT INTO metrics (run_id, key, value) VALUES (?, ?, ?)', (run_id, key, value))
            conn.commit()
            return c.lastrowid

    def new_object(self, run_id: int, type: str, url: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO objects (run_id, type, url) VALUES (?, ?, ?)', (run_id, type, url))
            conn.commit()
            return c.lastrowid

    def new_property(self, run_id: int, key: str, value: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO properties (run_id, key, value) VALUES (?, ?, ?)', (run_id, key, value))
            conn.commit()
            return c.lastrowid

    def find_best_model_within_run(self, parent_run_id: int, metric: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            query = '''
                SELECT r.name
                FROM runs r
                JOIN metrics m ON r.id = m.run_id
                WHERE r.parent_id = ? AND m.key = ?
                ORDER BY m.value ASC
                LIMIT 1
            '''
            c.execute(query, (parent_run_id, metric))
            result = c.fetchone()
            if result:
                return result[0]  # return run id
            return None

    def get_all_metrics(self, run_id: int):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT key, value FROM metrics WHERE run_id = ?', (run_id,))
            results = c.fetchall()
            return {key: value for key, value in results}

    def get_all_child_runs(self, parent_run_id: int):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT id, name FROM runs WHERE parent_id = ?', (parent_run_id,))
            results = c.fetchall()
            return [{"id": id, "name": name} for id, name in results]

    def get_all_objects(self, run_id: int):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT type, url FROM objects WHERE run_id = ?', (run_id,))
            results = c.fetchall()
            return {type: url for type, url in results}

    def get_all_objects_under_parent(self, parent_run_id: int):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            query = '''
                SELECT o.type, o.url
                FROM objects o
                JOIN runs r ON o.run_id = r.id
                WHERE r.parent_id = ?
            '''
            c.execute(query, (parent_run_id,))
            results = c.fetchall()
            return {type: url for type, url in results}

    def get_all_published_candidates(self, experiment_id: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            query = '''
                SELECT r.id, r.name
                FROM runs r
                JOIN tags t ON r.id = t.run_id
                WHERE r.experiment_id = ? AND t.key = 'status.deployment' AND t.value = 'published'
            '''
            c.execute(query, (experiment_id,))
            results = c.fetchall()
            return [{"id": id, "name": name} for id, name in results]
    
    def get_model_run_id(self, model_name: str, parent_run_id: int = None):
        """Resolve a model's short run name to (id, parent_id).

        Scoped to the parent run when one is supplied. Names are six characters
        drawn from a 36-symbol alphabet, so they are only unique by luck; an
        unscoped lookup would silently return another run's model on a
        collision, and the caller would nominate the wrong thing.
        """
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            if parent_run_id is None:
                c.execute('SELECT id, parent_id FROM runs WHERE name = ?', (model_name,))
            else:
                c.execute(
                    'SELECT id, parent_id FROM runs WHERE name = ? AND parent_id = ?',
                    (model_name, parent_run_id),
                )
            result = c.fetchone()
            if result:
                return result  # return run id
            scope = "" if parent_run_id is None else f" under parent run {parent_run_id}"
            raise ValueError(f"Model with name {model_name} not found{scope}")

    def run_name_exists(self, name: str) -> bool:
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT 1 FROM runs WHERE name = ? LIMIT 1', (name,))
            return c.fetchone() is not None

    def get_intent(self, run_id: int):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT value FROM properties WHERE run_id = ? AND key = ?', (run_id, "name.intent"))
            result = c.fetchone()
            if result:
                return result[0]  # return intent value
            raise ValueError(f"Intent not found for run id {run_id}")

    def select_previously_published(self, experiment_id: str, intent: str, primary_metric: str):
        """Best score among the runs currently published for this intent.

        Ordered explicitly rather than relying on row order. The score being
        compared is the published run's *best* child, so a lower value wins for
        the error metrics this project uses.
        """
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            query = '''
                SELECT r.id, m.value
                FROM runs r
                JOIN tags t ON r.id = t.run_id
                JOIN runs rc ON rc.parent_id = r.id
                JOIN properties p ON r.id = p.run_id
                JOIN metrics m ON rc.id = m.run_id
                WHERE t.key = 'status.deployment' AND t.value = 'published'
                    AND p.key = 'name.intent' AND p.value = ? AND r.experiment_id = ? AND m.key = ?
                ORDER BY m.value ASC, r.id ASC
                LIMIT 1
            '''
            c.execute(query, (intent, experiment_id, f"validation.test.{primary_metric}"))
            result = c.fetchone()
            if result is None:
                return None, float('inf')
            return result

    def upsert_tag(self, run_id: int, key: str, value: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT id FROM tags WHERE run_id = ? AND key = ?', (run_id, key))
            row = c.fetchone()
            if row:
                c.execute('UPDATE tags SET value = ? WHERE id = ?', (value, row[0]))
                value =  row[0]
            else:
                c.execute('INSERT INTO tags (run_id, key, value) VALUES (?, ?, ?)', (run_id, key, value))
                value = c.lastrowid
            conn.commit()
            return value

    def get_object(self, run_id: int, key: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT url FROM objects WHERE run_id = ? AND type = ?', (run_id, key))
            result = c.fetchone()
            if result:
                return json.loads(result[0])  # return the JSON object
            raise ValueError(f"Object with key {key} not found for run id {run_id}")

    def find_tagged_best_model(self, run_id: int):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('''
                SELECT r.id, r.name FROM runs r
                JOIN tags t ON r.id = t.run_id
                JOIN runs rp ON r.parent_id = rp.id
                WHERE rp.id = ? AND t.key = 'level' AND t.value = 'best'
            ''', (run_id,))
            result = c.fetchone()
            if result:
                return result
            raise ValueError(f"Model best model not found for run id {run_id}")

    def find_property(self, run_id: int, key: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT value FROM properties WHERE run_id = ? AND key = ?', (run_id, key))
            result = c.fetchone()
            if result:
                return result[0]  # return property value
            return None

    def get_all_properties(self, run_id: int):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT key, value FROM properties WHERE run_id = ?', (run_id,))
            results = c.fetchall()
            return {key: value for key, value in results}

    def get_all_metrics(self, run_id: int):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT key, value FROM metrics WHERE run_id = ?', (run_id,))
            results = c.fetchall()
            return {key: value for key, value in results}

    def migrate(self):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            # Enable foreign key support
            c.execute('PRAGMA foreign_keys = ON;')
            c.execute('''CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name VARCHAR(100)
            )''')

            c.execute('''CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100),
                parent_id INTEGER,
                experiment_id TEXT,
                FOREIGN KEY(parent_id) REFERENCES runs(id)
            )''')

            c.execute('''CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                key VARCHAR(100),
                value VARCHAR(255),
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )''')

            c.execute('''CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                key VARCHAR(100),
                value FLOAT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )''')

            c.execute('''CREATE TABLE IF NOT EXISTS objects (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                type VARCHAR(100),
                url TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )''')

            c.execute('''CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                key VARCHAR(100),
                value VARCHAR(255),
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )''')

            c.execute('''CREATE INDEX IF NOT EXISTS idx_objects_runid_type ON objects(run_id, type)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_properties_runid_key ON properties(run_id, key)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_metrics_runid_key ON metrics(run_id, key)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_tags_runid_key ON tags(run_id, key)''')

            # audit_logs was created here and never written to by anything.
            c.execute('''DROP TABLE IF EXISTS audit_logs''')

            conn.commit()

class ObjectStorage:
    # Class constants
    INTENT_TRANSFORMATION_INSTRUCTION = "transformation_instruction"
    INTENT_TRANSFORMATION_OBJECT = "transformation_object"
    INTENT_MODEL = "model"

    TYPE_JSON = "json"
    TYPE_PICKLE = "pkl"
    def __init__(self, name='example.db', migrate=False):
        self.name = name
        if migrate:
            self.migrate()

    def migrate(self):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            # `hash` is retained in the schema for backward compatibility with
            # databases written before it was dropped: it was computed and
            # stored but never read, and content-addressed dedup would have
            # saved nothing (the duplicated blobs are ~118-byte transform
            # pickles, while the 900 MB of models are unique per run).
            c.execute('''CREATE TABLE IF NOT EXISTS blobs (
                      id INTEGER PRIMARY KEY,
                      run_id INTEGER,
                      intent VARCHAR(100),
                      type VARCHAR(100),
                      data BLOB
            )''')

            c.execute('''CREATE INDEX IF NOT EXISTS idx_blobs_runid_type ON blobs(run_id, type)''')
            # Blobs are keyed by (run_id, intent). Without this, re-saving a
            # run's artifacts silently appended duplicate rows.
            c.execute('''CREATE UNIQUE INDEX IF NOT EXISTS idx_blobs_runid_intent
                         ON blobs(run_id, intent)''')
            conn.commit()

    def insert_blob(self, run_id: str, intent: str, type: str, data: bytes) -> None:
        """Store a blob under (run_id, intent), replacing any previous value.

        Re-saving the same artifact for a run is now idempotent rather than
        appending a second row that `load_*` would pick between arbitrarily.
        """
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('INSERT OR REPLACE INTO blobs (run_id, intent, type, data) VALUES (?, ?, ?, ?)',
                      (run_id, intent, type, data))
            conn.commit()

    def save_transformation_instruction(self, run_id: str, instructions: List[TransformationInstruction]):
        all_instruction = [inst.to_dict() for inst in instructions]
        json_data = json.dumps(all_instruction, indent=2, ensure_ascii=False)
        data = json_data.encode('utf-8')
        return self.insert_blob(run_id, "transformation_instruction", "json", data)

    def save_transformation_object(self, run_id: str, transformation_objects: List[TransformationObject]):
        """Persist fitted transformation objects; returns the filenames written.

        Matches the disk backend, which already returned filenames.
        """
        filenames = []
        for obj in transformation_objects:
            buffer = io.BytesIO()
            pickle.dump(obj.object, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            buffer.seek(0)
            data = buffer.getvalue()
            intent = f"transformation_object/{obj.filename}"
            self.insert_blob(run_id, intent, "pkl", data)
            filenames.append(obj.filename)
        return filenames

    def load_transformation_instruction(self, run_id: str) -> List[TransformationInstruction]:
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT data FROM blobs WHERE run_id = ? AND intent = ? AND type = ?', 
                      (run_id, self.INTENT_TRANSFORMATION_INSTRUCTION, self.TYPE_JSON))
            result = c.fetchone()
            if result:
                data = json.loads(result[0].decode('utf-8'))
                return [TransformationInstruction(**item) for item in data]
            raise ValueError(f"Transformation instruction not found for run id {run_id}")

    def load_transformation_object(self, run_id: str, transformation_id: str = "", ttype: str = ""):
        intent = f"{self.INTENT_TRANSFORMATION_OBJECT}/{transformation_id}.{ttype}"
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT data FROM blobs WHERE run_id = ? AND intent = ? AND type = ?', 
                      (run_id, intent, self.TYPE_PICKLE))
            result = c.fetchone()
            if result:
                data = pickle.loads(result[0])
                return TransformationObject(filename=transformation_id, object=data)
            raise ValueError(f"Transformation object {transformation_id} not found for run id {run_id}")

    def save_model(self, run_id:str,  model: ModelObject):
        # Serialize model using pickle
        buffer = io.BytesIO()
        pickle.dump(model.object, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        buffer.seek(0)
        data = buffer.getvalue()
        
        # Insert into blobs
        intent = f"{self.INTENT_MODEL}/{model.filename}"
        return self.insert_blob(run_id, intent, self.TYPE_PICKLE, data)
    
    def load_model(self, run_id:str, model_filename: str):
        intent = f"{self.INTENT_MODEL}/{model_filename}"
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT data FROM blobs WHERE run_id = ? AND intent = ? AND type = ?', 
                      (run_id, intent, self.TYPE_PICKLE))
            result = c.fetchone()
            if result:
                data = pickle.loads(result[0])
                return ModelObject(filename=model_filename, object=data)
            raise ValueError(f"Model {model_filename} not found for run id {run_id}")



if __name__ == "__main__":
    # manual call for migration
    SQLiteRepository(migrate=True)
