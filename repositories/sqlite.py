import sqlite3
import uuid

class SQLiteRepository:
    def __init__(self, name='example.db', migrate=False):
        self.name = name
        if migrate:
            self.migrate()
    """
    Communication layer with SQLite database for storing and retrieving experiments, runs, tags, metrics, objects, properties, and audit logs.
    """
    def new_experiment(self, name: str):
        with sqlite3.connect(self.name) as conn:
            id = str(uuid.uuid4())
            c = conn.cursor()
            c.execute('INSERT INTO experiments (id, name) VALUES (?, ?)', (id, name))
            conn.commit()
            return id

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

    def new_tag(self, run_id: int, key: str, value: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            c.execute('SELECT id FROM tags WHERE run_id = ? AND key = ?', (run_id, key))
            row = c.fetchone()
            if row:
                # update existing tag
                c.execute('UPDATE tags SET value = ? WHERE id = ?', (value, row[0]))
                value =  row[0]
            else:
                # insert new tag
                c.execute('INSERT INTO tags (run_id, key, value) VALUES (?, ?, ?)', (run_id, key, value))
                value = c.lastrowid
            conn.commit()
            return value

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

    def find_best_model_run(self, experiment_id: str, metric: str):
        with sqlite3.connect(self.name) as conn:
            c = conn.cursor()
            query = '''
                SELECT r.name
                FROM runs r
                JOIN metrics m ON r.id = m.run_id
                JOIN tags t ON r.id = t.run_id
                WHERE r.experiment_id = ? AND m.key = ? AND tags.key = 'status' AND tags.value = 'champion'
                ORDER BY m.value ASC
                LIMIT 1
            '''
            c.execute(query, (experiment_id, metric))
            result = c.fetchone()
            if result:
                return result[0]  # return run id
            return None

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
                WHERE r.experiment_id = ? AND t.key = 'status.deployment' AND t.value = 'ready'
            '''
            c.execute(query, (experiment_id,))
            results = c.fetchall()
            return [{"id": id, "name": name} for id, name in results]

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

            c.execute('''CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY,
                table_name VARCHAR(255),
                reference_id VARCHAR(255),
                type VARCHAR(10),
                previous TEXT,
                current TEXT,
                created_at TIMESTAMP
            )''')

            c.execute('''CREATE INDEX IF NOT EXISTS idx_objects_runid_type ON objects(run_id, type)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_auditlogs_tablename_refid ON audit_logs(table_name, reference_id)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_properties_runid_key ON properties(run_id, key)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_metrics_runid_key ON metrics(run_id, key)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_tags_runid_key ON tags(run_id, key)''')

            conn.commit()

if __name__ == "__main__":
    # manual call for migration
    SQLiteRepository(migrate=True)