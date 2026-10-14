"""Tests for 07-data-and-deployment/02-databases.

Three groups:
- Plain SQL tasks run against an in-memory SQLite database with small made-up tables
  (no server needed).
- MySQL-only tasks (CREATE DATABASE, AUTO_INCREMENT, ENUM, triggers, procedures,
  functions) run against a real MySQL server through the `mysql` command-line client.
  They are skipped unless one is reachable (see the README for a Docker one-liner).
- MongoDB: the pymongo functions run against `mongomock` (an in-memory fake), the
  mongosh scripts need a real MongoDB server and `mongosh` (skipped otherwise).
"""
import os
import pathlib
import shutil
import sqlite3
import subprocess

import pytest

from helpers import project_dir

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def task_path(name):
    path = project_dir(__file__) / name
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    return path


# =====================================================================
# SQLite part
# =====================================================================

def run_sqlite(conn, name):
    """Run every statement of a task file; return (column names, rows) of the last one."""
    sql = task_path(name).read_text()
    cur = None
    for stmt in sql.split(";"):
        if stmt.strip() and any(l.strip() and not l.strip().startswith("--")
                                for l in stmt.splitlines()):
            cur = conn.execute(stmt)
    conn.commit()
    if cur is None or cur.description is None:
        return [], []
    return [d[0] for d in cur.description], cur.fetchall()


def check_sorted_desc(rows, expected, key=1):
    """Same rows as expected, ordered by `key` descending (ties in any order)."""
    assert sorted(rows) == sorted(expected)
    assert [r[key] for r in rows] == sorted((r[key] for r in expected), reverse=True)


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def first_table(db):
    db.execute("CREATE TABLE first_table (id INT, name VARCHAR(256))")
    return db


@pytest.fixture
def second_table(db):
    db.execute("CREATE TABLE second_table (id INT, name VARCHAR(256), score INT)")
    db.executemany("INSERT INTO second_table VALUES (?, ?, ?)",
                   [(1, "Bob", 14), (2, "Roy", 5), (3, "John", 10), (4, "Bryan", 8)])
    return db


TEMPS = [
    ("San Francisco", "CA", 60), ("San Francisco", "CA", 64),
    ("Los Angeles", "CA", 75), ("Los Angeles", "CA", 81),
    ("Chicago", "IL", 50), ("Chicago", "IL", 41),
    ("Austin", "TX", 90), ("Austin", "TX", 70),
    ("Houston", "TX", 93), ("Houston", "TX", 85),
]


@pytest.fixture
def temperatures(db):
    db.execute("CREATE TABLE temperatures (city VARCHAR(256), state VARCHAR(128), "
               "year INT, month INT, value INT)")
    db.executemany("INSERT INTO temperatures VALUES (?, ?, 2020, 1, ?)", TEMPS)
    return db


GENRES = [(1, "Drama"), (2, "Mystery"), (3, "Adventure"), (4, "Fantasy"),
          (5, "Comedy"), (6, "Crime"), (7, "Thriller")]
SHOWS = [(1, "Breaking Bad"), (2, "Dexter"), (3, "Game of Thrones"), (4, "House"),
         (5, "New Girl"), (6, "Silicon Valley"), (7, "The Big Bang Theory"),
         (8, "Better Call Saul"), (9, "The Last Man on Earth")]
SHOW_GENRES = [(1, 1), (1, 6), (2, 1), (2, 2), (3, 1), (3, 3), (3, 4),
               (4, 1), (4, 2), (5, 5), (6, 5), (7, 5)]
RATINGS = [(1, 9), (1, 10), (2, 8), (2, 6), (3, 10), (3, 9), (3, 8),
           (4, 7), (5, 3), (6, 12), (7, 5), (7, 6), (8, 4)]


@pytest.fixture
def tvshows(db):
    db.execute("CREATE TABLE tv_genres (id INT PRIMARY KEY, name VARCHAR(256))")
    db.execute("CREATE TABLE tv_shows (id INT PRIMARY KEY, title VARCHAR(256))")
    db.execute("CREATE TABLE tv_show_genres (show_id INT, genre_id INT)")
    db.execute("CREATE TABLE tv_show_ratings (show_id INT, rate INT)")
    db.executemany("INSERT INTO tv_genres VALUES (?, ?)", GENRES)
    db.executemany("INSERT INTO tv_shows VALUES (?, ?)", SHOWS)
    db.executemany("INSERT INTO tv_show_genres VALUES (?, ?)", SHOW_GENRES)
    db.executemany("INSERT INTO tv_show_ratings VALUES (?, ?)", RATINGS)
    return db


BANDS = [
    (1, "Iron Maiden", 4195, 1975, "United Kingdom", None, "New wave of british heavy,Heavy"),
    (2, "Metallica", 3712, 1981, "USA", None, "Heavy,Bay area thrash"),
    (3, "Alice Cooper", 1200, 1964, "USA", None, "Glam rock,Shock rock"),
    (4, "Motley Crue", 1500, 1981, "USA", 2015, "Glam rock,Heavy"),
    (5, "Def Leppard", 900, 1977, "United Kingdom", None, "Hard rock,Glam rock"),
    (6, "Amon Amarth", 3054, 1988, "Sweden", None, "Melodic death"),
    (7, "Hanoi Rocks", 300, 1979, "Finland", 1985, "Glam rock,Glam punk"),
]


@pytest.fixture
def metal_bands(db):
    db.execute("CREATE TABLE metal_bands (id INT, band_name VARCHAR(255), fans INT, "
               "formed INT, origin VARCHAR(255), split INT, style VARCHAR(255))")
    db.executemany("INSERT INTO metal_bands VALUES (?, ?, ?, ?, ?, ?, ?)", BANDS)
    return db


def test_1_first_table(db):
    run_sqlite(db, "1-first_table.sql")
    run_sqlite(db, "1-first_table.sql")   # must not fail if it already exists
    cols = [r[1] for r in db.execute("PRAGMA table_info(first_table)")]
    assert cols == ["id", "name"]
    text = task_path("1-first_table.sql").read_text().upper()
    assert "SELECT" not in text and "SHOW" not in text


def test_2_list_values(first_table):
    first_table.executemany("INSERT INTO first_table VALUES (?, ?)", [(1, "a"), (2, "b")])
    cols, rows = run_sqlite(first_table, "2-list_values.sql")
    assert cols == ["id", "name"]
    assert sorted(rows) == [(1, "a"), (2, "b")]


def test_3_insert_value(first_table):
    run_sqlite(first_table, "3-insert_value.sql")
    assert first_table.execute("SELECT id, name FROM first_table").fetchall() == \
        [(89, "Holberton School")]


def test_4_best_score(second_table):
    cols, rows = run_sqlite(second_table, "4-best_score.sql")
    assert cols == ["score", "name"]
    assert rows == [(14, "Bob"), (10, "John")]


def test_5_average(second_table):
    cols, rows = run_sqlite(second_table, "5-average.sql")
    assert cols == ["average"]
    assert rows[0][0] == pytest.approx(9.25)


def test_6_avg_temperatures(temperatures):
    cols, rows = run_sqlite(temperatures, "6-avg_temperatures.sql")
    assert cols == ["city", "avg_temp"]
    assert [r[0] for r in rows] == ["Houston", "Austin", "Los Angeles",
                                    "San Francisco", "Chicago"]
    assert [r[1] for r in rows] == pytest.approx([89, 80, 78, 62, 45.5])


def test_7_max_state(temperatures):
    cols, rows = run_sqlite(temperatures, "7-max_state.sql")
    assert cols == ["state", "max_temp"]
    assert rows == [("CA", 81), ("IL", 50), ("TX", 93)]


def test_8_genre_id_by_show(tvshows):
    cols, rows = run_sqlite(tvshows, "8-genre_id_by_show.sql")
    assert cols == ["title", "genre_id"]
    titles = dict(SHOWS)
    expected = sorted((titles[s], g) for s, g in SHOW_GENRES)
    assert rows == expected


def test_9_no_genre(tvshows):
    cols, rows = run_sqlite(tvshows, "9-no_genre.sql")
    assert cols == ["title", "genre_id"]
    assert rows == [("Better Call Saul", None), ("The Last Man on Earth", None)]


def test_10_count_shows_by_genre(tvshows):
    cols, rows = run_sqlite(tvshows, "10-count_shows_by_genre.sql")
    assert cols == ["genre", "number_of_shows"]
    expected = [("Drama", 4), ("Comedy", 3), ("Mystery", 2),
                ("Adventure", 1), ("Fantasy", 1), ("Crime", 1)]
    check_sorted_desc(rows, expected)   # Thriller has no show: not listed


def test_11_rating_shows(tvshows):
    cols, rows = run_sqlite(tvshows, "11-rating_shows.sql")
    assert cols == ["title", "rating"]
    expected = [("Game of Thrones", 27), ("Breaking Bad", 19), ("Dexter", 14),
                ("Silicon Valley", 12), ("The Big Bang Theory", 11), ("House", 7),
                ("Better Call Saul", 4), ("New Girl", 3)]
    check_sorted_desc(rows, expected)


def test_12_rating_genres(tvshows):
    cols, rows = run_sqlite(tvshows, "12-rating_genres.sql")
    assert cols == ["name", "rating"]
    expected = [("Drama", 67), ("Mystery", 21), ("Comedy", 26), ("Crime", 19),
                ("Adventure", 27), ("Fantasy", 27)]
    check_sorted_desc(rows, expected)


def test_15_fans(metal_bands):
    cols, rows = run_sqlite(metal_bands, "15-fans.sql")
    assert cols == ["origin", "nb_fans"]
    expected = [("USA", 6412), ("United Kingdom", 5095), ("Sweden", 3054), ("Finland", 300)]
    assert rows == expected


def test_16_glam_rock(metal_bands):
    cols, rows = run_sqlite(metal_bands, "16-glam_rock.sql")
    assert cols == ["band_name", "lifespan"]
    assert rows == [("Alice Cooper", 56), ("Def Leppard", 43),
                    ("Motley Crue", 34), ("Hanoi Rocks", 6)]


# =====================================================================
# MySQL part (needs a running server, see README)
# =====================================================================

MYSQL_DB = "ml_curriculum_test"


def _mysql_args(database=None):
    args = ["mysql", "-h", os.environ.get("MYSQL_HOST", "127.0.0.1"),
            "-P", os.environ.get("MYSQL_PORT", "3306"),
            "-u", os.environ.get("MYSQL_USER", "root"), "--batch", "--raw"]
    if database:
        args.append(database)
    return args


def mysql(sql, database=MYSQL_DB, check=True):
    """Run SQL text through the mysql client; return a list of output rows (tab-split)."""
    proc = subprocess.run(_mysql_args(database), input=sql, capture_output=True,
                          text=True, timeout=60)
    if check and proc.returncode != 0:
        raise AssertionError(proc.stderr)
    return proc


def rows_of(proc):
    lines = [l for l in proc.stdout.splitlines() if l]
    return [tuple(l.split("\t")) for l in lines[1:]]   # drop the header line


@pytest.fixture
def mysql_db():
    if shutil.which("mysql") is None:
        pytest.skip("needs the mysql client and a MySQL server (see README)")
    try:
        ok = subprocess.run(_mysql_args() + ["-e", "SELECT 1"], capture_output=True,
                            timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    if not ok:
        pytest.skip("no MySQL server reachable (see README for a Docker one-liner)")
    mysql(f"DROP DATABASE IF EXISTS {MYSQL_DB}; CREATE DATABASE {MYSQL_DB};", database=None)
    yield MYSQL_DB
    mysql(f"DROP DATABASE IF EXISTS {MYSQL_DB};", database=None)


def test_0_create_database(mysql_db):
    sql = task_path("0-create_database_if_missing.sql").read_text()
    assert "SELECT" not in sql.upper() and "SHOW" not in sql.upper()
    mysql("DROP DATABASE IF EXISTS db_0;", database=None)
    mysql(sql, database=None)
    mysql(sql, database=None)   # second run must not fail
    out = mysql("SHOW DATABASES LIKE 'db_0';", database=None).stdout
    assert "db_0" in out
    mysql("DROP DATABASE IF EXISTS db_0;", database=None)


def test_13_uniq_users(mysql_db):
    sql = task_path("13-uniq_users.sql").read_text()
    mysql(sql)
    mysql(sql)   # must not fail if it exists
    mysql("INSERT INTO users (email, name) VALUES ('bob@dylan.com', 'Bob');"
          "INSERT INTO users (email, name) VALUES ('sylvie@dylan.com', 'Sylvie');")
    dup = mysql("INSERT INTO users (email, name) VALUES ('bob@dylan.com', 'Jean');",
                check=False)
    assert dup.returncode != 0, "email must be unique"
    assert rows_of(mysql("SELECT id, email FROM users ORDER BY id;")) == \
        [("1", "bob@dylan.com"), ("2", "sylvie@dylan.com")]
    nulls = mysql("INSERT INTO users (name) VALUES ('NoEmail');", check=False)
    assert nulls.returncode != 0, "email must be NOT NULL"


def test_14_country_users(mysql_db):
    mysql(task_path("14-country_users.sql").read_text())
    mysql("INSERT INTO users (email, name) VALUES ('bob@dylan.com', 'Bob');"
          "INSERT INTO users (email, name, country) VALUES ('s@dylan.com', 'S', 'CO');")
    bad = mysql("INSERT INTO users (email, name, country) VALUES ('j@d.com', 'J', 'FR');",
                check=False)
    assert bad.returncode != 0, "only US, CO and TN are allowed"
    assert rows_of(mysql("SELECT email, country FROM users ORDER BY id;")) == \
        [("bob@dylan.com", "US"), ("s@dylan.com", "CO")]


def test_17_store_trigger(mysql_db):
    mysql((DATA / "17-init.sql").read_text())
    mysql(task_path("17-store.sql").read_text())
    mysql("INSERT INTO orders (item_name, number) VALUES ('apple', 1);"
          "INSERT INTO orders (item_name, number) VALUES ('apple', 3);"
          "INSERT INTO orders (item_name, number) VALUES ('pear', 2);")
    assert rows_of(mysql("SELECT name, quantity FROM items ORDER BY name;")) == \
        [("apple", "6"), ("pear", "8"), ("pineapple", "10")]


def test_18_valid_email_trigger(mysql_db):
    mysql((DATA / "18-init.sql").read_text())
    mysql(task_path("18-valid_email.sql").read_text())
    mysql("UPDATE users SET valid_email = 1 WHERE email = 'bob@dylan.com';"
          "UPDATE users SET email = 'sylvie+new@dylan.com' WHERE email = 'sylvie@dylan.com';"
          "UPDATE users SET name = 'Jannis' WHERE email = 'jeanne@dylan.com';"
          "UPDATE users SET email = 'bob@dylan.com' WHERE email = 'bob@dylan.com';")
    assert rows_of(mysql("SELECT email, valid_email FROM users ORDER BY id;")) == \
        [("bob@dylan.com", "1"), ("sylvie+new@dylan.com", "0"), ("jeanne@dylan.com", "1")]


def test_19_add_bonus_procedure(mysql_db):
    mysql((DATA / "19-init.sql").read_text())
    mysql(task_path("19-bonus.sql").read_text())
    mysql("CALL AddBonus((SELECT id FROM users WHERE name = 'Jeanne'), 'Python is cool', 100);"
          "CALL AddBonus((SELECT id FROM users WHERE name = 'Jeanne'), 'Bonus project', 100);"
          "CALL AddBonus((SELECT id FROM users WHERE name = 'Bob'), 'Bonus project', 10);"
          "CALL AddBonus((SELECT id FROM users WHERE name = 'Jeanne'), 'New bonus', 90);")
    projects = [r[0] for r in rows_of(mysql("SELECT name FROM projects ORDER BY id;"))]
    assert projects == ["C is fun", "Python is cool", "Bonus project", "New bonus"]
    assert rows_of(mysql("SELECT COUNT(*) FROM corrections;")) == [("8",)]


def test_20_average_score_procedure(mysql_db):
    mysql((DATA / "20-init.sql").read_text())
    mysql(task_path("20-average_score.sql").read_text())
    mysql("CALL ComputeAverageScoreForUser((SELECT id FROM users WHERE name = 'Jeanne'));")
    rows = dict(rows_of(mysql("SELECT name, average_score FROM users;")))
    assert float(rows["Jeanne"]) == pytest.approx(82)
    assert float(rows["Bob"]) == 0


def test_21_safe_div_function(mysql_db):
    mysql((DATA / "21-init.sql").read_text())
    mysql(task_path("21-div.sql").read_text())
    rows = rows_of(mysql("SELECT SafeDiv(a, b) FROM numbers;"))
    assert [float(r[0]) for r in rows] == pytest.approx([5, 0.8, 0.666667, 2, 0, 0.75],
                                                        abs=1e-4)


# =====================================================================
# MongoDB: pymongo functions with mongomock
# =====================================================================

def load(filename, attr):
    from helpers import load_task
    return load_task(__file__, filename, attr)


@pytest.fixture
def school():
    mongomock = pytest.importorskip("mongomock")
    return mongomock.MongoClient().my_db.school


SCHOOLS = [
    {"name": "Holberton school", "topics": ["Algo", "C", "Python", "React"]},
    {"name": "UCSF", "topics": ["Algo", "MongoDB"]},
    {"name": "UCLA", "topics": ["C", "Python"]},
    {"name": "UCSD", "topics": ["Cassandra"]},
    {"name": "Stanford", "topics": ["C", "React", "Javascript"]},
]


def test_30_list_all(school):
    list_all = load("30-all.py", "list_all")
    assert list_all(school) == []
    school.insert_many([{"name": "Holberton school"}, {"name": "UCSF"}])
    docs = list_all(school)
    assert isinstance(docs, list)
    assert sorted(d["name"] for d in docs) == ["Holberton school", "UCSF"]


def test_31_insert_school(school):
    insert_school = load("31-insert_school.py", "insert_school")
    new_id = insert_school(school, name="UCSF", address="505 Parnassus Ave")
    doc = school.find_one({"_id": new_id})
    assert doc["name"] == "UCSF" and doc["address"] == "505 Parnassus Ave"


def test_32_update_topics(school):
    update_topics = load("32-update_topics.py", "update_topics")
    school.insert_many([{"name": "Holberton school"}, {"name": "Holberton school"},
                        {"name": "UCSF"}])
    update_topics(school, "Holberton school", ["Sys admin", "AI", "Algorithm"])
    update_topics(school, "Holberton school", ["iOS"])
    docs = list(school.find({"name": "Holberton school"}))
    assert len(docs) == 2 and all(d["topics"] == ["iOS"] for d in docs)
    assert "topics" not in school.find_one({"name": "UCSF"})


def test_33_schools_by_topic(school):
    schools_by_topic = load("33-schools_by_topic.py", "schools_by_topic")
    school.insert_many([dict(s) for s in SCHOOLS])
    names = sorted(d["name"] for d in schools_by_topic(school, "Python"))
    assert names == ["Holberton school", "UCLA"]
    assert schools_by_topic(school, "Rust") == []


def test_34_log_stats(capsys):
    mongomock = pytest.importorskip("mongomock")
    log_stats = load("34-log_stats.py", "log_stats")
    logs = mongomock.MongoClient().logs.nginx
    docs = ([{"method": "GET", "path": "/status"}] * 3 + [{"method": "GET", "path": "/"}] * 4
            + [{"method": "POST", "path": "/"}] * 2 + [{"method": "DELETE", "path": "/x"}]
            + [{"method": "GET", "path": "/status/other"}])
    logs.insert_many([dict(d) for d in docs])
    log_stats(logs)
    assert capsys.readouterr().out == (
        "11 logs\n"
        "Methods:\n"
        "\tmethod GET: 8\n"
        "\tmethod POST: 2\n"
        "\tmethod PUT: 0\n"
        "\tmethod PATCH: 0\n"
        "\tmethod DELETE: 1\n"
        "3 status check\n"
    )


# =====================================================================
# MongoDB: mongosh scripts (need a real server)
# =====================================================================

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB = "ml_curriculum_test"


@pytest.fixture
def mongo():
    if shutil.which("mongosh") is None:
        pytest.skip("needs mongosh and a MongoDB server (see README)")
    pymongo = pytest.importorskip("pymongo")
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("no MongoDB server reachable (see README for a Docker one-liner)")
    client.drop_database(MONGO_DB)
    yield client[MONGO_DB]
    client.drop_database(MONGO_DB)


def mongosh(name, database=MONGO_DB):
    script = task_path(name).read_text()
    base = MONGO_URI.rstrip("/")
    proc = subprocess.run(["mongosh", "--quiet", f"{base}/{database}"], input=script,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_22_list_databases(mongo):
    assert "admin" in mongosh("22-list_databases")


def test_23_use_or_create_database(mongo):
    assert "my_db" in mongosh("23-use_or_create_database")


def test_24_insert(mongo):
    mongosh("24-insert")
    docs = list(mongo.school.find({}, {"_id": 0}))
    assert docs == [{"name": "Holberton school"}]


def test_25_all(mongo):
    mongo.school.insert_many([{"name": "Holberton school"}, {"name": "UCSF"}])
    out = mongosh("25-all")
    assert "Holberton school" in out and "UCSF" in out


def test_26_match(mongo):
    mongo.school.insert_many([{"name": "Holberton school"}, {"name": "UCSF"}])
    out = mongosh("26-match")
    assert "Holberton school" in out and "UCSF" not in out


def test_27_count(mongo):
    mongo.school.insert_many([{"name": "a"}, {"name": "b"}, {"name": "c"}])
    assert mongosh("27-count").strip().splitlines()[-1].strip() == "3"


def test_28_update(mongo):
    mongo.school.insert_many([{"name": "Holberton school"}, {"name": "Holberton school"},
                              {"name": "UCSF"}])
    mongosh("28-update")
    assert mongo.school.count_documents({"address": "972 Mission street"}) == 2
    assert "address" not in mongo.school.find_one({"name": "UCSF"})


def test_29_delete(mongo):
    mongo.school.insert_many([{"name": "Holberton school"}, {"name": "Holberton school"},
                              {"name": "UCSF"}])
    mongosh("29-delete")
    assert mongo.school.count_documents({}) == 1
    assert mongo.school.find_one()["name"] == "UCSF"
