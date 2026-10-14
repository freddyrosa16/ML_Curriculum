# Databases

**Status:** Original (the MongoDB shell tasks now use `mongosh`, which replaced the old `mongo` shell, and its current commands).

## Why this matters
Training data and model outputs usually live in a database, not a CSV. Being able to ask a database the right question (filter, join, group, aggregate) means you pull exactly the data you need instead of loading everything into pandas. You'll learn the two big families: **relational** databases with SQL (MySQL here), where data lives in tables with fixed columns, and **document** databases (MongoDB), where data lives in flexible JSON-like documents. Then you'll drive MongoDB from Python.

## Concepts to understand first
- **What is a relational database?** A set of tables. Each table has named, typed columns; each row is one record. Tables are linked by IDs: a `tv_show_genres` row says "show 3 has genre 1".
- **What is SQL?** The language for asking relational databases questions. The core shape is `SELECT columns FROM table WHERE condition GROUP BY ... ORDER BY ...`. It's *declarative*: you describe the result, the database figures out how.
- **What are primary keys, `UNIQUE`, `NOT NULL`, `AUTO_INCREMENT`?** Constraints the database enforces for you: a primary key identifies each row, `UNIQUE` forbids duplicates, `NOT NULL` forbids missing values, `AUTO_INCREMENT` hands out the next ID automatically.
- **What is a JOIN?** Combining rows of two tables that match on some column. An *inner* join keeps only matching pairs; a *left* join keeps every row of the left table, filling `NULL` when there's no match (handy for "which shows have no genre?").
- **What are aggregate functions and `GROUP BY`?** `COUNT`, `SUM`, `AVG`, `MAX`... collapse many rows into one value. `GROUP BY city` does that separately for each city. An alias (`AVG(value) AS avg_temp`) names the result column.
- **What are triggers, stored procedures and functions?** Code that lives *inside* the database. A trigger runs automatically when rows are inserted/updated; a procedure is a named routine you `CALL`; a function returns a value you can use inside a query. They keep data consistent even if several apps write to it.
- **What is NoSQL / MongoDB?** MongoDB stores *documents* (like Python dicts, saved as BSON) in *collections*, without a fixed schema. Queries are themselves documents: `{name: "UCSF"}` matches documents whose `name` is "UCSF"; `{topics: "Python"}` matches documents whose `topics` list contains "Python".
- **What is pymongo?** The official Python driver for MongoDB: a collection object has methods like `find`, `insert_one`, `update_many`, `count_documents`.

## Resources
- [SQLBolt: interactive SQL lessons (free)](https://sqlbolt.com/)
- [MySQL 8.4 Reference Manual: tutorial](https://dev.mysql.com/doc/refman/8.4/en/tutorial.html)
- [MySQL: CREATE TRIGGER](https://dev.mysql.com/doc/refman/8.4/en/create-trigger.html) · [CREATE PROCEDURE and CREATE FUNCTION](https://dev.mysql.com/doc/refman/8.4/en/create-procedure.html)
- [Mode SQL tutorial (joins, aggregates)](https://mode.com/sql-tutorial/)
- [MongoDB: CRUD operations](https://www.mongodb.com/docs/manual/crud/)
- [mongosh documentation](https://www.mongodb.com/docs/mongodb-shell/)
- [PyMongo tutorial](https://pymongo.readthedocs.io/en/stable/tutorial.html)

## Rules
- SQL files: one script per task, ending each statement with `;`. Write standard SQL where you can; the plain query tasks (1-12, 15, 16) are tested on SQLite, so avoid MySQL-only syntax in those. All keywords in uppercase is a good habit.
- Unless a task says otherwise, a script must not fail if run twice (`IF NOT EXISTS`).
- Tasks 22-29: plain mongosh scripts (files with no extension), run by piping them into `mongosh`.
- Tasks 30-34: Python, `import pymongo` / `from pymongo import MongoClient` only when you need it. The functions receive a pymongo collection object.

## Setup
**Without any server** you can do and test tasks 1-12, 15, 16 (SQLite, built into Python) and 30-34 (`uv pip install pymongo mongomock`; mongomock is an in-memory fake of MongoDB used by the tests).

**For the MySQL-only tasks (0, 13, 14, 17-21)** you need a MySQL server and the `mysql` command-line client. The easiest way is Docker:
```bash
docker run -d --name ml-mysql -e MYSQL_ALLOW_EMPTY_PASSWORD=yes -p 3306:3306 mysql:8.4 --log-bin-trust-function-creators=1
brew install mysql-client      # then add "$(brew --prefix mysql-client)/bin" to your PATH
```
The tests connect as `root` with no password on `127.0.0.1:3306` (override with the `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PWD` environment variables) and use a throwaway database called `ml_curriculum_test`. Run a script by hand with `mysql -h 127.0.0.1 -u root <database> < your_file.sql`.

**For the mongosh tasks (22-29)** you need a MongoDB server and `mongosh`:
```bash
docker run -d --name ml-mongo -p 27017:27017 mongo:8
brew install mongosh
```
Tests use `mongodb://127.0.0.1:27017` (override with `MONGO_URI`) and a throwaway database `ml_curriculum_test`. Without a server those tests are skipped, not failed.

## Data
The `data/` folder has the tables used by the original course, as MySQL dumps and setup scripts. They're data, not solutions:
- `second_table.sql`: a `second_table (id, name, score)` with 4 students (tasks 4-5).
- `temperatures.sql`: monthly temperatures by `city`, `state`, `year`, `month`, `value` in °F (tasks 6-7). Load it into a database named `hbtn_0c_0`.
- `tvshows.sql`: tables `tv_shows (id, title)`, `tv_genres (id, name)`, `tv_show_genres (show_id, genre_id)` (tasks 8-10, database `hbtn_0d_tvshows`).
- `tvshows_rate.sql`: the same plus `tv_show_ratings (show_id, rate)` (tasks 11-12, database `hbtn_0d_tvshows_rate`).
- `metal_bands.sql`: `metal_bands (id, band_name, fans, formed, origin, split, style)` (tasks 15-16).
- `17-init.sql` ... `21-init.sql`: the starting tables for tasks 17-21, described in each task below.

To load a dump: create the database, then feed the file to it, e.g. `mysql -h 127.0.0.1 -u root -e "CREATE DATABASE hbtn_0c_0"` and `mysql -h 127.0.0.1 -u root hbtn_0c_0 < data/temperatures.sql`. The tests don't need these files for the SQLite tasks; they build small tables of their own with the same columns.

## Tasks: SQL basics

### 0. Create a database (MySQL)
**File:** `0-create_database_if_missing.sql`

Create the database `db_0`. Running the script when `db_0` already exists must not fail. You may not use `SELECT` or `SHOW`.
```
mysql -h 127.0.0.1 -u root < 0-create_database_if_missing.sql   (twice: no error)
SHOW DATABASES;   ->  ... db_0 ...
```

### 1. First table
**File:** `1-first_table.sql`

Create a table `first_table` in the current database with columns `id` (INT) and `name` (VARCHAR(256)). Must not fail if it already exists. No `SELECT` or `SHOW`.

### 2. List all in table
**File:** `2-list_values.sql`

List every row (all columns) of `first_table`.
```
id    name
89    Holberton School     (after task 3)
```

### 3. First add
**File:** `3-insert_value.sql`

Insert a row into `first_table` with `id` 89 and `name` "Holberton School".

### 4. Select the best
**File:** `4-best_score.sql`

From `second_table`, list the `score` and `name` (in that order) of every record with a score of at least 10, highest score first.
```
score   name
14      Bob
10      John
```

### 5. Average
**File:** `5-average.sql`

Compute the average `score` of all records in `second_table`. Name the result column `average`.
```
average
9.25
```

### 6. Temperatures #0
**File:** `6-avg_temperatures.sql`

From `temperatures`, show each city's average temperature, highest first. Columns: `city`, `avg_temp`.
```
city             avg_temp
Chandler         72.8627
Gilbert          71.8088
...
```

### 7. Temperatures #1
**File:** `7-max_state.sql`

Show the maximum temperature of each state, ordered by state name. Columns: `state`, `max_temp`.
```
state   max_temp
AZ      110
CA      110
...
```

### 8. Genre ID by show
**File:** `8-genre_id_by_show.sql`

List every show that has at least one genre linked: one row per (show, genre) pair, showing `tv_shows.title` and `tv_show_genres.genre_id`, sorted by title then genre_id, both ascending. One `SELECT` statement only.
```
title              genre_id
Breaking Bad       1
Breaking Bad       6
...
```

### 9. No genre
**File:** `9-no_genre.sql`

List every show with **no** genre linked, showing `tv_shows.title` and `tv_show_genres.genre_id` (which will be `NULL`), sorted by title then genre_id. One `SELECT` only.
```
title                    genre_id
Better Call Saul         NULL
Homeland                 NULL
```

### 10. Number of shows by genre
**File:** `10-count_shows_by_genre.sql`

For every genre that has at least one show, show its name and how many shows are linked to it, most shows first. Columns: `genre`, `number_of_shows`. One `SELECT` only.
```
genre       number_of_shows
Drama       5
Comedy      4
...
```

### 11. Rotten tomatoes
**File:** `11-rating_shows.sql`

List shows by the **sum** of their ratings (from `tv_show_ratings`), highest first. Columns: `title`, `rating`. One `SELECT` only.

### 12. Best genre
**File:** `12-rating_genres.sql`

List genres by the sum of the ratings of all their shows, highest first. Columns: `name`, `rating`. One `SELECT` only. (You'll need to join three tables.)

## Tasks: MySQL advanced

### 13. We are all unique! (MySQL)
**File:** `13-uniq_users.sql`

Create a table `users` with: `id` (integer, never null, auto-increment, primary key), `email` (string of 255 characters, never null, unique) and `name` (string of 255 characters). Must not fail if the table exists. Putting the uniqueness rule in the schema means the database itself refuses duplicates, however buggy the application is.
```
INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Bob");     ->  ok, id 1
INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Jean");    ->  ERROR 1062: Duplicate entry
```

### 14. In and not out (MySQL)
**File:** `14-country_users.sql`

Same `users` table as task 13, plus `country`: an enumeration of `US`, `CO` and `TN`, never null, defaulting to `US`.
```
INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Bob");                    ->  country US
INSERT INTO users (email, name, country) VALUES ("j@d.com", "J", "FR");            ->  ERROR (not in the list)
```

### 15. Best band ever!
**File:** `15-fans.sql`

Rank the countries of origin of bands by their total number of fans. Columns: `origin`, `nb_fans`, highest first.
```
origin            nb_fans
USA               99349
Sweden            47169
...
```

### 16. Old school band
**File:** `16-glam_rock.sql`

List the bands whose `style` contains "Glam rock", ranked by longevity. Columns: `band_name`, `lifespan`, where lifespan is in years, computed from `formed` and `split` (if the band hasn't split, count until 2020). Longest first.
```
band_name          lifespan
Alice Cooper       56
Mötley Crüe        34
...
```

### 17. Buy buy buy (MySQL)
**File:** `17-store.sql`

Starting tables (`data/17-init.sql`): `items (name, quantity)` with apple, pineapple and pear, each with quantity 10; and an empty `orders (item_name, number)`. Create a **trigger** that decreases an item's quantity by the ordered number every time a row is added to `orders`. Quantities may go negative.
```
insert orders: apple 1, apple 3, pear 2
SELECT * FROM items;   ->  apple 6, pineapple 10, pear 8
```

### 18. Email validation to sent (MySQL)
**File:** `18-valid_email.sql`

Starting table (`data/18-init.sql`): `users (id, email, name, valid_email)`, with `valid_email` a boolean defaulting to 0. Create a trigger that resets `valid_email` to 0 **only when the email actually changes**. Updating other columns, or setting the email to the same value, must not reset it.
```
bob: set valid_email = 1                 ->  1
sylvie: change email                     ->  valid_email 0
jeanne: change name only                 ->  valid_email stays 1
bob: set email to the same address       ->  valid_email stays 1
```

### 19. Add bonus (MySQL)
**File:** `19-bonus.sql`

Starting tables (`data/19-init.sql`): `users (id, name, average_score)`, `projects (id, name)` and `corrections (user_id, project_id, score)`, with Bob and Jeanne each having scores for "C is fun" and "Python is cool". Create a **stored procedure** `AddBonus(user_id, project_name, score)` that adds a correction for that user on that project, creating the project first if no project with that name exists.
```
CALL AddBonus(<Jeanne's id>, "Python is cool", 100);   ->  new correction, no new project
CALL AddBonus(<Jeanne's id>, "Bonus project", 100);    ->  creates project "Bonus project" + correction
SELECT * FROM projects;   ->  C is fun, Python is cool, Bonus project, ...
```

### 20. Average score (MySQL)
**File:** `20-average_score.sql`

Same starting tables (`data/20-init.sql`). Create a stored procedure `ComputeAverageScoreForUser(user_id)` that computes the user's average correction score and stores it in `users.average_score`.
```
CALL ComputeAverageScoreForUser(<Jeanne's id>);
SELECT * FROM users;   ->  Bob 0, Jeanne 82     (Jeanne's scores: 91 and 73)
```

### 21. Safe divide (MySQL)
**File:** `21-div.sql`

Starting table (`data/21-init.sql`): `numbers (a, b)` with the pairs (10, 2), (4, 5), (2, 3), (6, 3), (7, 0), (6, 8). Create a **function** `SafeDiv(a, b)` that returns `a / b`, or 0 when `b` is 0.
```
SELECT a, b, SafeDiv(a, b) FROM numbers;
->  10 2 5 / 4 5 0.8 / 2 3 0.666667 / 6 3 2 / 7 0 0 / 6 8 0.75
```

## Tasks: MongoDB shell (mongosh)
Each file is run as `mongosh <database> < file` (except 22-23, run without a database). Use the current mongosh commands (`insertOne`, `countDocuments`, `updateMany`...) rather than the deprecated `insert`, `count`, `update`.

### 22. List all databases
**File:** `22-list_databases` — list all databases on the server.
```
admin    40.00 KiB
config   72.00 KiB
local    40.00 KiB
```

### 23. Create a database
**File:** `23-use_or_create_database` — switch to the database `my_db` (MongoDB creates it on first write).
```
switched to db my_db
```

### 24. Insert document
**File:** `24-insert` — insert a document with one attribute `name` = "Holberton school" into the collection `school`.

### 25. All documents
**File:** `25-all` — list all documents of `school`.
```
[ { _id: ObjectId('...'), name: 'Holberton school' } ]
```

### 26. All matches
**File:** `26-match` — list the documents of `school` whose `name` is "Holberton school".

### 27. Count
**File:** `27-count` — print the number of documents in `school`.
```
1
```

### 28. Update
**File:** `28-update` — add the attribute `address` = "972 Mission street" to **every** document of `school` named "Holberton school".

### 29. Delete by match
**File:** `29-delete` — delete every document of `school` named "Holberton school".

## Tasks: MongoDB from Python (pymongo)

### 30. List all documents in Python
**File:** `30-all.py` · **Signature:** `list_all(mongo_collection)`

Return a list of all documents in the collection (an empty list if there are none).
```
list_all(client.my_db.school)   ->  [{'_id': ObjectId('...'), 'name': 'Holberton school'}, ...]
```

### 31. Insert a document in Python
**File:** `31-insert_school.py` · **Signature:** `insert_school(mongo_collection, **kwargs)`

Insert a document built from the keyword arguments and return its new `_id`.
```
insert_school(school, name="UCSF", address="505 Parnassus Ave")   ->  ObjectId('5a8f60cfd4321e1403ba7ab9')
```

### 32. Change school topics
**File:** `32-update_topics.py` · **Signature:** `update_topics(mongo_collection, name, topics)`

Set the `topics` attribute (a list of strings) of **every** document whose `name` matches, replacing any previous topics.
```
update_topics(school, "Holberton school", ["Sys admin", "AI", "Algorithm"])
update_topics(school, "Holberton school", ["iOS"])
->  every "Holberton school" document now has topics ['iOS']
```

### 33. Where can I learn Python?
**File:** `33-schools_by_topic.py` · **Signature:** `schools_by_topic(mongo_collection, topic)`

Return the list of schools whose `topics` contain `topic`.
```
schools: Holberton (Algo, C, Python, React), UCSF (Algo, MongoDB), UCLA (C, Python), UCSD (Cassandra), Stanford (C, React, Javascript)
schools_by_topic(school, "Python")   ->  [Holberton school document, UCLA document]
```

### 34. Log stats
**File:** `34-log_stats.py` · **Signature:** `log_stats(mongo_collection)`, plus a script part

Nginx access logs are stored as documents with (at least) `method` and `path` attributes. `log_stats` prints, exactly in this format:
- first line: `<n> logs`, the number of documents;
- `Methods:`, then one line per method in the order GET, POST, PUT, PATCH, DELETE: a tab, then `method <METHOD>: <count>`;
- last line: `<n> status check`, the number of documents with method GET **and** path `/status`.

When run as a script (`python 34-log_stats.py`, inside `if __name__ == "__main__":`), connect to `mongodb://127.0.0.1:27017` and call `log_stats` on the `nginx` collection of the `logs` database. To try the script, insert some log documents into `logs.nginx` yourself (the original course's 14 MB dump isn't included).
```
log_stats(logs)
->  11 logs
    Methods:
        method GET: 8
        method POST: 2
        method PUT: 0
        method PATCH: 0
        method DELETE: 1
    3 status check
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 07-data-and-deployment/02-databases -rs
```
`-rs` lists the skipped tests and why (usually: no MySQL or MongoDB server running).
