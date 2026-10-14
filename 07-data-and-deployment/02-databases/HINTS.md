# Hints: Databases

Open one hint at a time. Try for a while before opening the next.

## 0-1. Create a database / first table
<details><summary>Hint 1</summary>
Both `CREATE DATABASE` and `CREATE TABLE` accept a clause that turns "already exists" from an error into a no-op.
</details>
<details><summary>Hint 2</summary>
A table definition is the table name followed by a parenthesized, comma-separated list of "column name, column type".
</details>

## 2-3. List all / first add
<details><summary>Hint 1</summary>
There's a shorthand for "all columns" in a `SELECT`. Inserting names the columns, then lists the values in the same order; text values go in quotes.
</details>

## 4-5. Select the best / average
<details><summary>Hint 1</summary>
Filtering rows is the `WHERE` clause; sorting is `ORDER BY`, ascending unless you say otherwise. The order of columns in the output is the order you list them after `SELECT`.
</details>
<details><summary>Hint 2</summary>
An aggregate function computes one value over all rows. `AS` gives the output column a name.
</details>

## 6-7. Temperatures
<details><summary>Hint 1</summary>
"Per city" and "per state" mean grouping. Every column you select is either a grouping column or an aggregate.
</details>
<details><summary>Hint 2</summary>
You can sort by an alias you defined in the `SELECT` list.
</details>

## 8. Genre ID by show
<details><summary>Hint 1</summary>
The title lives in one table and the genre id in another; the two are linked by the show's id. You need a join on that link.
</details>
<details><summary>Hint 2</summary>
`ORDER BY` accepts several columns, separated by commas: the second breaks ties in the first.
</details>

## 9. No genre
<details><summary>Hint 1</summary>
An inner join drops shows with no match. Which kind of join keeps every row of the left table and fills the right side with NULL?
</details>
<details><summary>Hint 2</summary>
After that join, the shows without a genre are exactly the rows where the right table's column is NULL. Testing for NULL uses `IS`, not `=`.
</details>

## 10-12. Counting and summing across tables
<details><summary>Hint 1</summary>
Combine what you did in 6 (group + aggregate) and 8 (join). An inner join already leaves out genres with no shows.
</details>
<details><summary>Hint 2</summary>
For 11, join ratings to shows and sum the rates per show. For 12, the path from a genre to its ratings goes through the show-genre link table: that's two joins.
</details>
<details><summary>Hint 3</summary>
Order by the aggregated alias, descending, and name the columns exactly as the task says.
</details>

## 13-14. Unique users, enum
<details><summary>Hint 1</summary>
All of these rules are column options written right after the column's type: never null, auto increment, primary key, unique, default.
</details>
<details><summary>Hint 2</summary>
MySQL has an `ENUM` type that lists the allowed string values. The default can be any one of them.
</details>

## 15-16. Bands
<details><summary>Hint 1</summary>
15 is task 6 again with a different aggregate. For 16, pattern matching on text uses `LIKE` with `%` as a wildcard.
</details>
<details><summary>Hint 2</summary>
For the lifespan, you need "split, or 2020 if split is missing", minus formed. SQL has functions that return the first non-NULL of their arguments (or a `CASE` expression does the same).
</details>

## 17. Buy buy buy
<details><summary>Hint 1</summary>
A trigger has a name, a timing (before/after), an event (insert/update/delete), a table, and a body that runs for each row. Read the MySQL `CREATE TRIGGER` page.
</details>
<details><summary>Hint 2</summary>
Inside the body, `NEW` refers to the row being inserted, so you can read the ordered item name and number from it and update the matching item.
</details>

## 18. Email validation
<details><summary>Hint 1</summary>
This trigger must change the row *before* it's written, so it runs before update. Inside it you have both `OLD` (the row as it was) and `NEW` (the row as it will be).
</details>
<details><summary>Hint 2</summary>
A body with an `IF` needs `BEGIN ... END`. Because the body contains semicolons, the mysql client needs a temporary different statement delimiter while you define it (look up `DELIMITER`), restored afterwards.
</details>

## 19. Add bonus
<details><summary>Hint 1</summary>
A procedure declares its parameters with a direction (`IN`) and a type. Inside, you can declare local variables.
</details>
<details><summary>Hint 2</summary>
Try to look up the project's id into a variable. If nothing was found the variable is still NULL: insert the project, and get its new id with the function that returns the last auto-increment value. Then insert the correction.
</details>
<details><summary>Hint 3</summary>
Same `DELIMITER` trick as task 18.
</details>

## 20. Average score
<details><summary>Hint 1</summary>
It's a single `UPDATE` whose new value comes from a subquery that averages the user's scores.
</details>

## 21. Safe divide
<details><summary>Hint 1</summary>
A function declares its return type and must be marked `DETERMINISTIC` (same inputs, same output), otherwise MySQL may refuse to create it.
</details>
<details><summary>Hint 2</summary>
Branch on whether the divisor is zero and `RETURN` the right value in each branch.
</details>

## 22-29. mongosh
<details><summary>Hint 1</summary>
22 and 23 are shell helpers (one starts with `show`, the other with `use`). The rest are methods on `db.<collection>`.
</details>
<details><summary>Hint 2</summary>
Every query is a document: an empty one matches everything, `{field: value}` matches equality. Updates take a filter and an update document using the `$set` operator. Look at the mongosh CRUD page for the insert/count/update/delete method names that work on one vs. many documents.
</details>

## 30-33. pymongo
<details><summary>Hint 1</summary>
The pymongo collection methods are named like the mongosh ones, in snake_case. `find` returns a cursor, not a list: convert it.
</details>
<details><summary>Hint 2</summary>
`**kwargs` is already a dictionary: it can be inserted as the document. The insert result object carries the new id.
</details>
<details><summary>Hint 3</summary>
Matching "the list contains this value" in MongoDB needs no special operator: an equality filter on an array field matches if any element equals the value.
</details>

## 34. Log stats
<details><summary>Hint 1</summary>
Every number in the output is one `count_documents` call with a different filter. The status-check filter has two conditions in one document.
</details>
<details><summary>Hint 2</summary>
Loop over the five methods in the given order. The tab is the escape sequence `\t` at the start of the line. Compare your output to the example character by character.
</details>
