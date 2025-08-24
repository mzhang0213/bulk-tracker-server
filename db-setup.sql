CREATE TABLE foods --standardized to per g
(
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(100) NOT NULL,
    category_id INTEGER REFERENCES food_categories(id),
    calories NUMERIC(5, 2),
    carbs    NUMERIC(5, 2),
    protein  NUMERIC(5, 2),
    fat      NUMERIC(5, 2)
);

CREATE TABLE food_categories --but idk if im gonna use ts
(
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(50) NOT NULL,
    color VARCHAR(7) -- ui grouping ez
);

CREATE TABLE meals
(
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(50) NOT NULL,
    color VARCHAR(7) CHECK (color ~ '^#[0-9A-Fa-f]{6}$') --hex
    );

CREATE TABLE meal_foods
(
    id       SERIAL PRIMARY KEY,
    meal_id  INTEGER       NOT NULL REFERENCES meals (id) ON DELETE CASCADE,
    food_id  INTEGER       NOT NULL REFERENCES foods (id) ON DELETE CASCADE,
    quantity NUMERIC(8, 2) NOT NULL --max 999999.99 grams
);

CREATE TABLE meal_plans --like diets
(
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(25),
    start DATE --use for day tracking
);


CREATE TABLE meal_plans_items
(
    id           SERIAL PRIMARY KEY,
    meal_plan_id INTEGER NOT NULL REFERENCES meal_plans (id) ON DELETE CASCADE,
    meal_id      INTEGER NOT NULL REFERENCES meals (id) ON DELETE CASCADE,
    day          SERIAL  NOT NULL, --sequential ordered increasing
    UNIQUE (meal_plan_id, day, meal_id)
);

CREATE TABLE muscles
(
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(100) NOT NULL UNIQUE,
    image VARCHAR(500) NOT NULL
);

CREATE TABLE workouts
(
    id               SERIAL PRIMARY KEY,
    name             VARCHAR(100) NOT NULL,
    video_url        VARCHAR(500) NOT NULL,
    description      TEXT,
    duration_minutes INTEGER
);

CREATE TABLE workout_plans
(
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(200) NOT NULL,
    description TEXT
);

CREATE TABLE workout_plan_exercises
(
    id              INTEGER PRIMARY KEY,
    workout_plan_id INTEGER NOT NULL REFERENCES workout_plans (id) ON DELETE CASCADE,
    workout_id      INTEGER NOT NULL REFERENCES workouts (id) ON DELETE CASCADE,
    order_position  INTEGER NOT NULL,
    sets            INTEGER DEFAULT 1,
    reps            INTEGER,
    rest_seconds    INTEGER,
    notes           TEXT,
    UNIQUE (workout_plan_id, order_position)
);

CREATE TABLE workout_muscles
( --every workout has multiple muscles it trains
    id         INTEGER PRIMARY KEY,
    workout_id INTEGER NOT NULL,
    muscle_id  INTEGER NOT NULL,
    FOREIGN KEY (workout_id) REFERENCES workouts (id) ON DELETE CASCADE,
    FOREIGN KEY (muscle_id) REFERENCES muscles (id) ON DELETE CASCADE,
    UNIQUE (workout_id, muscle_id)
);