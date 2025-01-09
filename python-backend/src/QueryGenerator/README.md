# Overview
This Python script processes restaurant data to generate queries and responses in JSON. It utilizes a database of restaurant information and provides functionalities to process and analyze data for use in natural language processing (NLP) applications, particularly for restaurant-related queries.

## Key Functionalities:
1. Data Processing:
    * Generate basic dataset and Enhanced dataset.

2. Query Generation:
    * Create city and category-based queries (e.g., "Where can I find good sushi in New York?").
    * Identify tokens for NLP training, labeling categories like "CUISINE" or "LOC" (location).

3. Data Export:
    * Save processed queries and responses to JSON files. Save progress using checkpoint files for batch processing.