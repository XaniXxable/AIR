# Overview
FastDineAPI is a FastAPI-based backend system designed to provide restaurant recommendations based on user input. It uses a recommendation system to rank and return a list of restaurants with relevant details such as location, type, ratings, and reviews.


# Features
* Restaurant Recommendations: Fetch top restaurant recommendations based on user input.


# Installation

## Prerequisites
  * **Python**: Ensure you have Python 3.10.
  * **Poetry**: For managing dependencies and running the application.


## Steps
1. Clone the repository:
  ```bash
  git clone <repository-url>
  cd FastDineAPI
  ```
2. Install dependencies:
  ```bash
  poetry install
  ```

3. Run the application:
  ```bash
  poetry run start
  ```

# API Endpoints
1. `POST /restaurants/`
  * Description: Fetches recommended restaurants based on user input.
  * Request Body:
  ```json
  {
    "UserInput": "string"
  }
  ```
* Response:
```json
{
  "Restaurants": [
    {
      "Name": "string",
      "Type": "string",
      "Location": "string",
      "Starts": 5,
      "Score": 9.0,
      "Reviews": 100,
      "Image": "assets/images/restaurant-types/default.jpg"
    }
  ]
}
```
* Error Handling: Returns `{"Error": "Parsing error"}` for invalid input or server-side issues.
