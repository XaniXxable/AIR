#!/bin/bash

curl -X POST "http://127.0.0.1:5000/restauratnts/" \
-H "Content-Type: application/json" \
-d '{
  "Filters": [
    {"PetFriendly": true, "FamilyFriendly": false},
    {"PetFriendly": false, "FamilyFriendly": true}
  ],
  "UserInput": "Find me Italian restaurants near the park"
}'
