#!/bin/bash

#Here we obtain the most watched Netflix title in our clicks dataset by following the steps below:
#1. We cut the 4th column of the csv file, which contains the title of the movies
#2. We obtain the titles without the header by tailing the first line
#3. We sort the titles alphabetically
#4. We obtain the unique titles and count the number of times they appear
#5. We sort the titles by the number of times they appear in descending order
#6. We obtain the first line of the sorted titles, which is the most watched title
#7. We print the most watched title
echo -n "The most-watched Netflix title is: " && cut -d ',' -f4 data/vodclickstream_uk_movies.csv | tail -n +2 | sort| uniq -c | sort -rn | head -n 1 | awk '{$1=""; print $0}'

#Here we obtain the average time between subsequent clicks on Netflix by following the steps below:
#1. We cut the 3rd column of the csv file, which contains the time between subsequent clicks
#2. We obtain the time without the header by tailing the first line
#3. We obtain the sum of the time between subsequent clicks and we count the number of times we have a time between subsequent clicks
#4. We divide the sum of the time between subsequent clicks by the number of times we have a time between subsequent clicks. This is the average time between subsequent clicks
echo -n "The average time (in seconds) between subsequent clicks on Netflix is: " && cut -d ',' -f3 data/vodclickstream_uk_movies.csv | tail -n +2 |  awk '{sum += $1; count++} END {print sum/count}'

#Here we obtain the ID of the user that has spent the most time on Netflix by following the steps below:
#1. Using the csvcut command, we cut the duration and user_id columns of the csv file
#2. We obtain these columns without the header by tailing the first line
#3. We obtain the sum of the duration for each user_id by storing the duration in an array and summing the duration for each user_id
#4. We sort the user_ids by the sum of the duration in descending order
#5. We obtain the first line of the sorted user_ids, which is the user_id that has spent the most time on Netflix
#6. We print the user_id that has spent the most time on Netflix
echo -n "The ID of the user that has spent the most time on Netflix is: " && csvcut -c duration,user_id data/vodclickstream_uk_movies.csv | tail -n +2 |awk -F',' '{arr[$2]+=$1} END{for(i in arr) print arr[i], i}' | sort -rn | head -n 1| awk '{$1=""; print $0}' 
