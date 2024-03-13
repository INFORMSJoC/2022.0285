# Data for replication

1. /reddit/
   
    reddit_data.json: The id of sarcasm text, the clauses and the annotated labels are provided. The sarcasm text can be collected through [Reddit API](https://www.reddit.com/dev/api/).
   
    --id: the id of the sarcasm text
   
    --clauses: the context of the sarcasm text
   
    --clause_labels: 1 for sarcasm cause clause, 0 for non-cause clause
   
    --subreddit: the subreddit of the sarcasm text
   
    subreddit_dict.pkl: the domain knowledge dictionary of the subreddit

3. /twitter/twitter_data.json: The ids of the sarcasm text and the context are provided. The data can be collected through [Twitter API](https://developer.twitter.com/en/docs/twitter-api).
   
    --tweet_id: the tweet id of the sarcasm text
   
    --clause_tweetid: the tweet id of the context of the sarcasm text
   
    --clause_labels: 1 for sarcasm cause clause, 0 for non-cause clause
   

