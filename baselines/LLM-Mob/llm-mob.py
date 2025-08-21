import os
import pickle
import time
import ast
import logging
from datetime import datetime
import pandas as pd
from openai import OpenAI

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())

# Deprecated since 1.x version of OpanAI API
# openai.api_key  = os.getenv('OPENAI_API_KEY')
# openai.api_key  = "YOUR OPENAI API KEY HERE"


# Helper function
def get_chat_completion(client, prompt, model="gpt-4o-mini", json_mode=True, max_tokens=None):
    """
    args:
        client: the openai client object (new in 1.x version)
        prompt: the prompt to be completed
        model: specify the model to use
        json_mode: whether return the response in json format (new in 1.x version)
    """
    messages = [{"role": "user", "content": prompt}]
    if json_mode:
        completion = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0,  # the degree of randomness of the model's output
            max_tokens=max_tokens  # the maximum number of tokens to generate
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens
        )
    # res_content = response.choices[0].message["content"]
    # token_usage = response.usage
    return completion


def get_dataset(dataname, sub_dataname=None):
    """
    Loads the dataset files.
    For the 'fsq' dataset, a 'sub_dataname' ('nyc' or 'tky') must be provided
    to specify the path.
    """
    if dataname == 'fsq':
        if sub_dataname not in ['nyc', 'tky']:
            raise ValueError("For the 'fsq' dataset, sub_dataname must be either 'nyc' or 'tky'.")
        # Construct the path for fsq sub-datasets
        base_path = f"data/{dataname}/{sub_dataname}/{sub_dataname}"
    else:
        # Original path for other datasets like 'geolife'
        base_path = f"data/{dataname}/{dataname}"

    train_data_path = f"{base_path}_train.csv"
    valid_data_path = f"{base_path}_valid.csv"
    test_file_path = f"{base_path}_testset.pk"

    # Get training and validation set and merge them
    train_data = pd.read_csv(train_data_path)
    valid_data = pd.read_csv(valid_data_path)

    # Get test data
    with open(test_file_path, "rb") as f:
        test_file = pickle.load(f)  # test_file is a list of dict

    # merge train and valid data
    tv_data = pd.concat([train_data, valid_data], ignore_index=True)
    tv_data.sort_values(['user_id', 'start_day', 'start_min'], inplace=True)
    if dataname == 'geolife':
        tv_data['duration'] = tv_data['duration'].astype(int)

    print("Number of total test sample: ", len(test_file))
    return tv_data, test_file


def convert_to_12_hour_clock(minutes):
    if minutes < 0 or minutes >= 1440:
        return "Invalid input. Minutes should be between 0 and 1439."

    hours = minutes // 60
    minutes %= 60

    period = "AM"
    if hours >= 12:
        period = "PM"

    if hours == 0:
        hours = 12
    elif hours > 12:
        hours -= 12

    return f"{hours:02d}:{minutes:02d} {period}"


def int2dow(int_day):
    tmp = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
           3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    return tmp[int_day]


def get_logger(logger_name, log_dir='logs/'):
    # Create log dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set its log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a file handler and set its log level
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    log_file = 'log_file' + formatted_datetime + '.log'
    log_file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_user_data(train_data, uid, num_historical_stay, logger):
    user_train = train_data[train_data['user_id']==uid]
    logger.info(f"Length of user {uid} train data: {len(user_train)}")
    user_train = user_train.tail(num_historical_stay)
    logger.info(f"Number of user historical stays: {len(user_train)}")
    return user_train


# Organising data
def organise_data(dataname, user_train, test_file, uid, logger, num_context_stay=5):
    # Use another way of organising data
    historical_data = []

    if dataname == 'geolife':
        for _, row in user_train.iterrows():
            historical_data.append(
                (convert_to_12_hour_clock(int(row['start_min'])),
                int2dow(row['weekday']),
                int(row['duration']),
                row['location_id'])
                )
    elif dataname == 'fsq':
        for _, row in user_train.iterrows():
            historical_data.append(
                (convert_to_12_hour_clock(int(row['start_min'])),
                int2dow(row['weekday']),
                row['location_id'])
                )

    logger.info(f"historical_data: {historical_data}")
    logger.info(f"Number of historical_data: {len(historical_data)}")

    # Get user ith test data
    list_user_dict = []
    for i_dict in test_file:
        if dataname == 'geolife':
            i_uid = i_dict['user_X'][0]
        elif dataname == 'fsq':
            i_uid = i_dict['user_X']
        if i_uid == uid:
            list_user_dict.append(i_dict)

    predict_X = []
    predict_y = []
    for i_dict in list_user_dict:
        construct_dict = {}
        if dataname == 'geolife':
            context = list(zip([convert_to_12_hour_clock(int(item)) for item in i_dict['start_min_X'][-num_context_stay:]], 
                            [int2dow(i) for i in i_dict['weekday_X'][-num_context_stay:]], 
                            [int(i) for i in i_dict['dur_X'][-num_context_stay:]], 
                            i_dict['X'][-num_context_stay:]))
        elif dataname == 'fsq':
            context = list(zip([convert_to_12_hour_clock(int(item)) for item in i_dict['start_min_X'][-num_context_stay:]], 
                            [int2dow(i) for i in i_dict['weekday_X'][-num_context_stay:]], 
                            i_dict['X'][-num_context_stay:]))
        target = (convert_to_12_hour_clock(int(i_dict['start_min_Y'])), int2dow(i_dict['weekday_Y']), None, "<next_place_id>")
        construct_dict['context_stay'] = context
        construct_dict['target_stay'] = target
        predict_y.append(i_dict['Y'])
        predict_X.append(construct_dict)

    #logger.info(f"predict_data: {predict_X}")
    logger.info(f"Number of predict_data: {len(predict_X)}")
    logger.info(f"predict_y: {predict_y}")
    logger.info(f"Number of predict_y: {len(predict_y)}")
    return historical_data, predict_X, predict_y


def single_query_top1(client, historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visit to a certain place during certain time;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different times (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys: "prediction" (place ID) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    completion = get_chat_completion(client, prompt)
    return completion


def single_query_top10(client, historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. Note that this will be None in the <target_stay> introduced later.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and day_of_week) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    completion = get_chat_completion(client, prompt)
    return completion


def single_query_top1_wot(client, historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. 
    place_id: an integer representing the unique place ID, which indicates where the stay is.    
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visit to a certain place during certain time;
    2. the context stays in <context>, which provide more recent activities of this user.

    Please organize your answer in a JSON object containing following keys: "prediction" (place ID) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    """
    completion = get_chat_completion(client, prompt)
    return completion


def single_query_top10_wot(client, historical_data, X):
    """
    Make a single query of 10 most likely places, without time information
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, duration, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    duration: an integer indicating the duration (in minute) of each stay. 
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times;
    2. the context stays in <context>, which provide more recent activities of this user.
  
    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not use line breaks in the reason.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    """
    completion = get_chat_completion(client, prompt)
    return completion


def single_query_top1_fsq(client, historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visit to a certain place during certain time.
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and weekday) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "prediction" (place ID) and "reason" (a concise explanation that supports your prediction)

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    completion = get_chat_completion(client, prompt)
    return completion


def single_query_top1_wot_fsq(client, historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.    
    
    Please infer what the <next_place_id> is (i.e., the most likely place ID), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visit to a certain place during certain time;
    2. the context stays in <context>, which provide more recent activities of this user.

    Please organize your answer in a JSON object containing following keys: "prediction" (place ID) and "reason" (a concise explanation that supports your prediction). Do not include line breaks in your output.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    """
    completion = get_chat_completion(client, prompt)
    return completion


def single_query_top10_fsq(client, historical_data, X):
    """
    Make a single query.
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Then you need to do next location prediction on <target_stay> which is the prediction target with unknown place ID denoted as <next_place_id> and 
    unknown duration denoted as None, while temporal information is provided.      
    
    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times.
    2. the context stays in <context>, which provide more recent activities of this user; 
    3. the temporal information (i.e., start_time and weekday) of target stay, which is important because people's activity varies during different time (e.g., nighttime versus daytime)
    and on different days (e.g., weekday versus weekend).

    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction)

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <target_stay>: {X['target_stay']}
    """
    completion = get_chat_completion(client, prompt)
    return completion


def single_query_top10_wot_fsq(client, historical_data, X):
    """
    Make a single query of 10 most likely places, without time information
    param: 
    X: one single sample containing context_stay and target_stay
    """
    prompt = f"""
    Your task is to predict a user's next location based on his/her activity pattern.
    You will be provided with <history> which is a list containing this user's historical stays, then <context> which provide contextual information 
    about where and when this user has been to recently. Stays in both <history> and <context> are in chronological order.
    Each stay takes on such form as (start_time, day_of_week, place_id). The detailed explanation of each element is as follows:
    start_time: the start time of the stay in 12h clock format.
    day_of_week: indicating the day of the week.
    place_id: an integer representing the unique place ID, which indicates where the stay is.

    Please infer what the <next_place_id> might be (please output the 10 most likely places which are ranked in descending order in terms of probability), considering the following aspects:
    1. the activity pattern of this user that you leared from <history>, e.g., repeated visits to certain places during certain times.
    2. the context stays in <context>, which provide more recent activities of this user.
  
    Please organize your answer in a JSON object containing following keys:
    "prediction" (the ID of the ten most probable places in descending order of probability) and "reason" (a concise explanation that supports your prediction). Do not use line breaks in the reason.

    The data are as follows:
    <history>: {historical_data}
    <context>: {X['context_stay']}
    <next_place_id>: 
    """
    completion = get_chat_completion(client, prompt)
    return completion


def load_results(filename):
    # Load previously saved results from a CSV file    
    results = pd.read_csv(filename)
    return results


def single_user_query(client, dataname, uid, historical_data, predict_X, predict_y,logger, top_k, is_wt, output_dir, sleep_query, sleep_crash):
    # Initialize variables
    total_queries = len(predict_X)
    logger.info(f"Total_queries: {total_queries}")

    processed_queries = 0
    current_results = pd.DataFrame({
        'user_id': None,
        'ground_truth': None,
        'prediction': None,
        'reason': None
    }, index=[])

    out_filename = f"{uid:02d}" + ".csv"
    out_filepath = os.path.join(output_dir, out_filename)

    try:
        # Attempt to load previous results if available
        current_results = load_results(out_filepath)
        processed_queries = len(current_results)
        logger.info(f"Loaded {processed_queries} previous results.")
    except FileNotFoundError:
        logger.info("No previous results found. Starting from scratch.")

    # Process remaining queries
    for i in range(processed_queries, total_queries):
    #for query in queries[processed_queries:]:
        
        logger.info(f'The {i+1}th sample: ')
        #logger.info(f"context: {predict_X[i]['context_stay']}")
        #logger.info(f"target stay: {predict_X[i]['target_stay']}")
        if dataname == 'geolife':
            if is_wt is True:
                if top_k == 1:
                    completions = single_query_top1(client, historical_data, predict_X[i])
                elif top_k == 10:
                    completions = single_query_top10(client, historical_data, predict_X[i])
                else:
                    raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")
            else:
                if top_k == 1:
                    completions = single_query_top1_wot(client, historical_data, predict_X[i])
                elif top_k == 10:
                    completions = single_query_top10_wot(client, historical_data, predict_X[i])
                else:
                    raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")
        elif dataname == 'fsq':
            if is_wt is True:
                if top_k == 1:
                    completions = single_query_top1_fsq(client, historical_data, predict_X[i])
                elif top_k == 10:
                    completions = single_query_top10_fsq(client, historical_data, predict_X[i])
                else:
                    raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")
            else:
                if top_k == 1:
                    completions = single_query_top1_wot_fsq(client, historical_data, predict_X[i])
                elif top_k == 10:
                    completions = single_query_top10_wot_fsq(client, historical_data, predict_X[i])
                else:
                    raise ValueError(f"The top_k must be one of 1, 10. However, {top_k} was provided")

        response = completions.choices[0].message.content

        # Log the prediction results and usage.
        logger.info(f"Pred results: {response}")
        logger.info(f"Ground truth: {predict_y[i]}")
        logger.info(dict(completions).get('usage'))

        try:
            res_dict = ast.literal_eval(response)  # Convert the string to a dictionary object
            if top_k != 1:
                res_dict['prediction'] = str(res_dict['prediction'])
            res_dict['user_id'] = uid
            res_dict['ground_truth'] = predict_y[i]
        except Exception as e:
            res_dict = {'user_id': uid, 'ground_truth': predict_y[i], 'prediction': -100, 'reason': None}
            logger.info(e)
            logger.info(f"API request failed for the {i+1}th query")
            # time.sleep(sleep_crash)
        finally:
            new_row = pd.DataFrame(res_dict, index=[0])  # A dataframe with only one record
            current_results = pd.concat([current_results, new_row], ignore_index=True)  # Add new row to the current df

    # Save the current results
    current_results.to_csv(out_filepath, index=False)
    #save_results(current_results, out_filename)
    logger.info(f"Saved {len(current_results)} results to {out_filepath}")

    # Continue processing remaining queries
    if len(current_results) < total_queries:
        #remaining_predict_X = predict_X[len(current_results):]
        #remaining_predict_y = predict_y[len(current_results):]
        #remaining_queries = queries[len(current_results):]
        logger.info("Restarting queries from the last successful point.")
        single_user_query(client, dataname, uid, historical_data, predict_X, predict_y,
                          logger, top_k, is_wt, output_dir, sleep_query, sleep_crash)


def query_all_user(client, dataname, uid_list, logger, train_data, num_historical_stay,
                   num_context_stay, test_file, top_k, is_wt, output_dir, sleep_query, sleep_crash):
    for uid in uid_list:
        logger.info(f"=================Processing user {uid}==================")
        user_train = get_user_data(train_data, uid, num_historical_stay, logger)
        historical_data, predict_X, predict_y = organise_data(dataname, user_train, test_file, uid, logger, num_context_stay)
        single_user_query(client, dataname, uid, historical_data, predict_X, predict_y, logger, top_k=top_k, 
                          is_wt=is_wt, output_dir=output_dir, sleep_query=sleep_query, sleep_crash=sleep_crash)


def get_unqueried_user(dataname, sub_dataname=None, output_dir='output/'):
    """
    Determines the list of user IDs that have not yet been processed.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_user_id = []
    if dataname == "geolife":
        all_user_id = [i + 1 for i in range(45)]
    elif dataname == "fsq":
        if sub_dataname == 'nyc':
            # 纽约 (nyc) 用户总数
            all_user_id = [i + 1 for i in range(535)]
        elif sub_dataname == 'tky':
            # 东京 (tky) 用户总数
            all_user_id = [i + 1 for i in range(1643)]
        else:
            raise ValueError("For 'fsq' dataset, sub_dataname must be 'nyc' or 'tky'.")

    if not all_user_id:
        raise ValueError("Could not determine the list of users.")

    processed_id = [int(file.split('.')[0]) for file in os.listdir(output_dir) if file.endswith('.csv')]
    remain_id = [i for i in all_user_id if i not in processed_id]
    print(remain_id)
    print(f"Number of the remaining id: {len(remain_id)}")
    return remain_id


def main():
    client = OpenAI(
        # 重要提示：请将下面的密钥替换为您自己的有效OpenAI API密钥
        api_key="your_openai_api_key_here"
    )

    # --- 参数配置 ---
    # 1. 选择主数据集: "geolife" 或 "fsq"
    dataname = "fsq"
    
    # 2. 如果主数据集是 "fsq", 请选择子数据集: "nyc" 或 "tky"
    sub_dataname = "nyc"  # 在这里切换 'nyc' 或 'tky'

    # 3. 其他参数
    num_historical_stay = 40  # M
    num_context_stay = 5  # N
    top_k = 1  # 预测结果数量 k (1 或 10)
    with_time = False  # 目标中是否包含时间信息
    sleep_single_query = 0.1  # 每次查询之间的休眠时间
    sleep_if_crash = 1  # 如果API请求失败的休眠时间
    
    # 根据选择自动生成输出和日志目录
    # 示例: "output/fsq/nyc/top1_wot"
    experiment_name = f"top{top_k}" + ("_wt" if with_time else "_wot")
    output_dir = f"output/{dataname}/{sub_dataname}/{experiment_name}"
    log_dir = f"logs/{dataname}/{sub_dataname}/{experiment_name}"

    # --- 程序执行 ---
    # 根据选择加载对应的数据集
    tv_data, test_file = get_dataset(dataname, sub_dataname)

    logger = get_logger('my_logger', log_dir=log_dir)

    # 获取未处理的用户列表
    uid_list = get_unqueried_user(dataname, sub_dataname, output_dir)
    print(f"uid_list: {uid_list}")

    # 对所有未处理的用户进行查询
    query_all_user(client, dataname, uid_list, logger, tv_data, num_historical_stay, num_context_stay,
                   test_file, output_dir=output_dir, top_k=top_k, is_wt=with_time,
                   sleep_query=sleep_single_query, sleep_crash=sleep_if_crash)

    print("Query done")


if __name__=="__main__":
    main()