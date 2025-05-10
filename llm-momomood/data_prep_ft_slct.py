
import pandas as pd 
import json
import numpy as np

df_battery = pd.read_csv('/m/cs/scratch/networks-nima-mmm2018/momo_processed/battery_4epochs.csv', index_col = None)
df_accelerometer = pd.read_csv('/m/cs/scratch/networks-nima-mmm2018/momo_processed/accelerometer_4epochs.csv', index_col = None)
df_screen = pd.read_csv('/m/cs/scratch/networks-nima-mmm2018/momo_processed/screen_4epochs.csv', index_col = None)
df_apps = pd.read_csv('/m/cs/scratch/networks-nima-mmm2018/momo_processed/application_4epochs.csv', index_col = None)
df_location = pd.read_csv('/m/cs/scratch/networks-nima-mmm2018/momo_processed/location_4epochs.csv', index_col = None)

df_PHQ = pd.read_csv('/m/cs/scratch/networks-nima-mmm2018/momo_processed/PHQ9_scores.csv', index_col = None)
df_survey = pd.read_csv('/m/cs/scratch/networks-nima-mmm2018/momo_processed/survey_all.csv', index_col = None)
df_survey = df_survey.drop_duplicates(subset=['user'])

df_screen['screen:screen_use_durationtotal:afternoon']

def avg_std(row, col_names):
    s = 0
    for n in col_names:
        s += row[n]**2
    return np.sqrt(s/len(col_names)) if s > 0 else 0

def get_battery_sequence(biweekdate, df, two_week_before, target_date, userid):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for the specified user and date range
    df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]
    
    # Define battery level columns
    battery_cols = ['battery:battery_mean_level:night', 
                    'battery:battery_mean_level:morning', 
                    'battery:battery_mean_level:afternoon', 
                    'battery:battery_mean_level:evening',
                    'battery:battery_std_level:morning',

                    ]
    
    battery_mean  = df[battery_cols[:4]].apply(lambda row: (row.sum() / len(row)) if row.sum() > 0 else 0, axis=1).astype('int32').to_list()
    battery_std = df[battery_cols[4:]].apply(lambda row: avg_std(row, battery_cols[4:]), axis=1).astype('int32').to_list()
    battery_afternoon = df['battery:battery_mean_level:afternoon'].astype('int32').to_list()
    battery_mean_14ds = [] # include data of all 14 days
    battery_std_14ds = []
    battery_afternoon_14ds = []

    for d in biweekdate:
        if d in list(df['date']):
            battery_mean_14ds.append(battery_mean[list(df['date']).index(d)])
            battery_std_14ds.append(battery_std[list(df['date']).index(d)])
            battery_afternoon_14ds.append(battery_afternoon[list(df['date']).index(d)])
        else:
            battery_mean_14ds.append('NaN')
            battery_std_14ds.append('NaN')
            battery_afternoon_14ds.append('NaN')

    
    nan_index = []
    for i, mean_value in enumerate(battery_mean_14ds):
        if mean_value == 0:
            nan_index.append(i)
            battery_mean_14ds[i] = 'NaN'
            battery_std_14ds[i] = 'NaN'
            battery_afternoon_14ds[i] = 'NaN'
    
    #print('nan_index', len(nan_index))
    
    return battery_std_14ds, battery_afternoon_14ds, nan_index

def get_accelerometer_sequence_max(biweekdate, nan_index, df, two_week_before, target_date, userid):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for the specified user and date range
    df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]

    accelerometer_cols = ['magnitude_mean:morning', 
                    'magnitude_mean:afternoon', 
                    'magnitude_mean:evening', 
                    'magnitude_mean:night',
                    ]
    
    accelerometer_max  = df[accelerometer_cols[:4]].apply(lambda row: row.max() if row.max() > 0 else 0, axis=1).round(1).to_list()
    accelerometer_max_14ds = [] # include data of all 14 days

    for d in biweekdate:
        if d in list(df['date']):
            accelerometer_max_14ds.append(accelerometer_max[list(df['date']).index(d)])
        else:
            accelerometer_max_14ds.append('NaN')
    for i in nan_index:
        accelerometer_max_14ds[i] = 'NaN'
        
    return accelerometer_max_14ds

def get_accelerometer_sequence_afternoon(biweekdate, nan_index, df, two_week_before, target_date, userid):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for the specified user and date range
    df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]

    accelerometer_cols = [
                    'magnitude_mean:afternoon', 
                    ]
    accelerometer_afternoon = df[accelerometer_cols[0]].round(1).to_list()
    accelerometer_afternoon_14ds = []

    for d in biweekdate:
        if d in list(df['date']):
            accelerometer_afternoon_14ds.append(accelerometer_afternoon[list(df['date']).index(d)])
        else:
            accelerometer_afternoon_14ds.append('NaN')
    for i in nan_index:
        accelerometer_afternoon_14ds[i] = 'NaN'
        
    return accelerometer_afternoon_14ds

def get_application_sequence(biweekdate, nan_index, df, two_week_before, target_date, userid, category, time_of_day):
    df['date'] = pd.to_datetime(df['date']) 
    df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]
    
    app_duration = (df[f'application:duration:{category}:{time_of_day}'].apply(lambda x: 0 if x == 0 else x / 60)).round(1).to_list() # min
    app_duration_14ds = []

    
    for d in biweekdate:
        if d in list(df['date']):
            app_duration_14ds.append(app_duration[list(df['date']).index(d)])
        else:
            app_duration_14ds.append('NaN')
    
    for i in nan_index:
        app_duration_14ds[i] = 'NaN'
    
    return app_duration_14ds


# def get_screen_morning_sequence(biweekdate, nan_index, df, two_week_before, target_date, userid):
#     # Convert date column to datetime
#     df['date'] = pd.to_datetime(df['date'])
    
#     # Filter data for the specified user and date range
#     df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]

#     screen_off_duration_morning = (df['screen:screen_off_durationtotal:morning'].apply(lambda x: 0 if x == 0 else x / (60 * 60))).round(1).to_list()
#     screen_off_duration_morning_14ds = []

#     for d in biweekdate:
#         if d in list(df['date']):
#             screen_off_duration_morning_14ds.append(screen_off_duration_morning[list(df['date']).index(d)])
#         else:
#             screen_off_duration_morning_14ds.append('NaN')
    
#     for i in nan_index:
#         screen_off_duration_morning_14ds[i] = 'NaN'

#     # Convert the result to a list
#     return screen_off_duration_morning_14ds

# def get_screen_evening_sequence(biweekdate, nan_index, df, two_week_before, target_date, userid):
#     # Convert date column to datetime
#     df['date'] = pd.to_datetime(df['date'])
    
#     # Filter data for the specified user and date range
#     df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]

#     screen_off_duration_evening = (df['screen:screen_off_durationtotal:evening'].apply(lambda x: 0 if x == 0 else x / (60 * 60))).round(1).to_list()
#     screen_off_duration_evening_14ds = []

#     for d in biweekdate:
#         if d in list(df['date']):
#             screen_off_duration_evening_14ds.append(screen_off_duration_evening[list(df['date']).index(d)])
#         else:
#             screen_off_duration_evening_14ds.append('NaN')
    
#     for i in nan_index:
#         screen_off_duration_evening_14ds[i] = 'NaN'

#     # Convert the result to a list
#     return screen_off_duration_evening_14ds

def get_screen_use_sequence(biweekdate, nan_index, df, two_week_before, target_date, userid, time_of_day):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for the specified user and date range
    df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]

    screen_use_duration = (df[f'screen:screen_use_durationtotal:{time_of_day}'].apply(lambda x: 0 if x == 0 else x / (60 * 60))).round(1).to_list()
    screen_use_duration_14ds = []

    for d in biweekdate:
        if d in list(df['date']):
            screen_use_duration_14ds.append(screen_use_duration[list(df['date']).index(d)])
        else:
            screen_use_duration_14ds.append('NaN')
    
    for i in nan_index:
        screen_use_duration_14ds[i] = 'NaN'

    # Convert the result to a list
    return screen_use_duration_14ds

def get_screen_off_sequence(biweekdate, nan_index, df, two_week_before, target_date, userid, time_of_day):
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for the specified user and date range
    df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]

    screen_off_duration = (df[f'screen:screen_off_durationtotal:{time_of_day}'].apply(lambda x: 0 if x == 0 else x / (60 * 60))).round(1).to_list()
    screen_off_duration_14ds = []

    for d in biweekdate:
        if d in list(df['date']):
            screen_off_duration_14ds.append(screen_off_duration[list(df['date']).index(d)])
        else:
            screen_off_duration_14ds.append('NaN')
    
    for i in nan_index:
        screen_off_duration_14ds[i] = 'NaN'

    # Convert the result to a list
    return screen_off_duration_14ds


def get_still_movement(biweekdate, nan_index, df, two_week_before, target_date, userid):
    
    df['date'] = pd.to_datetime(df['date']) 
    df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]
    still_times = df['location:n_static'].fillna('NaN').to_list()
    movement_times = df['location:n_moving'].fillna('NaN').to_list()

    still_times_14ds = []
    movement_times_14ds = []
    for d in biweekdate:
        if d in list(df['date']):
            still_times_14ds.append(still_times[list(df['date']).index(d)])
            movement_times_14ds.append(movement_times[list(df['date']).index(d)])
        else:
            still_times_14ds.append('NaN')
            movement_times_14ds.append('NaN')
    
    for i in nan_index:
        still_times_14ds[i] = 'NaN'
        movement_times_14ds[i] = 'NaN'
    
    # convert to still vs momvement ratio
    still_movement_ratio_14ds = []
    for i in range(len(still_times_14ds)):
        if still_times_14ds[i] == 'NaN' or movement_times_14ds[i] == 'NaN':
            still_movement_ratio_14ds.append('NaN')
        elif movement_times_14ds[i] == 0:
            still_movement_ratio_14ds.append('Inf')
        else:
            still_movement_ratio_14ds.append(round(still_times_14ds[i] / movement_times_14ds[i], 2))

    # Convert the result to a list
    return still_movement_ratio_14ds

def get_locations_visted(biweekdate, nan_index, df, two_week_before, target_date, userid):
    df['date'] = pd.to_datetime(df['date']) 
    df = df[(df['date'] > two_week_before) & (df['date'] <= target_date) & (df['user'] == userid)]
    locations_visted = df['location:n_bins'].fillna('NaN').to_list()
    locations_visted_14ds = []
    for d in biweekdate:
        if d in list(df['date']):
            locations_visted_14ds.append(locations_visted[list(df['date']).index(d)])
        else:
            locations_visted_14ds.append('NaN')
    for i in nan_index:
        locations_visted_14ds[i] = 'NaN'
    return locations_visted_14ds
    

users = []
# not_equal = 0
for user_id, group in df_PHQ.groupby('user'):
    user = {}
    user['user_id'] = user_id
    if len(df_survey[df_survey['user']==user_id]) > 0:
        user['gender'] = int(df_survey[df_survey['user']==user_id]['bg_sex'].values[0])
        user['group'] = df_survey[df_survey['user']==user_id]['group'].values[0]
        user['age'] = int(df_survey[df_survey['user']==user_id]['bg_age'].values[0])
        #print(type(user['age']))
     
    user['data'] = []
    for index, row in group.iterrows():
        batch_statistic = {}
        if(row['idx'] == 1 or pd.isna(row['PHQ9'])):
            continue
            
        target_date = pd.to_datetime(row['date'])
        two_week_before = target_date - pd.Timedelta(days=14)
        biweekdate = [two_week_before+pd.Timedelta(days=i) for i in range(1,15)]
        battery_std_14ds, battery_afternoon_14ds, nan_index = get_battery_sequence(biweekdate, df_battery, two_week_before, target_date, user_id)
        if battery_std_14ds.count('NaN') > 7:
            continue

        # socialmedia_app_duration_14ds = get_application_sequence(biweekdate, nan_index, df_apps, two_week_before, target_date, user_id, 'leisure', 'daily')
        # if socialmedia_app_duration_14ds.count('NaN')>12:
        #      continue
        #comm_app_duration_afternoon_14ds = get_application_sequence(biweekdate, nan_index, df_apps, two_week_before, target_date, user_id, 'comm', 'afternoon')
        #still_movement_ratio_14ds = get_still_movement(biweekdate, nan_index, df_location, two_week_before, target_date, user_id)
        #locations_visted_14ds = get_locations_visted(biweekdate, nan_index, df_location, two_week_before, target_date, user_id)
        batch_statistic = {'date' : f"{target_date.strftime('%Y-%m-%d')}", 'PHQ_week2': row['PHQ9']}
        # batch_statistic['socialmedia_app_duration'] = socialmedia_app_duration_14ds
        # batch_statistic['battery_std_level_morning'] = battery_std_14ds
        #batch_statistic['battery_afternoon'] = battery_afternoon_14ds
        batch_statistic['magnitude_max'] = get_accelerometer_sequence_max(biweekdate, nan_index, df_accelerometer, two_week_before, target_date, user_id)
        #batch_statistic['magnitude_afternoon'] = get_accelerometer_sequence_afternoon(biweekdate, nan_index, df_accelerometer, two_week_before, target_date, user_id)
        # batch_statistic['screen_off_morning_duration'] = get_screen_morning_sequence(biweekdate, nan_index, df_screen, two_week_before, target_date, user_id)
        #batch_statistic['screen_off_evening_duration'] = get_screen_evening_sequence(biweekdate, nan_index, df_screen, two_week_before, target_date, user_id)
        batch_statistic['screen_off_night_duration'] = get_screen_off_sequence(biweekdate, nan_index, df_screen, two_week_before, target_date, user_id, 'night')
        batch_statistic['screen_use_duration_afternoon'] = get_screen_use_sequence(biweekdate, nan_index, df_screen, two_week_before, target_date, user_id, 'afternoon')
        #batch_statistic['comm_app_duration_afternoon'] = comm_app_duration_afternoon_14ds
        # batch_statistic['sys_app_duration_night'] = get_application_sequence(biweekdate, nan_index, df_apps, two_week_before, target_date, user_id, 'system', 'night')
        #batch_statistic['still_movement_ratio'] = still_movement_ratio_14ds
        #batch_statistic['locations_visted'] = locations_visted_14ds
        # print(batch_statistic['n_static'].count('NaN'))
        user['data'].append(batch_statistic)
    
    if(user['data'] != []):
        users.append(user)

# print(not_equal)


num_data = 0
num_data_u = []
for u in users:
    num_data += len(u['data'])
    num_data_u.append(len(u['data']))
consective_date = 0
for u in users:
    for i in range(len(u['data'])-1):
        if str(pd.to_datetime(u['data'][i+1]['date']) - pd.to_datetime(u['data'][i]['date'])) == '14 days 00:00:00':
            consective_date += 1
consective_date        
# path = "./state_transition_sequences_magnitude_screen_sysappnight.json"
# path = "./state_transition_sequences_battery_magnitude_screen.json"
# path = "./state_transition_sequences_battery_screen.json"
# path = "./state_transition_sequences_battery_magnitude_screen_sysappnight.json"
# path = "./state_transition_sequences_sysappnight_noallnan.json"
# path = "./state_transition_sequences_socialmediaapp.json"
# path = "./feature_selection/state_transition_sequences_screen_evening.json"
# path = "./feature_selection/state_transition_sequences_battery_avg_afternoon.json"
# path = "./feature_selection/state_transition_sequences_screen_night.json"
# path = "./feature_selection/state_transition_sequences_screen_use_afternoon.json"
path = "./feature_selection/state_transition_sequences_magnitude_max+screen_off_night+screen_use_afternoon.json"
with open(path, 'w', encoding='utf-8') as f:
    json.dump(users, f)


# construct prompts using the information in .json, and store them into .csv;
# optional: add initial state; add history_info; add both;
# use_phq: whether use phq9 values to decide the label

feature = "magnitude_max+screen_off_night+screen_use_afternoon"
path_json = f"./feature_selection/state_transition_sequences_{feature}.json" 
def load_users(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        users = json.load(f)
    return users
users = load_users(path_json)

prompts = []

pre_metrics = {
            #'battery_std_level_morning': 'Standard deviation of battery levels in the morning over the previous two-week, reported as a percentage',
            #'battery_afternoon': 'Average battery levels in the afternoon over the previous two-week, reported as a percentage',
            'magnitude_max': 'Max values of the acceleration magnitude within the day over the previous two-week, measured using the accelerometer of the phone',
            #'magnitude_afternoon': 'Values of the acceleration magnitude in the afternoon over the previous two-week, measured using the accelerometer of the phone',
            #'screen_off_morning_duration': 'Screen off duration of the phone in the morning over the previous two-week, calculated in hours',
            #'screen_off_evening_duration': 'Screen off duration of the phone in the evening over the previous two-week, calculated in hours',
            'screen_off_night_duration': 'Screen off duration of the phone in the night over the previous two-week, calculated in hours',
            'screen_use_duration_afternoon': 'Screen use duration of the phone in the afternoon over the previous two-week, calculated in hours',
            #'sys_app_duration_night': 'System applications usage duration at night over the previous two-week, calculated in minutes',
            #'comm_app_duration_afternoon': 'Communication applications usage duration in the afternoon over the previous two-week, calculated in minutes',
            #'leisure_app_duration': 'Leisure applications daily usage over the previous two-week, calculated in minutes',
            #'socialmedia_app_duration': 'Social media applications daily usage over the previous two-week, calculated in minutes',
            #'n_static': 'Number of times staying static measured by using the mobile phone over the previous two-week',
            #'n_moving': 'Number of moves measured by using the mobile phone over the previous two-week',
            #'still_movement_ratio': 'Still vs movement ratio measured by using the mobile phone over the previous two-week',
            #'locations_visted': 'Number of locations visted measured by using the mobile phone over the previous two-week',
         }
cur_metrics = {
            #'battery_std_level_morning': 'Standard deviation of battery levels in the morning over the current two-week, reported as a percentage',
            #'battery_afternoon': 'Average battery levels in the afternoon over the current two-week, reported as a percentage',
            'magnitude_max': 'Max values of the acceleration magnitude within the day over the current two-week, measured using the accelerometer of the phone',
            #'magnitude_afternoon': 'Values of the acceleration magnitude in the afternoon over the current two-week, measured using the accelerometer of the phone',
            #'screen_off_morning_duration': 'Screen off duration of the phone in the morning over the current two-week, calculated in hours',
            #'screen_off_evening_duration': 'Screen off duration of the phone in the evening over the current two-week, calculated in hours',
            'screen_off_night_duration': 'Screen off duration of the phone in the night over the current two-week, calculated in hours',
            'screen_use_duration_afternoon': 'Screen use duration of the phone in the afternoon over the current two-week, calculated in hours',
            #'sys_app_duration_night': 'System applications usage duration at night over the current two-week, calculated in minutes',
            #'comm_app_duration_afternoon': 'Communication applications usage duration in the afternoon over the current two-week, calculated in minutes',
            #'leisure_app_duration': 'Leisure applications daily usage over the current two-week, calculated in minutes',
            #'socialmedia_app_duration': 'Social media applications daily usage over the current two-week, calculated in minutes',
            #'n_static': 'Number of times staying static measured by using the mobile phone over the current two-week',
            #'n_moving': 'Number of moves measured by using the mobile phone over the current two-week',
            #'still_movement_ratio': 'Still vs movement ratio measured by using the mobile phone over the current two-week',
            #'locations_visted': 'Number of locations visted measured by using the mobile phone over the current two-week',
        }


use_phq = True # fixed
threshold = 3 # fixed

for user in users:
    gender_str = "male" if user['gender'] == 1 else "female"
    initial_depresson = "not depressed" if user['group']=="mmm-control" else "depressed"
    user_context = f"The following data come from a {user['age']}-year-old {gender_str}."
    user_prompts = []
    for i in range(1,len(user['data'])):
        cur_data = user['data'][i]
        pre_data = user['data'][i-1]
        if str(pd.to_datetime(cur_data['date']) - pd.to_datetime(pre_data['date'])) != '14 days 00:00:00':
            continue

        # add history information based on data in previous weeks
        his_values_for_metrics = {
            #'battery_std_level_morning': ['battery levels standard deviation in the morning', 0, 0], 
            #'battery_afternoon': ['average battery levels in the afternoon', 0, 0],
            'magnitude_max': ['maximum acceleration magnitude', 0, 0],
            #'magnitude_afternoon': ['acceleration magnitude in the afternoon', 0, 0],
            #'screen_off_duration': ['Screen off duration', 0, 0],
            #'sys_app_duration_night': ['System applications usage duration at night', 0, 0], 
            #'comm_app_duration_afternoon': ['Communication applications usage in the afternoon', 0, 0],
            #'leisure_app_duration': ['Leisure applications daily usage', 0, 0],
            #'socialmedia_app_duration': ['Social media applications daily usage', 0, 0],
            #'n_static': ['Number of times staying static', 0, 0], 
            #'n_moving': ['Number of moves', 0, 0]
            #'still_movement_ratio': ['Still vs movement ratio', 0, 0],
            #'locations_visted': ['Number of locations visted', 0, 0]
            #'screen_off_morning_duration': ['Screen off duration in the morning', 0, 0],
            #'screen_off_evening_duration': ['Screen off duration in the evening', 0, 0],
            'screen_off_night_duration': ['Screen off duration in the night', 0, 0],
            'screen_use_duration_afternoon': ['Screen use duration in the afternoon', 0, 0],
            }
        
        for j in range(0, i):
            his_data = user['data'][j]
            for metric, description in cur_metrics.items():
                if metric in his_values_for_metrics:
                    # remove NaN/InF
                    v = [k for k in his_data[metric] if type(k)!= str]
                    # check if v is empty
                    if len(v) != 0:
                        # print(metric)
                        # print(his_values_for_metrics[metric])
                        his_values_for_metrics[metric][2] += len(v)
                        # print(his_values_for_metrics[metric][1])
                        his_values_for_metrics[metric][1] += sum(v)
                    # v = (np.mean(v)/i).round(1)
        for key, value in his_values_for_metrics.items():
            if his_values_for_metrics[key][2] != 0:
                avg = round((his_values_for_metrics[key][1] / his_values_for_metrics[key][2]),1)
                #avg = int((his_values_for_metrics[key][1] / his_values_for_metrics[key][2]))
                his_values_for_metrics[key][1] = avg
            else:
                his_values_for_metrics[key][1] = 'NaN'
                
        cur_depression = 'in depression' if cur_data['PHQ_week2'] >= 10 else 'in non-depression'
        pre_depression = 'in depression' if pre_data['PHQ_week2'] >= 10 else 'in non-depression'

        prompt = {
            'user': user['user_id'], 
            'data': user_context, 
            'data_with_init':user_context, 
            'data_with_history': user_context, 
            'data_with_init_his': user_context, }
        his_statistics = ""
        cur_statistics = ""
        pre_statistics = ""
        
        for metric, description in cur_metrics.items():
            if metric in cur_data:
                cur_values = cur_data[metric]
                pre_values = pre_data[metric]
                # if len(values) > 0:  # Ensure the list is not empty
                cur_stat_description = f"{description}: {', '.join(str(x) for x in cur_values)}. "
                cur_statistics += cur_stat_description
                pre_stat_description = f"{pre_metrics[metric]}: {', '.join(str(x) for x in pre_values)}. "
                pre_statistics += pre_stat_description
            if metric in his_values_for_metrics:
                his_values = his_values_for_metrics[metric][1]
                his_statistics += f"{his_values_for_metrics[metric][0]}: {his_values}. "
                
        initial_state = f" This person was {initial_depresson} when first entered the experiment." 
        pre_data_start = "\nNaN denotes missing data. Mobile phone data over the previous two-week: "
        cur_data_start = "\nMobile phone data over the current two-week: "
        pre_state = f"\nThis person was {pre_depression} over the previous two-week."
        cur_state_prediction_start = "\nDetermine how the mental health state changes and only output the label:"
        history_start = f"\nWithin the {2*i} weeks so far, the average mobile phone usage: "

        prompt['data'] += pre_data_start + pre_statistics + pre_state + cur_data_start + cur_statistics + cur_state_prediction_start
        prompt['data_with_history'] += pre_data_start + pre_statistics + pre_state + cur_data_start + cur_statistics + history_start + his_statistics + cur_state_prediction_start
        prompt['data_with_init'] += initial_state + pre_data_start + pre_statistics + pre_state + cur_data_start + cur_statistics + cur_state_prediction_start
        prompt['data_with_init_his'] += initial_state + pre_data_start + pre_statistics + pre_state + cur_data_start + cur_statistics + history_start + his_statistics + cur_state_prediction_start
        prompt['previous depression state'] = pre_depression
        
        if use_phq:
            prompt['current PHQ'] = cur_data['PHQ_week2']
            prompt['previous PHQ'] = pre_data['PHQ_week2']
            if cur_data['PHQ_week2'] - pre_data['PHQ_week2'] > threshold:
                prompt['label'] = "More Depressed"
            else:
                if pre_data['PHQ_week2'] - cur_data['PHQ_week2'] > threshold:
                    prompt['label'] = "Less Depressed"
                else:
                    prompt['label'] = 'Remains'
        else:
            prompt['current depression state'] = cur_depression
            if cur_depression == pre_depression:
                # prompt['label'] = 'Remains Depressed' if cur_depression == 'in depression' else 'Remains Non-depressed'
                prompt['label'] = 'Remains'
            else:
                # prompt['label'] = 'Declines' if cur_depression == 'in depression' else 'Improves'
                prompt['label'] = 'More Depressed' if cur_depression == 'in depression' else 'Less Depressed'
        
        user_prompts.append(prompt) 
    
    prompts.extend(user_prompts)
       
path_csv = f"./feature_selection/state_transition_sequences_{feature}_prompts_{threshold}.csv"
prompts_df = pd.DataFrame(prompts)
prompts_df.to_csv(path_csv, index=False)
