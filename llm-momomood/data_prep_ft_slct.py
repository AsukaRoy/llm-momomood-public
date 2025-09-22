import pandas as pd
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
import itertools
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthDataProcessor:
    """
    A class to process mental health data from mobile phone sensors and surveys.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the processor with data directory.
        
        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = Path(data_dir)
        self.dataframes = {}
        self.users = []
        
    def load_data(self) -> None:
        """Load all required CSV files."""
        file_mappings = {
            'battery': 'battery_4epochs.csv',
            'accelerometer': 'accelerometer_4epochs.csv',
            'screen': 'screen_4epochs.csv',
            'apps': 'application_4epochs.csv',
            'location': 'location_4epochs.csv',
            'phq': 'PHQ9_scores.csv',
            'survey': 'survey_all.csv'
        }
        
        for key, filename in file_mappings.items():
            filepath = self.data_dir / filename
            try:
                self.dataframes[key] = pd.read_csv(filepath, index_col=None)
                logger.info(f"Loaded {filename}")
            except FileNotFoundError:
                logger.error(f"File not found: {filepath}")
                raise
        
        # Remove duplicates from survey data
        self.dataframes['survey'] = self.dataframes['survey'].drop_duplicates(subset=['user'])
    
    @staticmethod
    def calculate_avg_std(row: pd.Series, col_names: List[str]) -> float:
        """
        Calculate average standard deviation for given columns.
        
        Args:
            row: Pandas Series containing the data
            col_names: List of column names to calculate from
            
        Returns:
            Average standard deviation
        """
        sum_squares = sum(row[col]**2 for col in col_names if col in row)
        return np.sqrt(sum_squares / len(col_names)) if sum_squares > 0 else 0
    
    def filter_user_data(self, df: pd.DataFrame, user_id: int, 
                        start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Filter dataframe for specific user and date range.
        
        Args:
            df: Input dataframe
            user_id: User ID to filter for
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Filtered dataframe
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        return df[(df['date'] > start_date) & (df['date'] <= end_date) & (df['user'] == user_id)]
    
    def get_battery_sequence(self, biweek_dates: List[pd.Timestamp], user_id: int,
                            start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[List, List, List[int]]:
        """
        Extract battery-related sequences for a user.
        
        Args:
            biweek_dates: List of dates for the two weeks period
            user_id: User ID
            start_date: Start date
            end_date: End date
            
        Returns:
            Tuple of (battery_std_14ds, battery_afternoon_14ds, nan_indices)
        """
        df = self.filter_user_data(self.dataframes['battery'], user_id, start_date, end_date)
        
        battery_cols = [
            'battery:battery_mean_level:night',
            'battery:battery_mean_level:morning',
            'battery:battery_mean_level:afternoon',
            'battery:battery_mean_level:evening',
            'battery:battery_std_level:morning'
        ]
        
        # Calculate battery metrics
        battery_mean = df[battery_cols[:4]].apply(
            lambda row: (row.sum() / len(row)) if row.sum() > 0 else 0, axis=1
        ).astype('int32').tolist()
        
        battery_std = df[battery_cols[4:]].apply(
            lambda row: self.calculate_avg_std(row, battery_cols[4:]), axis=1
        ).astype('int32').tolist()
        
        battery_afternoon = df['battery:battery_mean_level:afternoon'].astype('int32').tolist()
        
        # Map to 14-day sequence
        battery_mean_14ds, battery_std_14ds, battery_afternoon_14ds = [], [], []
        date_list = df['date'].tolist()
        
        for date in biweek_dates:
            if date in date_list:
                idx = date_list.index(date)
                battery_mean_14ds.append(battery_mean[idx])
                battery_std_14ds.append(battery_std[idx])
                battery_afternoon_14ds.append(battery_afternoon[idx])
            else:
                battery_mean_14ds.append('NaN')
                battery_std_14ds.append('NaN')
                battery_afternoon_14ds.append('NaN')
        
        # Handle zero values as NaN
        nan_indices = []
        for i, mean_val in enumerate(battery_mean_14ds):
            if mean_val == 0:
                nan_indices.append(i)
                battery_mean_14ds[i] = 'NaN'
                battery_std_14ds[i] = 'NaN'
                battery_afternoon_14ds[i] = 'NaN'
        
        return battery_std_14ds, battery_afternoon_14ds, nan_indices
    
    def get_accelerometer_sequence(self, biweek_dates: List[pd.Timestamp], nan_indices: List[int],
                                  user_id: int, start_date: pd.Timestamp, end_date: pd.Timestamp,
                                  metric_type: str = 'max') -> List:
        """
        Extract accelerometer sequences for a user.
        
        Args:
            biweek_dates: List of dates for the two weeks period
            nan_indices: Indices to mark as NaN
            user_id: User ID
            start_date: Start date
            end_date: End date
            metric_type: Type of metric ('max' or 'afternoon')
            
        Returns:
            List of accelerometer values
        """
        df = self.filter_user_data(self.dataframes['accelerometer'], user_id, start_date, end_date)
        
        accelerometer_cols = [
            'magnitude_mean:morning',
            'magnitude_mean:afternoon',
            'magnitude_mean:evening',
            'magnitude_mean:night'
        ]
        
        if metric_type == 'max':
            values = df[accelerometer_cols].apply(
                lambda row: row.max() if row.max() > 0 else 0, axis=1
            ).round(1).tolist()
        else:  # afternoon
            values = df['magnitude_mean:afternoon'].round(1).tolist()
        
        return self._map_to_14day_sequence(values, df['date'].tolist(), biweek_dates, nan_indices)
    
    def get_application_sequence(self, biweek_dates: List[pd.Timestamp], nan_indices: List[int],
                                user_id: int, start_date: pd.Timestamp, end_date: pd.Timestamp,
                                category: str, time_of_day: str) -> List:
        """
        Extract application usage sequences for a user.
        
        Args:
            biweek_dates: List of dates for the two weeks period
            nan_indices: Indices to mark as NaN
            user_id: User ID
            start_date: Start date
            end_date: End date
            category: App category
            time_of_day: Time period
            
        Returns:
            List of application usage values in minutes
        """
        df = self.filter_user_data(self.dataframes['apps'], user_id, start_date, end_date)
        
        app_duration = df[f'application:duration:{category}:{time_of_day}'].apply(
            lambda x: 0 if x == 0 else x / 60
        ).round(1).tolist()
        
        return self._map_to_14day_sequence(app_duration, df['date'].tolist(), biweek_dates, nan_indices)
    
    def get_screen_sequence(self, biweek_dates: List[pd.Timestamp], nan_indices: List[int],
                           user_id: int, start_date: pd.Timestamp, end_date: pd.Timestamp,
                           time_of_day: str, metric_type: str = 'off') -> List:
        """
        Extract screen usage sequences for a user.
        
        Args:
            biweek_dates: List of dates for the two weeks period
            nan_indices: Indices to mark as NaN
            user_id: User ID
            start_date: Start date
            end_date: End date
            time_of_day: Time period
            metric_type: Type of metric ('off' or 'use')
            
        Returns:
            List of screen usage values in hours
        """
        df = self.filter_user_data(self.dataframes['screen'], user_id, start_date, end_date)
        
        column_name = f'screen:screen_{metric_type}_durationtotal:{time_of_day}'
        screen_duration = df[column_name].apply(
            lambda x: 0 if x == 0 else x / (60 * 60)
        ).round(1).tolist()
        
        return self._map_to_14day_sequence(screen_duration, df['date'].tolist(), biweek_dates, nan_indices)
    
    def get_location_sequence(self, biweek_dates: List[pd.Timestamp], nan_indices: List[int],
                             user_id: int, start_date: pd.Timestamp, end_date: pd.Timestamp,
                             metric_type: str = 'ratio') -> List:
        """
        Extract location-based sequences for a user.
        
        Args:
            biweek_dates: List of dates for the two weeks period
            nan_indices: Indices to mark as NaN
            user_id: User ID
            start_date: Start date
            end_date: End date
            metric_type: Type of metric ('ratio' or 'locations')
            
        Returns:
            List of location-based values
        """
        df = self.filter_user_data(self.dataframes['location'], user_id, start_date, end_date)
        
        if metric_type == 'ratio':
            still_times = df['location:n_static'].fillna('NaN').tolist()
            movement_times = df['location:n_moving'].fillna('NaN').tolist()
            
            still_times_14ds = self._map_to_14day_sequence(still_times, df['date'].tolist(), biweek_dates, nan_indices)
            movement_times_14ds = self._map_to_14day_sequence(movement_times, df['date'].tolist(), biweek_dates, nan_indices)
            
            # Calculate ratio
            ratio_14ds = []
            for i in range(len(still_times_14ds)):
                if still_times_14ds[i] == 'NaN' or movement_times_14ds[i] == 'NaN':
                    ratio_14ds.append('NaN')
                elif movement_times_14ds[i] == 0:
                    ratio_14ds.append('Inf')
                else:
                    ratio_14ds.append(round(still_times_14ds[i] / movement_times_14ds[i], 2))
            
            return ratio_14ds
        else:  # locations
            locations_visited = df['location:n_bins'].fillna('NaN').tolist()
            return self._map_to_14day_sequence(locations_visited, df['date'].tolist(), biweek_dates, nan_indices)
    
    def _map_to_14day_sequence(self, values: List, date_list: List, biweek_dates: List,
                              nan_indices: List[int]) -> List:
        """
        Map values to 14-day sequence with NaN handling.
        
        Args:
            values: List of values to map
            date_list: List of dates corresponding to values
            biweek_dates: Target 14-day date sequence
            nan_indices: Indices to mark as NaN
            
        Returns:
            14-day sequence with NaN values where appropriate
        """
        result = []
        for date in biweek_dates:
            if date in date_list:
                result.append(values[date_list.index(date)])
            else:
                result.append('NaN')
        
        for i in nan_indices:
            if i < len(result):
                result[i] = 'NaN'
        
        return result
    
    
    def extract_user_features(self, user_id: int, target_date: pd.Timestamp,
                             feature_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features for a specific user and date.
        
        Args:
            user_id: User ID
            target_date: Target date for extraction
            feature_config: Configuration for features to extract
            
        Returns:
            Dictionary containing extracted features
        """
        two_week_before = target_date - pd.Timedelta(days=14)
        biweek_dates = [two_week_before + pd.Timedelta(days=i) for i in range(1, 15)]
        
        # Get battery sequence and NaN indices
        battery_std_14ds, battery_afternoon_14ds, nan_indices = self.get_battery_sequence(
            biweek_dates, user_id, two_week_before, target_date
        )
        
        # Check if too many NaN values
        if battery_std_14ds.count('NaN') > 7:
            return None
        
        # Initialize features dictionary
        features = {}
        
        # Only add features that are enabled in feature_config
        if feature_config.get('battery_afternoon', False):
            features['battery_afternoon'] = battery_afternoon_14ds
        
        if feature_config.get('screen_off_night_duration', False):
            features['screen_off_night_duration'] = self.get_screen_sequence(
                biweek_dates, nan_indices, user_id, two_week_before, target_date, 'night', 'off'
            )
        
        if feature_config.get('screen_use_duration_afternoon', False):
            features['screen_use_duration_afternoon'] = self.get_screen_sequence(
                biweek_dates, nan_indices, user_id, two_week_before, target_date, 'afternoon', 'use'
            )
        
        if feature_config.get('magnitude_max', False):
            features['magnitude_max'] = self.get_accelerometer_sequence(
                biweek_dates, nan_indices, user_id, two_week_before, target_date, 'max'
            )
        
        if feature_config.get('magnitude_afternoon', False):
            features['magnitude_afternoon'] = self.get_accelerometer_sequence(
                biweek_dates, nan_indices, user_id, two_week_before, target_date, 'afternoon'
            )
        
        return features
    
    def process_users(self, feature_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process all users and extract their data.
        
        Args:
            feature_config: Configuration for features to extract
            
        Returns:
            List of processed user data
        """
        users = []
        
        for user_id, group in self.dataframes['phq'].groupby('user'):
            user = {'user_id': user_id, 'data': []}
            
            # Add demographic information
            user_survey = self.dataframes['survey'][self.dataframes['survey']['user'] == user_id]
            if len(user_survey) > 0:
                user['gender'] = int(user_survey['bg_sex'].values[0])
                user['group'] = user_survey['group'].values[0]
                user['age'] = int(user_survey['bg_age'].values[0])
            
            # Process each PHQ measurement
            for _, row in group.iterrows():
                if row['idx'] == 1 or pd.isna(row['PHQ9']):
                    continue
                
                target_date = pd.to_datetime(row['date'])
                features = self.extract_user_features(user_id, target_date, feature_config)
                
                if features is not None:
                    batch_data = {
                        'date': target_date.strftime('%Y-%m-%d'),
                        'PHQ_week2': row['PHQ9'],
                        **features
                    }
                    user['data'].append(batch_data)
            
            if user['data']:
                users.append(user)
        
        return users
    
    def save_user_data(self, users: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save user data to JSON file.
        
        Args:
            users: List of user data
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2)
        logger.info(f"Saved user data to {output_path}")


class PromptGenerator:
    """
    A class to generate prompts for mental health prediction.
    """
    
    def __init__(self, threshold: float = 3.0):
        """
        Initialize the prompt generator.
        
        Args:
            threshold: Threshold for determining depression state changes
        """
        self.threshold = threshold
        self.pre_metrics = {
            'battery_afternoon': 'Average battery levels in the afternoon over the previous two weeks, reported as a percentage',
            'screen_off_night_duration': 'Screen off duration of the phone in the night over the previous two weeks, calculated in hours',
            'magnitude_max': 'Maximum magnitude of the phone in the previous two weeks, calculated in g',
            'magnitude_afternoon': 'Maximum magnitude of the phone in the afternoon over the previous two weeks, calculated in g',
            'screen_use_duration_afternoon': 'Screen use duration in the afternoon over the previous two weeks, calculated in hours',
        }
        self.cur_metrics = {
            'battery_afternoon': 'Average battery levels in the afternoon over the current two weeks, reported as a percentage',
            'screen_off_night_duration': 'Screen off duration of the phone in the night over the current two weeks, calculated in hours',
            'magnitude_max': 'Maximum magnitude of the phone in the current two weeks, calculated in g',
            'magnitude_afternoon': 'Maximum magnitude of the phone in the afternoon over the current two weeks, calculated in g',
            'screen_use_duration_afternoon': 'Screen use duration in the afternoon over the current two weeks, calculated in hours',
        }
    
    def determine_depression_change(self, current_phq: float, previous_phq: float) -> str:
        """
        Determine depression state change based on PHQ scores.
        
        Args:
            current_phq: Current PHQ score
            previous_phq: Previous PHQ score
            
        Returns:
            Depression change label
        """
        if current_phq - previous_phq > self.threshold:
            return "More Depressed"
        elif previous_phq - current_phq > self.threshold:
            return "Less Depressed"
        else:
            return "Remains"
    
    def calculate_historical_averages(self, user_data: List[Dict[str, Any]], 
                                    current_index: int) -> Dict[str, float]:
        """
        Calculate historical averages for user metrics.
        
        Args:
            user_data: List of user data points
            current_index: Index of current data point
            
        Returns:
            Dictionary of historical averages
        """
        # Get the available metrics from the current data point
        current_data = user_data[current_index]
        available_metrics = [key for key in current_data.keys() if key in self.cur_metrics]
        
        historical_values = {}
        
        # Only process metrics that are actually present in the data
        for metric in available_metrics:
            # Create proper historical description
            historical_desc = self.cur_metrics[metric].replace('over the current two weeks', 'over the previous weeks')
            historical_values[metric] = [
                historical_desc,
                0,  # sum
                0   # count
            ]
        
        for j in range(current_index):
            his_data = user_data[j]
            for metric in available_metrics:
                if metric in his_data:
                    # Remove NaN/Inf values
                    values = [k for k in his_data[metric] if isinstance(k, (int, float)) and not np.isnan(k)]
                    if values:
                        historical_values[metric][2] += len(values)
                        historical_values[metric][1] += sum(values)
        
        # Calculate averages
        for key in historical_values:
            if historical_values[key][2] != 0:
                avg = round(historical_values[key][1] / historical_values[key][2], 1)
                historical_values[key][1] = avg
            else:
                historical_values[key][1] = 'NaN'
        
        return historical_values
    
    def generate_prompt_text(self, user_context: str, pre_statistics: str, 
                           cur_statistics: str, pre_depression: str,
                           historical_stats: str = "", initial_state: str = "") -> Dict[str, str]:
        """
        Generate different versions of prompt text.
        
        Args:
            user_context: Context about the user
            pre_statistics: Previous period statistics
            cur_statistics: Current period statistics
            pre_depression: Previous depression state
            historical_stats: Historical statistics
            initial_state: Initial depression state
            
        Returns:
            Dictionary of different prompt versions
        """
        pre_data_start = "Mobile phone data over the previous two weeks: "
        cur_data_start = "Mobile phone data over the current two weeks: "
        pre_state = f"This person was {pre_depression} over the previous two weeks."
        cur_state_prediction = "Determine how the mental health state changes and only output the label:"
        history_start = "Within the previous weeks, the average mobile phone usage: "
        # basis prompt only contains current data
        PhQ_explain = ("PHQ-9 is an established diagnostic instrument designed to screen for the presence "
                       "and measure the severity of depressive symptoms in individuals. Respondents of the "
                       "PHQ-9 are asked to reflect on their experiences over the past two weeks, rating the "
                       "frequency of symptoms such as sleep disturbance, loss of interest, energy levels, and mood changes.")
        
        prompts = {
            "Basic": (pre_data_start + pre_statistics + pre_state + 
                      cur_data_start + cur_statistics + cur_state_prediction),
            "Health_Informtion": (PhQ_explain + pre_data_start + pre_statistics +
                                  pre_state + cur_data_start + cur_statistics +
                                  cur_state_prediction),
            "Statistics": (history_start + historical_stats +
                           pre_data_start + pre_statistics + pre_state + 
                           cur_data_start + cur_statistics + cur_state_prediction),
            "All": (PhQ_explain + user_context + initial_state +
                    history_start + historical_stats + 
                    pre_data_start + pre_statistics + pre_state + 
                    cur_data_start + cur_statistics + cur_state_prediction)
        }
        
        return prompts
    
    def generate_prompts(self, users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate prompts for all users.
        
        Args:
            users: List of user data
            
        Returns:
            List of generated prompts
        """
        prompts = []
        
        for user in users:
            gender_str = "male" if user['gender'] == 1 else "female"
            initial_depression = "not depressed" if user['group'] == "mmm-control" else "depressed"
            user_context = f"The following data comes from a {user['age']}-year-old {gender_str}. "
            initial_state = f" This person was {initial_depression} when they first entered the experiment. "
            
            for i in range(1, len(user['data'])):
                cur_data = user['data'][i]
                pre_data = user['data'][i-1]
                
                # Check if dates are 14 days apart
                if (pd.to_datetime(cur_data['date']) - pd.to_datetime(pre_data['date'])).days != 14:
                    continue
                
                # Calculate historical averages
                historical_values = self.calculate_historical_averages(user['data'], i)
                
                # Generate statistics text
                cur_statistics = ""
                pre_statistics = ""
                historical_stats = ""
                
                # Only process metrics that exist in the current data
                for metric in cur_data.keys():
                    if metric in self.cur_metrics:  # Only process known metrics
                        cur_values = cur_data[metric]
                        pre_values = pre_data.get(metric, [])  # Use get() in case metric doesn't exist in pre_data
                        
                        cur_statistics += f"{self.cur_metrics[metric]}: {', '.join(str(x) for x in cur_values)}. "
                        if pre_values:  # Only add if pre_values exists
                            pre_statistics += f"{self.pre_metrics[metric]}: {', '.join(str(x) for x in pre_values)}. "
                    
                    if metric in historical_values:
                        historical_stats += f"{historical_values[metric][0]}: {historical_values[metric][1]}. "
                
                # Determine depression states
                pre_depression = 'in depression' if pre_data['PHQ_week2'] >= 10 else 'in non-depression'
                cur_depression = 'in depression' if cur_data['PHQ_week2'] >= 10 else 'in non-depression'
                label = self.determine_depression_change(cur_data['PHQ_week2'], pre_data['PHQ_week2'])
                
                # Generate prompt variations
                prompt_texts = self.generate_prompt_text(
                    user_context, pre_statistics, cur_statistics, pre_depression,
                    historical_stats, initial_state
                )
                
                prompt = {
                    'user': user['user_id'],
                    'current PHQ': cur_data['PHQ_week2'],
                    'previous PHQ': pre_data['PHQ_week2'],
                    'previous depression state': pre_depression,
                    'current depression state': cur_depression,
                    'label': label,
                    **prompt_texts
                }
                
                prompts.append(prompt)
        
        return prompts
    
    def save_prompts(self, prompts: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save prompts to CSV file.
        
        Args:
            prompts: List of prompts
            output_path: Path to save the CSV file
        """
        df = pd.DataFrame(prompts)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved prompts to {output_path}")

def main():
    """
    Main function to run the mental health data processing pipeline.
    """
    # Configuration
    data_dir = "/m/cs/scratch/networks-nima-mmm2018/momo_processed"
    output_dir = "/m/cs/scratch/networks-nima-mmm2018/yunhao/llm-momomood/data/revision_prompt"
    
    threshold = 3
    
    # Feature configuration
    feature_keys = [
    'battery_afternoon',
    'screen_off_night_duration',
    'magnitude_max',
    'screen_use_duration_afternoon',
    ]
    
    for config_tuple in itertools.product([True, False], repeat=len(feature_keys)):
        feature_config = dict(zip(feature_keys, config_tuple))
        # Skip all-False configuration
        if not any(feature_config.values()):
            continue
    
        feature_name = "_".join([key for key, value in feature_config.items() if value])
    
        try:
            # Initialize processor
            processor = MentalHealthDataProcessor(data_dir)
            processor.load_data()
            
            # Process users
            users = processor.process_users(feature_config)
            
            # Save user data
            json_path = f"{output_dir}/state_transition_sequences_{feature_name}.json"
            processor.save_user_data(users, json_path)
            
            # Generate prompts
            prompt_generator = PromptGenerator(threshold=threshold)
            prompts = prompt_generator.generate_prompts(users)
            
            # Save prompts
            csv_path = f"{output_dir}/state_transition_sequences_{feature_name}_prompts_{threshold}.csv"
            prompt_generator.save_prompts(prompts, csv_path)
            
            logger.info(f"Pipeline completed successfully for {feature_name}. Processed {len(users)} users, generated {len(prompts)} prompts.")
            
        except Exception as e:
            logger.error(f"Pipeline failed for {feature_name} with error: {str(e)}")
            continue


if __name__ == "__main__":
    main()
