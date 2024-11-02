import os

import pandas as pd

from utils.result import Result


def get_activities_data(directory: str) -> Result[pd.DataFrame]:
    df = pd.read_csv(f'{directory}/macro.csv')
    # Filter only the activities TAKEOFF and EXPLORE
    df = df[df.activity.isin(['TAKEOFF', 'EXPLORE'])]
    # Drop the payload column
    df = df.drop(['payload'], axis=1)
    return Result(success=True, data=df)


def get_position_data(directory: str) -> Result[pd.DataFrame]:
    df = pd.read_csv(f'{directory}/odom.csv')
    return Result(success=True, data=df)


def build_artefacts():
    directories = [os.path.join("data", f) for f in os.listdir('data') if os.path.isdir(os.path.join("data", f))]
    for directory in directories:
        df_activity = get_activities_data(directory).data
        df_position = get_position_data(directory).data

        df_activity['time'] = pd.to_datetime(df_activity['time'])
        df_position['time'] = pd.to_datetime(df_position['time'])

        # Sort both DataFrames by the datetime column
        df_activity = df_activity.sort_values(by='time')
        df_position = df_position.sort_values(by='time')

        df_position = join_positions_and_activities(df_activity, df_position)

        # Save the enriched dataset to a new CSV file
        file_name = f'dataset_{directory.split("/")[-1]}'
        df_position.to_csv(f'artefacts/{file_name}.csv', index=False)


def join_positions_and_activities(df_activity: pd.DataFrame, df_position: pd.DataFrame) -> pd.DataFrame:
    # Initialize a column in odom_df to hold the activity information
    df_position['activity'] = None

    # Loop through macro_df to find activity time ranges
    for i in range(len(df_activity) - 1):
        row = df_activity.iloc[i]
        next_row = df_activity.iloc[i + 1]

        # Check if this row marks the start and the next row marks the end of an activity
        if row['lifecycle'] == 'START' and next_row['lifecycle'] == 'STOP' and row['activity'] == next_row['activity']:
            start_time = row['time']
            end_time = next_row['time']
            activity = row['activity']

            # Assign activity only to rows within the start and end time interval in odom_df
            df_position.loc[(df_position['time'] >= start_time) & (df_position['time'] <= end_time), 'activity'] = activity

    # Set "IDLE" as the default activity for rows without any assigned activity
    df_position['activity'] = df_position['activity'].fillna('IDLE')
    return df_position


def join_artefacts():
    frames = []

    artefacts = [os.path.join("artefacts", f) for f in os.listdir('artefacts') if f.startswith('dataset_')]
    # Loop through the artefacts
    for frame in artefacts:
        # Load the dataset
        df_partial = pd.read_csv(frame)
        # Append the dataset to the main DataFrame
        frames.append(df_partial)

    df = pd.concat(frames)
    df = df.sort_values(by='time')

    # Save the joined dataset to a new CSV file
    df.to_csv('artefacts/training_dataset.csv', index=False)
