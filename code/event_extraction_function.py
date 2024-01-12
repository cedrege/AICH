import numpy as np
import pandas as pd

def delete_short_awake_windows(df, period = 30):
    series = df.series_id.unique().tolist()
    
    for serie in series:
        onset_events = df[(df['series_id'] == serie) & (df['onset'] == 1)]
        wakeup_events = df[(df['series_id'] == serie) & (df['wakeup'] == 1)]
        
        if not onset_events.empty and not wakeup_events.empty:
            # Find the indices of 'onset' and 'wakeup' events
            onset_indices = onset_events.index
            wakeup_indices = wakeup_events.index
            
            min_length = min(len(wakeup_indices), len(onset_indices))
            wakeup_indices = wakeup_indices[:min_length]
            onset_indices = onset_indices[:min_length]

            # Create a matrix of time differences
            time_diff_matrix = (df.loc[wakeup_indices, 'timestamp'].values[:, None] -
                                df.loc[onset_indices, 'timestamp'].values).astype('timedelta64[m]').astype(float)
            
            # Identify rows where time difference is less than period minutes and wakeup occurs before onset
            small_breaks = (np.abs(time_diff_matrix) < period) & (np.expand_dims(wakeup_indices, axis=1) < onset_indices)
            
            wakeup_to_remove, onset_to_remove = np.where(small_breaks)
            df.loc[wakeup_indices[wakeup_to_remove], 'wakeup'] = 0
            df.loc[onset_indices[onset_to_remove], 'onset'] = 0
            
def delete_short_sleep_windows(df, period = 30):
    df['score'] = np.nan
    df['event'] = np.nan
    
    series = df.series_id.unique().tolist()
    for serie in series:
        onset_events = df[(df['series_id'] == serie) & (df['onset'] == 1)]
        wakeup_events = df[(df['series_id'] == serie) & (df['wakeup'] == 1)]
        
        if not onset_events.empty and not wakeup_events.empty:
            # Find the indices of 'onset' and 'wakeup' events
            onset_indices = onset_events.index
            wakeup_indices = wakeup_events.index
            
            min_length = min(len(wakeup_indices), len(onset_indices))
            wakeup_indices = wakeup_indices[:min_length]
            onset_indices = onset_indices[:min_length]

            # Create a matrix of time differences
            time_diff_matrix = (df.loc[wakeup_indices, 'timestamp'].values[:, None] -
                                df.loc[onset_indices, 'timestamp'].values).astype('timedelta64[m]').astype(float)

            # Identify rows where time difference is less than period minutes
            small_breaks = np.abs(time_diff_matrix) < period

            wakeup_to_remove, onset_to_remove = np.where(small_breaks)
            df.loc[wakeup_indices[wakeup_to_remove], 'wakeup'] = 0
            df.loc[onset_indices[onset_to_remove], 'onset'] = 0
            
            # Add confidence score
            onset_events = df[(df['series_id'] == serie) & (df['onset'] == 1)]
            wakeup_events = df[(df['series_id'] == serie) & (df['wakeup'] == 1)]
            
            if not onset_events.empty and not wakeup_events.empty:
                # Find the indices of 'onset' and 'wakeup' events
                onset_indices = onset_events.index
                wakeup_indices = wakeup_events.index
                
            for index in range (0, len(onset_indices)):
                if len(wakeup_indices) > index:
                    initial_score = (1 - df.loc[onset_indices[index]:wakeup_indices[index], 'probability'].mean())
                    
                    if pd.notna(initial_score):
                        initial_score = min(initial_score + 0.05, 1.0)
                        
                        current_onset_score = df.loc[onset_indices[index], 'score']
                        current_wakeup_score = df.loc[wakeup_indices[index], 'score']
                        
                        if pd.isna(current_onset_score) or initial_score > current_onset_score:
                            df.loc[onset_indices[index], 'score'] = initial_score
                            df.loc[onset_indices[index], 'event'] = 'onset'

                        if pd.isna(current_wakeup_score) or initial_score > current_wakeup_score:
                            df.loc[wakeup_indices[index], 'score'] = initial_score
                            df.loc[wakeup_indices[index], 'event'] = 'wakeup'



def set_remove_events(data):
    data['shift_1_d_past_enmo'] = data.shift(-1440)['enmo']
    data['diff'] = abs((data['enmo'] - data['shift_1_d_past_enmo']))
    data['diff_smoothed'] = data['diff'].rolling(window=1440).mean()
    data['remove_events'] = data['diff_smoothed'] < 0.01

    return data                          


def heuristic_function(df, period_1 = 30, period_2 = 30):
    df.rename(columns={'pred_awake': 'awake'}, inplace=True)
    
    # step1: restore timestamp information
    df['year'] = 2000
    df['second'] = 0
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    
    # step 2: only keep necessary columns
    # IF MEMORY ISSUE DO A INPLACE DROP WITH SET(KEEP) ^ SET(DF.COLUMNS)
    df = df[['step', 'awake', 'series_id', 'timestamp', 'probability', 'enmo']]

    # step 3: find changes
    df['awake_changes'] = df['awake'].diff().ne(0).astype(int)

    # step 4: find onset and wakeup
    onset_mask = (df['awake_changes'] == 1) & (df['awake'] == 0)
    wakeup_mask = (df['awake_changes'] == 1) & (df['awake'] == 1)
    df['onset'] = np.where(onset_mask, 1, 0)
    df['wakeup'] = np.where(wakeup_mask, 1, 0)
    
    # drop the wakeup on row 0 
    df.loc[0, 'wakeup'] = 0

    # step 5: drop small awake windows
    delete_short_awake_windows(df, period = period_1)
    
    # step 7: if sleeping period is smaller than 30 min delete it and add confidence
    delete_short_sleep_windows(df, period = period_2)
    
    # step 8: mark events in repetitive nights
    set_remove_events(df)

    # step 9: prepare df for output
    df = df[['series_id', 'step', 'event', 'score', 'probability', 'timestamp', 'enmo', 'remove_events']] 
    df = df.dropna(subset=['event'])
    df = df.dropna(subset=['series_id'])
    df.reset_index(drop=True, inplace=True)
    df['score'] = df['score'].astype(float)
    
    return df