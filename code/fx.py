import pandas as pd
import numpy as np
DEBUG = False
# JONAS' CODE

def split_timestamp_on_train(train):
    train['timestamp'] = train['timestamp'].str[:-5]

    train['hour'] = pd.to_numeric(train['timestamp'].str[-8:-6])
    train['minute'] = pd.to_numeric(train['timestamp'].str[-5:-3])
    train['seconds'] = pd.to_numeric(train['timestamp'].str[-2:])

    train['day'] = pd.to_numeric(train['timestamp'].str[-11:-9])
    train['month'] = pd.to_numeric(train['timestamp'].str[-14:-12])
    train['year'] = pd.to_numeric(train['timestamp'].str[-20:-15])

    train.drop('timestamp', axis=1, inplace=True)

    return train

def engineer_anglez_features(train, periods):
    return engineer_sensor_features(train, periods, 'anglez')

def engineer_enmo_features(train, periods):
    return engineer_sensor_features(train, periods, 'enmo')

def engineer_sensor_features(train, periods, feature_name):
    train[f"{feature_name}_abs"] = abs(train[feature_name]).astype("float32")

    for period in periods:
        train = engineer_sensor_periods_features(train, period, feature_name)

    return train

def engineer_sensor_periods_features(train, periods, feature_name):
    train[f"{feature_name}_rolling_mean_{periods}"] = train[feature_name].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_rolling_sum_{periods}"] = train[feature_name].rolling(periods,center=True).sum().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_rolling_max_{periods}"] = train[feature_name].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_rolling_min_{periods}"] = train[feature_name].rolling(periods,center=True).min().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_rolling_std_{periods}"] = train[feature_name].rolling(periods,center=True).std().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_rolling_median_{periods}"] = train[feature_name].rolling(periods,center=True).median().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_rolling_variance_{periods}"] = train[feature_name].rolling(periods,center=True).var().fillna(method="bfill").fillna(method="ffill").astype('float32')

    quantiles = [0.25, 0.75]
    for quantile in quantiles:
        train[f"{feature_name}_rolling_{int(quantile * 100)}th_percentile_{periods}"] = train[feature_name].rolling(periods,center=True).quantile(quantile).fillna(method="bfill").fillna(method="ffill").astype('float32')

    train[f"{feature_name}_diff_{periods}"] = train[feature_name].diff(periods=periods).fillna(method="bfill").astype('float32')

    train[f"{feature_name}_diff_rolling_mean_{periods}"] = train[f"{feature_name}_diff_{periods}"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_diff_rolling_sum_{periods}"] = train[f"{feature_name}_diff_{periods}"].rolling(periods,center=True).sum().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_diff_rolling_max_{periods}"] = train[f"{feature_name}_diff_{periods}"].rolling(periods,center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_diff_rolling_min_{periods}"] = train[f"{feature_name}_diff_{periods}"].rolling(periods,center=True).min().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_diff_rolling_std_{periods}"] = train[f"{feature_name}_diff_{periods}"].rolling(periods,center=True).std().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_diff_rolling_median_{periods}"] = train[f"{feature_name}_diff_{periods}"].rolling(periods,center=True).median().fillna(method="bfill").fillna(method="ffill").astype('float32')
    train[f"{feature_name}_diff_rolling_variance_{periods}"] = train[f"{feature_name}_diff_{periods}"].rolling(periods,center=True).var().fillna(method="bfill").fillna(method="ffill").astype('float32')

    quantiles = [0.25, 0.75]
    for quantile in quantiles:
        train[f"{feature_name}_diff_rolling_{int(quantile * 100)}th_percentile_{periods}"] = train[f"{feature_name}_diff_{periods}"].rolling(periods,center=True).quantile(quantile).fillna(method="bfill").fillna(method="ffill").astype('float32')

    return train

def get_train_series(series, path):
    train_series = pd.read_parquet(path, filters=[('series_id','=',series)])
    
    train_series = split_timestamp_on_train(train_series)

    train_series = engineer_anglez_features(train_series, [12, 60])

    train_series = engineer_enmo_features(train_series, [12, 60])

    return train_series

def methodFromJonas(batch_size, test_series_path):
    train_series = pd.read_parquet(test_series_path, columns=['series_id'])
    series_ids = train_series['series_id'].unique()

    batch = pd.DataFrame([])

    for series_id in series_ids:
        if batch.empty:
            batch = get_train_series(series_id, test_series_path)
        else:
            batch = pd.concat([batch, get_train_series(series_id, test_series_path)])

        if len(batch) >= batch_size:
            yield batch
            batch = pd.DataFrame([])  # because of unbound error
            #del batch
    yield batch


## RAHEL'S CODE

def delete_small_breaks(df):
    series = df.series_id.unique().tolist()
    for serie in series:
        #onset_events = df[(df['series_id'] == serie) & (df['onset'] == 1)]
        wakeup_events = df[(df['series_id'] == serie) & (df['wakeup'] == 1)]
        
        for index, wakeup_row in wakeup_events.iterrows():
            next_onset_rows = df.iloc[index:].loc[df['onset'] == 1]
            if not next_onset_rows.empty:
                next_onset_index = next_onset_rows.iloc[0].name
                time_diff = (df.loc[next_onset_index]['timestamp'] - df.loc[index]['timestamp']).total_seconds() / 60
                if time_diff < 30:
                    df.loc[index, 'wakeup'] = 0
                    df.loc[next_onset_index, 'onset'] = 0
            else:
                if DEBUG: print("No row found with 'onset' == 1 after index", index)

def delete_too_small_periods(df):
    series = df.series_id.unique().tolist()
    for serie in series:
        onset_events = df[(df['series_id'] == serie) & (df['onset'] == 1)]
        #wakeup_events = df[(df['series_id'] == serie) & (df['wakeup'] == 1)]   
        
        for index, onset_row in onset_events.iterrows():
            #print(df.loc[index].name)
            next_wakeup_rows = df.iloc[index:].loc[df['wakeup'] == 1]
            if not next_wakeup_rows.empty:
                next_wakeup_index = next_wakeup_rows.iloc[0].name
                #print(next_onset_index)
                time_diff = (df.loc[next_wakeup_index]['timestamp'] - df.loc[index]['timestamp']).total_seconds() / 60
                if time_diff < 30:
                    df.loc[index, 'onset'] = 0
                    df.loc[next_wakeup_index, 'wakeup'] = 0
            else:
                if DEBUG: print("No row found with 'onset' == 1 after index", index)

def add_column_night(df):
    series = df.series_id.unique().tolist()
    for serie in series:
        counter = 1
        df.loc[df['series_id'] == serie, 'night'] = 1
        next_index = 0
        
        # Check if there are any occurrences for series_id and hour is 15
        while (df[(df['series_id'] == serie) & (df.index > next_index + 60)]['timestamp'].dt.hour == 15).any():
            counter += 1
            next_index = df.loc[(df['series_id'] == serie) & (df.index > next_index + 60)].loc[df['timestamp'].dt.hour == 15].index[0]
            df.loc[(df['series_id'] == serie) & (df.index >= next_index), 'night'] = counter
        
        if DEBUG: print("No occurrences found that meet the conditions.")

def one_sleep_widow(df):
    series = df.series_id.unique().tolist()
    df['event'] = np.NAN
    for serie in series:
        nights = df[df['series_id'] == serie]['night'].unique().tolist()
        for night in nights:
            max_window_duration = pd.Timedelta(0)
            current_window_start = None
            current_window_end = None
            onset_index = None
            wakeup_index = None
            
            for index, row in df[(df['series_id'] == serie) & (df['night'] == night)].iterrows():
                if row['onset'] == 1:
                    # Start of a potential sleeping window
                    current_window_start = row['timestamp']
                    onset_index = index
                elif row['wakeup'] == 1 and current_window_start is not None:
                    # End of a potential sleeping window
                    current_window_end = row['timestamp']
                    wakeup_index = index
                    window_duration = current_window_end - current_window_start
                    if window_duration > max_window_duration:
                        max_window_duration = window_duration
                        max_onset_index = onset_index
                        max_wakeup_index = wakeup_index

            if DEBUG: print(f"The longest sleeping window duration is: {max_window_duration}")
            if DEBUG: print(f"Starts at index: {max_onset_index}, Ends at index: {max_wakeup_index}")
            # Check if the longest window duration is longer than 30 minutes
            if max_window_duration > pd.Timedelta(minutes=30):
                # Assign 'onset' and 'wakeup' events based on indexes
                df.loc[max_onset_index, 'event'] = 'onset'
                df.loc[max_wakeup_index, 'event'] = 'wakeup'

def calculate_score(row):
    if row['event'] == 'onset' or row['event'] == 'wakeup':
        return 1
    else:
        return None

def add_score(df):
    
    df['score'] = df.apply(calculate_score, axis=1)
    
    for index, row in df.iterrows():
        if row['score'] == 1:
            score_value = 1.0
            event_value = row['event']
            for i in range(index, -1, -1):
                df.at[i, 'score'] = score_value
                df.at[i, 'event'] = event_value
                score_value -= 0.1
                if score_value < 0:
                    break
            
    for index, row in df.iterrows():
        if row['score'] == 1:
            score_value = 1.0
            event_value = row['event']
            for i in range(index, len(df)):
                df.at[i, 'score'] = score_value
                df.at[i, 'event'] = event_value
                score_value -= 0.1
                if score_value < 0:
                    break
                
    df['score'] = df['score'].map('{:.1f}'.format)

def heuristic_function(df):
    df.rename(columns={'pred_awake': 'awake'}, inplace=True)
    # step1: restore timestamp information
    #restore_timestamp(df)
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'seconds']])
    #df.drop(['year', 'month', 'day', 'hour', 'minute', 'seconds'], axis=1, inplace=True)

    # step 2: only keep necessary columns
    # IF MEMORY ISSUE DO A INPLACE DROP WITH SET(KEEP) ^ SET(DF.COLUMNS)
    df = df[['step', 'awake', 'series_id', 'timestamp']]

    # step 2.1: fill missing rows
    #processed = missing_rows(df)

    # step 3: binning the data
    #result = binning(processed)
    df.set_index('timestamp', inplace=True)
    df = df.groupby('series_id').resample('1T').agg({'step': 'first', 'awake': 'mean'}).reset_index() # we use mean and round to get whether the person is awake or not
    df = df[df['awake'].notna()] # drop rows where awake is na, meaning that there is no data for this minute
    df.loc[:, 'awake'] = df['awake'].round().astype(int)


    # step 4: find changes
    df['awake_changes'] = df['awake'].diff().ne(0).astype(int)

    # # step 5: find onset and wakeup
    #add_onset_add_wakeup(df)
    onset_mask = (df['awake_changes'] == 1) & (df['awake'] == 0)
    wakeup_mask = (df['awake_changes'] == 1) & (df['awake'] == 1)
    df['onset'] = np.where(onset_mask, 1, 0)
    df['wakeup'] = np.where(wakeup_mask, 1, 0)

    # step 6: if break between two sleeping windows is smaller than 30 min make one window out of it
    delete_small_breaks(df)

    # step 7: if sleeping period is smaller than 30 min delete it
    delete_too_small_periods(df)

    # step 8: get a column for night
    add_column_night(df)

    # step 9: only keep 1 sleeping window per night
    one_sleep_widow(df)

    # step 10: add score around the onset and wakeup
    df.to_csv('heu.csv') # TODO: for debugging, remove later
    add_score(df)

    # step 11: delete not necessary rows and columns
    df.drop(['timestamp', 'awake', 'awake_changes', 'wakeup', 'onset', 'night'], axis = 1, inplace=True)
    df =df.dropna(subset=['event'])

    # step 12: reset index
    df.reset_index(drop=True, inplace=True)
    #df['row_id'] = df.reset_index().index # i have to do this at the end
    
    # step 13 floats for score
    df['score'] = df['score'].astype(float)

    return df




















if __name__ == "__main__":
    pass