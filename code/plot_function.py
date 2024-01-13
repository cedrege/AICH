import matplotlib.pyplot as plt
def plot_whole_series(series_id, train_series, train_events, pred, color_sensor_data = 'lightgrey', color_onset = 'blue', color_wakeup = 'green', font_size=20):
    # Filter based on the series_id
    sample_serie = train_series[train_series['series_id'] == series_id]
    sample_events = train_events[train_events['series_id'] == series_id]
    pred_events = pred[pred['series_id'] == series_id]

        
    # Helper function to plot data
    def plot_data(data, ylabel, events=True):
        plt.figure(figsize=(22, 4))
        plt.plot(sample_serie['step'], sample_serie[data], label=data, linewidth=0.5, color = color_sensor_data)

        # Plot predicted events
        for event, color in zip(['onset', 'wakeup'], [color_onset, color_wakeup]):
            pred_event = pred_events.loc[pred_events['event'] == event, 'step'].dropna()
            for pred_time in pred_event:
                plt.axvline(x=pred_time, ymin=0.3, ymax=0.7, color=color, linestyle=':', label=f'pred {event}', linewidth=3)

        if events:
            # Plot true events
            for event, color in zip(['onset', 'wakeup'], [color_onset, color_wakeup]):
                sample_event = sample_events.loc[sample_events['event'] == event, 'step'].dropna()
                for true_time in sample_event:
                    plt.axvline(x=true_time, color=color, linestyle='-', label=f'true {event}', linewidth=1.5)

        # Optimize legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = set(labels)
        unique_handles = [handles[labels.index(label)] for label in unique_labels]

        plt.legend(unique_handles, unique_labels, fontsize=font_size)
        plt.xlabel('step', fontsize=font_size, labelpad=15)
        plt.ylabel(ylabel, fontsize=font_size, labelpad=20)
        plt.title(f'{ylabel} of series {series_id}', fontsize=font_size)
        plt.xticks(fontsize=font_size)  
        plt.yticks(fontsize=font_size)
        
        if data == 'enmo':
            plt.ylim(0, 3.5)  
        elif data == 'anglez':
            plt.ylim(-100, 100)  
            
        plt.tick_params(axis='both', which='both', length=10, width=2)
        plt.show()
    

    plot_data('enmo', 'ENMO value', events=True)
    plot_data('anglez', 'anglez value', events=True)



