import pandas as pd

class BaselineModel:
    """
    A simple baseline model for generating onset and wakeup predictions, 
    based on the time.

    """
    def __init__(self, df: pd.DataFrame):
        self.__df = df
    
    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    def predict(self, onset_time: str = "22:00:00", wakeup_time: str = "06:30:00", score: float = 0.5) -> pd.DataFrame:
        """
        Generate submission predictions based on fixed onset and wakeup times.

        Parameters:
        - onset_time (str): The onset time in "HH:mm:ss" format.
        - wakeup_time (str): The wakeup time in "HH:mm:ss" format.
        - score (float): The score assigned to the predictions.

        Returns:
        - pd.DataFrame: The submission DataFrame with row_id, series_id, step, event, and score.
        """
        
        onset = self.df[self.df["timestamp"].str.slice(11, 19) == onset_time].set_index("series_id")["step"]
        wakeup = self.df[self.df["timestamp"].str.slice(11, 19) == wakeup_time].set_index("series_id")["step"]
        
        submission = pd.concat([
            onset.reset_index().assign(event="onset"),
            wakeup.reset_index().assign(event="wakeup"),
        ])
        
        submission["score"] = score
        submission.sort_values(["series_id", "step"], ascending=[0, 1], inplace=True)
        submission.reset_index(drop=True, inplace=True)
        submission["row_id"] = submission.index
        submission = submission[["row_id", "series_id", "step", "event", "score"]]

        return submission
    
