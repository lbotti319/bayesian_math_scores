"""
Create missing not at random (MNAR) data
"""
import pandas as pd
import numpy as np

def create_mnar(df, a0, a1, share_yes, share_no):
    """
    df: A pandas data frame
    a0: the intercept for the G2 missingness probability
    a1: the slope for the G2 missingness probability
    share_yes: the probability of switching a yes to na
    share_no: the probability of switching a no to na
    """
  
    def scaler_to_na(x, a0, a1):
        """
        Randomly turn values from g2 to NaN
        Different rates can be applied with the intercept a0 and age coefficient a1
        """
        rand = np.random.uniform(size=1)
        prob = np.exp(a0 + a1*x)/(1 + np.exp(a0 + a1*x))
        if rand < prob:
            x = np.nan
        
        return x
    
    def higher_to_na(x, share_yes, share_no):
        """
        Randomly turn values from higher to NaN
        Different rates can be applied to "yes" and "no" with share_yes and share_no
        """
        rand = np.random.uniform(size=1)
        if x=="yes":
            if rand < share_yes:
                x = np.nan
        elif x=="no":
            if rand < share_no:
                x = np.nan
        
        return x
    
    df["G2"] = df["G2"].apply(scaler_to_na, a0=a0, a1=a1)
    
    df["higher"] = df["higher"].apply(higher_to_na, share_yes=share_yes, share_no=share_no)
  
    return df
  
