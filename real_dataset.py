# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:11:41 2026

@author: anjum
"""

import pandas as pd

datafolder = r"C:\Users\anjum\OneDrive - Ottawa University\Documents\Research\New Project\Sample Data and Events"

az_2023 = pd.read_csv(datafolder + "/arizona_air_quality_2023.csv")
az_2023_events = pd.read_csv(datafolder + "/arizona_events_2023.csv")

berlin_2023 = pd.read_csv(datafolder + "/berlin_air_quality_2023.csv")
berlin_2023_events = pd.read_csv(datafolder + "/berlin_events_2023.csv")

ca_2023 = pd.read_csv(datafolder + "/california_air_quality_2023.csv")
ca_2023_events = pd.read_csv(datafolder + "/california_events_2023.csv")

paris_2023 = pd.read_csv(datafolder + "/arizona_air_quality_2023.csv")
paris_2023_events = pd.read_csv(datafolder + "/arizona_events_2023.csv")


datafolder2 = r"C:\Users\anjum\OneDrive - Ottawa University\Documents\Research\New Project\Events"

events_us_v3 =  pd.read_csv(datafolder2 + "/events_us_v3.csv")
events_eu_v2 =  pd.read_csv(datafolder2 + "/events_eu_v2.csv")
events_global =  pd.read_csv(datafolder2 + "/events_global.csv")
