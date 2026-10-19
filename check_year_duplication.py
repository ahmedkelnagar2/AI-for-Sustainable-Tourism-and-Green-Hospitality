"""Diagnostic: are the 2018 and 2019 records independent measurements?
Usage: python check_year_duplication.py final_manuscript_dataset_v2.csv"""
import sys, pandas as pd
d = pd.read_csv(sys.argv[1], parse_dates=["DateTime"]).set_index("DateTime")
for col in ["Outdoor_Air_Temperature_C", "Outdoor_Air_Humidity_kgkg", "Moisture_Removal_Capacity_kgWater_per_h"]:
    a = d.loc["2018-03":"2018-11", col]
    b = d.loc["2019-03":"2019-11", col]
    b.index = b.index - pd.DateOffset(years=1)
    j = pd.concat([a, b], axis=1, keys=["y2018", "y2019"]).dropna()
    print(f"{col}: same-hour 2018 vs 2019 r = {j.y2018.corr(j.y2019):.3f} | "
          f"within-2018 lag-1h autocorr = {a.autocorr(1):.3f}")
print("Independent weather years give same-hour r far below the lag-1h autocorrelation.")
