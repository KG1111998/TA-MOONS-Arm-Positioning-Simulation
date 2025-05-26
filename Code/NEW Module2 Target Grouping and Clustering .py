#!/usr/bin/env python
# coding: utf-8
# %%

# %%


source_name=input('Enter Source name (Format="V1139_cyg"): ')


# %%


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import KMeans
from math import ceil


# %%


csv_path = fr"C:\Users\Sanja\Downloads\selected_ysos_{source_name}.csv"
ysos_df = pd.read_csv(csv_path)


# %%


ra_center_deg = ysos_df['RA_deg'].median()
dec_center_deg = ysos_df['DEC_deg'].median()
coord_center = SkyCoord(ra=ra_center_deg*u.deg, dec=dec_center_deg*u.deg)
print(f" Using median center: RA={ra_center_deg}, DEC={dec_center_deg}")


# %%


ysos_df['coord'] = [SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                    for ra, dec in zip(ysos_df['RA_deg'], ysos_df['DEC_deg'])]
ysos_df['offset_arcmin'] = [c.separation(coord_center).arcminute for c in ysos_df['coord']]


# %%


ysos_df = ysos_df[(ysos_df['offset_arcmin'] > 0.3) & (ysos_df['offset_arcmin'] <= 6)]


# %%


ysos_df['Jmag_bin'] = ysos_df['Jmag'].apply(lambda x: round(x * 2) / 2)
magnitude_groups = ysos_df.groupby('Jmag_bin')


# %%


initial_sublists = []
for mag, group in magnitude_groups:
    group = group.sort_values('offset_arcmin')
    chunks = [] 
    i = 0        
    while i < len(group):
        chunk = group.iloc[i:i+8].copy()  
        chunks.append(chunk)              
        i += 8                            
    initial_sublists.extend(chunks)


# %%
def collision_check_acco_offset(sublist, min_sep=0.3):
    valid_stars = []
    flagged_stars = []
    for i in range(len(sublist)):
        current_star = sublist.iloc[i]
        current_coord = current_star["coord"]

        too_close = False
        for accepted_star in valid_stars:
            accepted_coord = accepted_star["coord"]
            separation = current_coord.separation(accepted_coord).arcminute

            if separation < min_sep:
                too_close = True
                break

        if not too_close:
            valid_stars.append(current_star)
        else:
            flagged_stars.append(current_star)

    return pd.DataFrame(valid_stars), pd.DataFrame(flagged_stars)

valid_groups = []     
flagged_list = []    

for sublist in initial_sublists:
    valid, flagged = collision_check_acco_offset(sublist)
    if not valid.empty:
        valid_groups.append(valid)
    if not flagged.empty:
        flagged_list.append(flagged)
if flagged_list:
    retry_df = pd.concat(flagged_list).reset_index(drop=True)
    retry_df["Jmag_bin"] = retry_df["Jmag"].apply(lambda x: round(x * 2) / 2)
    for _, group in retry_df.groupby("Jmag_bin"):
        group = group.sort_values("offset_arcmin")
        i = 0
        while i < len(group):
            chunk = group.iloc[i:i + 8]
            valid, _ = collision_check_acco_offset(chunk)
            if not valid.empty:
                valid_groups.append(valid)
            i=i+8

# %%


final_groups = []
for group in valid_groups:
    if len(group) > 8:
        coords_array = np.array([[c.ra.degree, c.dec.degree] for c in group["coord"]])
        k = ceil(len(group) / 8)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(coords_array)
        group["cluster"] = kmeans.labels_
        for label in range(k):
            cluster_group = group[group["cluster"] == label].drop(columns="cluster")
            final_groups.append(cluster_group)
    else:
        final_groups.append(group)


# %%


output = []
for i, group in enumerate(final_groups, 1):
    for _, row in group.iterrows():
        coord = row["coord"]
        ra_hms, dec_dms = coord.to_string("hmsdms").split()
        output.append({
            "Group": i,
            "GAIA_Source_ID": row["GAIA_Source_ID"],
            "RA_HMS": ra_hms,
            "DEC_DMS": dec_dms,
            "RA_deg": coord.ra.degree,
            "DEC_deg": coord.dec.degree,
            "Jmag": row["Jmag"],
            "Offset_arcmin": row["offset_arcmin"]
        })

df_out = pd.DataFrame(output)
df_out.to_csv(f"grouped_ysos_{source_name}.csv", index=False)
print(f"Total groups: {len(final_groups)}")
print(f"Output saved to grouped_ysos_{source_name}.csv")


# %%





# %%





# %%





# %%





# %%





# %%





# %%




