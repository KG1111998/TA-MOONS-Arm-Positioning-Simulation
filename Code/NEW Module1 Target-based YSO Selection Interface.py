#!/usr/bin/env python
# coding: utf-8
# %%

# %%


#Imported Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipyaladin import Aladin
from ipywidgets import Layout, Box, Button, HTML
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import os


# %%


target_input = input('Enter target coordinates (format: HH MM SS DD MM SS): ')


# %%


source_name=input('Enter Source Name (format: IC_348): ')


# %%


aladin = Aladin(
    layout=Layout(width="60%", height="500px"),
    target=target_input,
    fov=0.2,
    show_projection_control=False,
    show_fullscreen_control=True,)


# %%


def fetch_ysos_with_pm(target_input, radius_deg=0.1):
    Vizier.ROW_LIMIT = -1
    radius = radius_deg * u.deg
    coord = SkyCoord(target_input, unit=(u.hourangle, u.deg))

    v_yso = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "Jmag"])
    yso_result = v_yso.query_region(coord, radius=radius, catalog="II/360")
    if not yso_result:
        raise ValueError("No YSOs found in II/360.")
    ysos_df = yso_result[0].to_pandas()

    source_ids = ysos_df["Source"].astype(str).tolist()

    v_gaia = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS","Jmag","pmRA", "pmDE", "Epoch"])
    gaia_result = v_gaia.query_constraints(Source=source_ids, catalog="I/345/gaia2")
    if not gaia_result:
        raise ValueError("No Gaia DR2 matches found.")
    gaia_df = gaia_result[0].to_pandas()

    combined_df = ysos_df.merge(gaia_df, on="Source", suffixes=("", "_GAIA"))
    combined_df = combined_df.rename(columns={
        "Source": "GAIA_Source_ID",
        "RA_ICRS": "RA_deg",
        "DE_ICRS": "DEC_deg",
        "Jmag": "Jmag",
        "pmRA": "PMRA_masyr",
        "pmDE": "PMDEC_masyr",
        "Epoch": "Epoch_year"
    })

    columns_to_keep = ["GAIA_Source_ID", "RA_deg", "DEC_deg", "Jmag", "PMRA_masyr", "PMDEC_masyr", "Epoch_year"]
    return combined_df[columns_to_keep]

df = fetch_ysos_with_pm(target_input)
if df.empty:
    print("No YSOs found.")
    exit(1)


# %%


table_output = HTML(layout=Layout(height="500px", overflow="auto"))
selected_ysos_df = pd.DataFrame()

def auto_selection():
    global selected_ysos_df
    center_coord = target_input
    yso_table = fetch_ysos_with_pm(center_coord)
    if len(yso_table) == 0:
        table_output.value = "<b>No YSOs found.</b>"
        return

    records = []
    html = '<table border="1" style="border-collapse:collapse;">'
    html += "<tr><th>GAIA_Source_ID</th><th>RA (deg)</th><th>DEC (deg)</th><th>Jmag</th><th>pmRA</th><th>pmDE</th><th>Epoch</th></tr>"
    
    for row in yso_table:
        record = {
            "GAIA_Source_ID": row["GAIA_Source_ID"],
            "RA_deg": float(row["RA_deg"]),
            "DEC_deg": float(row["DEC_deg"]),
            "Jmag": float(row["Jmag"]),
            "pmRA": float(row["PMRA_masyr"]),
            "pmDE": float(row["PMDEC_masyr"]),
            "Epoch": float(row["Epoch_year"])
        }
        records.append(record)
        html += f"<tr><td>{record['GAIA_Source_ID']}</td><td>{record['RA_deg']:.6f}</td><td>{record['DEC_deg']:.6f}</td><td>{record['Jmag']:.2f}</td><td>{record['pmRA']}</td><td>{record['pmDE']}</td><td>{record['Epoch']}</td></tr>"
    
    html += "</table>"
    table_output.value = html

    selected_ysos_df = pd.DataFrame(records)
    csv_path = os.path.join(os.getcwd(), f"selected_ysos_{source_name}.csv")
    selected_ysos_df.to_csv(csv_path, index=False)
    print(f"Selected YSOs saved to: {csv_path}")


# %%


select_button = Button(description="Auto-select YSOs (6 arcmin)")
select_button.on_click(lambda _: auto_selection())


# %%


ysos_table = fetch_ysos_with_pm(target_input)
if len(ysos_table) > 0:
    aladin.add_table(ysos_table)
layout = Box(
    children=[aladin, select_button, table_output],
    layout=Layout(display="flex", flex_flow="row", align_items="stretch", width="100%")
)
layout


# %%





# %%

# %%
