#!/usr/bin/env python
# coding: utf-8
# %%
from IPython.display import Image, display
import glob
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
from astroquery.skyview import SkyView
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import KMeans
from math import ceil
import asyncio
import nest_asyncio
from scipy.optimize import linear_sum_assignment
from math import radians, cos, sin, sqrt
import math
import argparse


# %%

#Module 1
def run_module1(target_input, source_name):
    def fetch_ysos(center_coord_str: str, search_radius: u.Quantity = 6 * u.arcmin) -> pd.DataFrame:
        coord = SkyCoord(center_coord_str, unit=(u.hourangle, u.deg), frame='icrs')
        Vizier.ROW_LIMIT = -1
        catalog_id = "II/360"
        columns = ["Source", "RA_ICRS", "DE_ICRS", "Jmag"]
        vizier = Vizier(columns=columns)
        result = vizier.query_region(coord, radius=search_radius, catalog=catalog_id)

        if len(result) == 0:
            print(" No YSOs found in the region.")
            return pd.DataFrame()

        yso_data = result[0]
        yso_data.rename_column("Source", "GAIA_Source_ID")
        yso_data.rename_column("RA_ICRS", "RA_deg")
        yso_data.rename_column("DE_ICRS", "DEC_deg")
        df = yso_data.to_pandas()
        df = df[["GAIA_Source_ID", "RA_deg", "DEC_deg", "Jmag"]]
        return df

    df = fetch_ysos(target_input)
    if df.empty:
        print("No YSOs found. Aborting pipeline.")
        exit(1)

    output_path = os.path.join(os.getcwd(), f"selected_ysos_{source_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"Module 1: YSOs saved to {output_path}")



# %%


def run_module2(source_name):
    csv_path = f"selected_ysos_{source_name}.csv"
    ysos_df = pd.read_csv(csv_path)
    ysos_df["coord"] = [SkyCoord(ra, dec, unit="deg") for ra, dec in zip(ysos_df["RA_deg"], ysos_df["DEC_deg"])]
    ra_center_deg = ysos_df["RA_deg"].median()
    dec_center_deg = ysos_df["DEC_deg"].median()
    coord_center = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg)
    ysos_df["offset_arcmin"] = [c.separation(coord_center).arcminute for c in ysos_df["coord"]]
    ysos_df = ysos_df[ysos_df["offset_arcmin"] <= 6]
    ysos_df["Jmag_bin"] = ysos_df["Jmag"].apply(lambda x: round(x * 2) / 2)
    magnitude_groups = ysos_df.groupby("Jmag_bin")

    initial_sublists = []
    for mag, group in magnitude_groups:
        group = group.sort_values("offset_arcmin")
        i = 0
        while i < len(group):
            chunk = group.iloc[i:i + 8].copy()
            initial_sublists.append(chunk)
            i += 8

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


def run_module3(source_name):
    grouped_path = fr"grouped_ysos_{source_name}.csv"
    df = pd.read_csv(grouped_path)
    final_sublists = []
    grouped = df.groupby('Group')  
    for name, group in grouped:
        group = group.reset_index(drop=True)  
        final_sublists.append(group)          
    ra_global_median = df['RA_deg'].median()
    dec_global_median = df['DEC_deg'].median()
    global_median_coord = SkyCoord(ra=ra_global_median*u.deg, dec=dec_global_median*u.deg)
    main_dir = os.getcwd()
    fits_dir = os.path.join(main_dir, f"fits_images_{source_name}")
    preview_dir = os.path.join(main_dir, f"fits_previews_{source_name}")
    os.makedirs(fits_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    group_summaries = []
    
    for i, group in enumerate(final_sublists, 1):
        group_coords = SkyCoord(ra=group['RA_deg'].values*u.deg, dec=group['DEC_deg'].values*u.deg)
        ra_median = group['RA_deg'].median()
        dec_median = group['DEC_deg'].median()
        median_coord = SkyCoord(ra=ra_median*u.deg, dec=dec_median*u.deg)
        offsets = [median_coord.separation(c).arcminute for c in group_coords]
        min_offset = min(offsets)
        if min_offset < 1.0:
            ra_used, dec_used = ra_global_median, dec_global_median
            coord_used = global_median_coord
            ra_used_hms=global_median_coord.ra.to_string(unit=u.hour, sep=':')
            dec_used_dms=global_median_coord.dec.to_string(unit=u.deg, sep=':', alwayssign=True)
            median_jmag = group['Jmag'].median()
            print(f" Group {i} center replaced with global median (too close)")
        else:
            ra_used, dec_used = ra_median, dec_median
            coord_used = median_coord
            ra_used_hms=median_coord.ra.to_string(unit=u.hour, sep=':')
            dec_used_dms=median_coord.dec.to_string(unit=u.deg, sep=':', alwayssign=True)
            median_jmag = group['Jmag'].median()
        group_summaries.append({
            'Group': i,
            'N_Targets': len(group),
            'RA_center': ra_used,
            'DEC_center': dec_used,
            'Median_Jmag': median_jmag
        })
        try:
            print(f" Downloading DSS Group {i} image...")
            images = SkyView.get_images(position=coord_used, survey=['DSS'], radius=0.1 * u.deg)
            if not images:
                print(f" No FITS found for Group {i}")
                continue

            fits_path = os.path.join(fits_dir, f"group_{i}_{source_name}.fits")
            images[0].writeto(fits_path, overwrite=True)

            hdul = fits.open(fits_path)
            hdu = hdul[0]
            wcs = WCS(hdu.header)
            data = hdu.data

            plt.figure(figsize=(6, 6))
            try:
                vmin = np.nanpercentile(data, 5)
                vmax = np.nanpercentile(data, 95)
            except:
                vmin, vmax = np.nanmin(data), np.nanmax(data)

            plt.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            center_x, center_y = wcs.world_to_pixel(coord_used)

            for _, star in group.iterrows():
                star_coord = SkyCoord(star['RA_deg'], star['DEC_deg'], unit='deg')
                xpix, ypix = wcs.world_to_pixel(star_coord)

                dx_arcmin = (xpix - center_x) * abs(wcs.wcs.cdelt[0]) * 60
                dy_arcmin = (ypix - center_y) * abs(wcs.wcs.cdelt[1]) * 60
                offset = np.sqrt(dx_arcmin**2 + dy_arcmin**2)

                plt.plot(xpix, ypix, 'ro', markersize=5)
                plt.text(xpix + 5, ypix + 5, f"{offset:.2f}\"", color='yellow', fontsize=8)

            x_ticks = np.linspace(-6, 6, 5)
            y_ticks = np.linspace(-6, 6, 5)
            arcmin_per_pix = abs(wcs.wcs.cdelt[0]) * 60
            xticks_pix = center_x + x_ticks / arcmin_per_pix
            yticks_pix = center_y + y_ticks / arcmin_per_pix
            plt.xticks(xticks_pix, [f"{x:.1f}" for x in x_ticks])
            plt.yticks(yticks_pix, [f"{y:.1f}" for y in y_ticks])
            plt.xlabel("ΔRA (arcmin)")
            plt.ylabel("ΔDEC (arcmin)")
            plt.title(f"Group {i} RA={ra_used_hms}, DEC={dec_used_dms}, Median Jmag={median_jmag:.2f}")
            plt.grid(True)
            plt.tight_layout()
            preview_path = os.path.join(preview_dir, f"group_{i}_preview_{source_name}.png")
            plt.savefig(preview_path, bbox_inches='tight')
            plt.close()

            print(f" Group {i} preview saved")

        except Exception as e:
            print(f" Error Group {i}: {e}")
    summary_df = pd.DataFrame(group_summaries)
    summary_out = os.path.join(main_dir, f"group_summary_{source_name}.csv")
    summary_df.to_csv(summary_out, index=False)
    print(f"\nGroup summary saved at: {summary_out}")
    preview_dir = os.path.join(main_dir, f"fits_previews_{source_name}")
    png_files = sorted(glob.glob(os.path.join(preview_dir, "*.png")))
    print(f" Displaying {len(png_files)} preview images from: {preview_dir}\n")
    for png in png_files:
        print(f"Showing: {os.path.basename(png)}")
        display(Image(filename=png))


# %%


async def run_module4(source_name):
    NUM_ARMS = 8
    FOCAL_PLANE_RADIUS = 6
    ARM_RADIUS = 6.5
    THETA_STEP = 360 / NUM_ARMS
    MIN_SAFE_DISTANCE = 0.3
    pickup_arm_centers = [(i * THETA_STEP, ARM_RADIUS) for i in range(NUM_ARMS)]
    PARKED_RADIUS = 6.0
    FIXED_END_RADIUS = 6.5
    
    def convert_grouped_targets_to_polar(grouped_yso_path, groups_summary_path):
        grouped_df = pd.read_csv(grouped_yso_path)
        summary_df = pd.read_csv(groups_summary_path)
        grouped_polar_targets = {}
        for _, group_row in summary_df.iterrows():
            group_id = int(group_row['Group'])
            origin_coord = SkyCoord(group_row['RA_center'], group_row['DEC_center'], unit=(u.deg, u.deg))
            origin_ra_hms = origin_coord.ra.to_string(unit=u.hour, sep=':')
            origin_dec_dms = origin_coord.dec.to_string(unit=u.deg, sep=':', alwayssign=True)
            median_jmag = group_row['Median_Jmag']
            group_targets = grouped_df[grouped_df['Group'] == group_id]
            polar_targets = []

            for _, row in group_targets.iterrows():
                target_coord = SkyCoord(row['RA_deg'], row['DEC_deg'], unit=(u.deg, u.deg))
                sep = origin_coord.separation(target_coord).arcmin
                pa = origin_coord.position_angle(target_coord).deg % 360
                offset_arcmin = row.get('offset_arcmin', None)

                if offset_arcmin is not None:
                    if abs(sep - offset_arcmin) > 0.05:
                        print(f" Warning: Group {group_id} Target mismatch: Calculated={sep:.3f} arcmin, Offset_Arcmin={offset_arcmin:.3f} arcmin")
                polar_targets.append((pa, sep))

            if polar_targets:
                grouped_polar_targets[group_id] = polar_targets

        return grouped_polar_targets
    
    def polar_to_cartesian(theta_deg, r):
        theta_rad = radians(theta_deg)
        return r * cos(theta_rad), r * sin(theta_rad)
    def euclidean_distance(p1, p2):
        return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def adjust_distances(targets):
        adjusted_targets = []
        for i, (theta_i, r_i) in enumerate(targets):
            is_safe = True
            for j, (theta_j, r_j) in enumerate(targets):
                if i != j:
                    distance = sqrt(r_i**2 + r_j**2 - 2 * r_i * r_j * cos(radians(theta_i - theta_j)))
                    if distance < MIN_SAFE_DISTANCE:
                        is_safe = False
                        break
            if is_safe:
                adjusted_targets.append((theta_i, r_i))
            else:
                print(f"Target {i + 1} adjusted due to proximity.")
        return adjusted_targets
    
    def assign_targets_to_positioners(targets):
        assignments = []
        targets = sorted(targets, key=lambda x: x[0])
        for i in range(NUM_ARMS):
            assignments.append((pickup_arm_centers[i][0], targets[i]))
        return assignments
    
    def assign_targets_with_parking(grouped_polar_targets):
        results = {}
        arm_cartesian = [polar_to_cartesian(theta, ARM_RADIUS) for theta, _ in pickup_arm_centers]

        for group_id, targets in grouped_polar_targets.items():
            adjusted_targets = adjust_distances(targets)
            num_targets = len(adjusted_targets)
            target_cartesian = [polar_to_cartesian(theta, r) for theta, r in adjusted_targets]
            arm_to_target = {}
            assigned_arms = []

            cost_matrix = np.zeros((NUM_ARMS, num_targets))
            for i in range(NUM_ARMS):
                for j in range(num_targets):
                    cost_matrix[i][j]= euclidean_distance(arm_cartesian[i], target_cartesian[j])

            row_index, col_index = linear_sum_assignment(cost_matrix)
            for arm_index, target_index in zip(row_index, col_index):
                assigned_arms.append(arm_index)
                arm_to_target[arm_index] = adjusted_targets[target_index]
            unused_arms = [a for a in range(NUM_ARMS) if a not in assigned_arms]
            for arm in unused_arms:
                arm_to_target[arm] = (pickup_arm_centers[arm][0], FOCAL_PLANE_RADIUS)
            results[group_id] = { "assignments": arm_to_target }
        return results
    
    async def move_arm(index, start, end, results):
        theta_start, r_start = start
        theta_end, r_end = end
        steps = 100
        theta_values = np.linspace(theta_start, theta_end, steps)
        r_values = np.linspace(r_start, r_end, steps)

        print(f"Arm {index + 1} moving from {start} to {end}...")
        for theta, r in zip(theta_values, r_values):
            results[index] = (theta % 360, r)
            await asyncio.sleep(0.05)
        print(f"Arm {index + 1} reached {end}.")
        
    async def arrange_arms_asynchronously(assignments):
        results = [(theta, PARKED_RADIUS) for theta, _ in pickup_arm_centers]
        arms_by_distance = sorted(assignments.items(), key=lambda x: x[1][1], reverse=False)
        for arm_index, target in arms_by_distance:
            await move_arm(arm_index, (pickup_arm_centers[arm_index][0], PARKED_RADIUS), target, results)
        return results
    

    def plot_positions(initial, final, ra_center_hms, dec_center_dms, median_jmag, group_id=None):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
        ax.set_title(f"Group {group_id}: RA={ra_center_hms} DEC={dec_center_dms} Median Jmag={median_jmag:.2f}", va='bottom')
        used_arm_indices = [i for i, (_, r) in enumerate(final) if r <= ARM_RADIUS]
        final_positions = {}
        for i, pos in enumerate(final):
            if isinstance(pos, (list, tuple, np.ndarray)):
                angle, radius = float(pos[0]), float(pos[1])
            else:
                angle, radius = 0.0, 0.0
            final_positions[i] = (angle, radius)

        for i, (theta_init, _) in enumerate(initial):
            theta_final, r_final = final_positions.get(i, (theta_init, PARKED_RADIUS))
            theta_rad = np.radians(theta_final)        
            is_active = i in used_arm_indices
            color = 'tab:red' if is_active else 'black'
            label = ''
            ax.plot([np.radians(theta_init), theta_rad], [FIXED_END_RADIUS, r_final], color=color, linewidth=2, marker='o', label=label)
            ax.text(theta_rad, r_final + 0.6, f"A{i+1}", fontsize=9, ha='center', va='center')
            if label:
                legend_flags[label] = True

        fov = np.linspace(0, 2 * np.pi, 360)
        r_boundary = np.full_like(fov, FOCAL_PLANE_RADIUS)
        ax.plot(fov, r_boundary, 'k--', linewidth=2.5, alpha=0.5)
        ax.set_rmax(FOCAL_PLANE_RADIUS + 2)
        ax.set_rticks(np.arange(0, FOCAL_PLANE_RADIUS + 2, 1))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        
    async def run_simulation(grouped_yso_path, groups_summary_path):
        grouped_targets = convert_grouped_targets_to_polar(grouped_yso_path, groups_summary_path)
        assigned = assign_targets_with_parking(grouped_targets)
        summary_df = pd.read_csv(groups_summary_path)
        group_row = summary_df[summary_df['Group'] == group_id].iloc[0]
        median_jmag = group_row['Median_Jmag']
        origin_coord = SkyCoord(group_row['RA_center'], group_row['DEC_center'], unit=(u.deg, u.deg))
        origin_ra_hms = origin_coord.ra.to_string(unit=u.hour, sep=':')
        origin_dec_dms = origin_coord.dec.to_string(unit=u.deg, sep=':', alwayssign=True)
        assignments = assigned[group_id]['assignments']
        final_positions = await arrange_arms_asynchronously(assignments)
        plot_positions(pickup_arm_centers, final_positions, origin_ra_hms, origin_dec_dms, median_jmag, group_id)
        results_list = []
        for i, (theta, r) in enumerate(final_positions):
            results_list.append({
                "Group": group_id,
                "Arm_ID": f"A{i+1}",
                "Angle_deg": round(theta, 2),
                "Radius_arcmin": round(r, 2),
                "Median_Jmag": median_jmag
            })
        csv_output_path = os.path.join(os.getcwd(), f"arm_positions_{source_name}.csv")
        df_out = pd.DataFrame(results_list)
        print(f" Arm position data saved to: {csv_output_path}")
        
    nest_asyncio.apply()
    grouped_yso_path = fr"grouped_ysos_{source_name}.csv"
    groups_summary_path = fr"group_summary_{source_name}.csv"
    grouped_targets = convert_grouped_targets_to_polar(grouped_yso_path, groups_summary_path)
    summary_df = pd.read_csv(groups_summary_path)
    for group_id in grouped_targets.keys():
        group_row = summary_df[summary_df['Group'] == group_id].iloc[0]
        median_jmag = group_row['Median_Jmag']
        await run_simulation(grouped_yso_path, groups_summary_path)


# %%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TA-MOONS Robotic Arm Positioning Simulation Pipeline")
    parser.add_argument("--target", required=True, help="Target coordinates in format 'HH MM SS DD MM SS'")
    parser.add_argument("--source", required=True, help="Source name (e.g., IC_348)")
    args = parser.parse_args()

    run_module1(args.target, args.source)
    run_module2(args.source)
    run_module3(args.source)
    asyncio.run(run_module4(args.source))

