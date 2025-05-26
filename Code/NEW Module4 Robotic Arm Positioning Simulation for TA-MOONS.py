#!/usr/bin/env python
# coding: utf-8
# %%

# %%


source_name=input('Enter Source Name (format: IC_348): ')


# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import nest_asyncio
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.optimize import linear_sum_assignment
from math import radians, cos, sin, sqrt
import math


# %%


NUM_ARMS = 8
FOCAL_PLANE_RADIUS = 6
ARM_RADIUS = 6.5
THETA_STEP = 360 / NUM_ARMS
MIN_SAFE_DISTANCE = 0.3
pickup_arm_centers = [(i * THETA_STEP, ARM_RADIUS) for i in range(NUM_ARMS)]
PARKED_RADIUS = 6.0
FIXED_END_RADIUS = 6.5
OBS_EPOCH = 2025.5


# %%


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
            ra_orig = row['RA_deg']
            dec_orig = row['DEC_deg']
            pm_ra = row.get('PMRA_masyr', 0.0)
            pm_dec = row.get('PMDEC_masyr', 0.0)
            epoch = row.get('Epoch_year', 2015.5)
            delta_t = OBS_EPOCH - epoch
            ra_corr = ra_orig + (pm_ra / 3.6e6) * delta_t
            dec_corr = dec_orig + (pm_dec / 3.6e6) * delta_t
            target_coord = SkyCoord(ra_corr, dec_corr, unit=(u.deg, u.deg))
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


# %%


def polar_to_cartesian(theta_deg, r):
    theta_rad = radians(theta_deg)
    return r * cos(theta_rad), r * sin(theta_rad)
def euclidean_distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# %%


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


# %%


def assign_targets_to_positioners(targets):
    assignments = []
    targets = sorted(targets, key=lambda x: x[0])
    for i in range(NUM_ARMS):
        assignments.append((pickup_arm_centers[i][0], targets[i]))
    return assignments


# %%


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

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for arm_idx, tgt_idx in zip(row_ind, col_ind):
            assigned_arms.append(arm_idx)
            arm_to_target[arm_idx] = adjusted_targets[tgt_idx]

        unused_arms = [a for a in range(NUM_ARMS) if a not in assigned_arms]
        for arm in unused_arms:
            arm_to_target[arm] = (pickup_arm_centers[arm][0], FOCAL_PLANE_RADIUS)

        results[group_id] = {
            "assignments": arm_to_target
        }

    return results


# %%


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


# %%


async def arrange_arms_asynchronously(assignments):
    results = [(theta, PARKED_RADIUS) for theta, _ in pickup_arm_centers]
    arms_by_distance = sorted(assignments.items(), key=lambda x: x[1][1], reverse=False)
    for arm_index, target in arms_by_distance:
        await move_arm(arm_index, (pickup_arm_centers[arm_index][0], PARKED_RADIUS), target, results)
    return results


# %%


def plot_positions(initial, final, ra_center_hms, dec_center_dms, median_jmag, group_id=None):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
        ax.set_title(f"Source: {source_name}\n Group {group_id}: RA={ra_center_hms} DEC={dec_center_dms} Median Jmag={median_jmag:.2f}", va='bottom')
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
        ax.set_rmax(FOCAL_PLANE_RADIUS + 1)
        ax.set_rticks(np.arange(0, FOCAL_PLANE_RADIUS + 2, 1))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.grid(True)
        plt.tight_layout()
        plt.show()


# %%


async def run_simulation(group_id, grouped_yso_path, groups_summary_path):
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
    if os.path.exists(csv_output_path):
        df_existing = pd.read_csv(csv_output_path)
        df_combined = pd.concat([df_existing, df_out], ignore_index=True)
    else:
        df_combined = df_out

    df_combined.to_csv(csv_output_path, index=False)
    print(f" Arm position data saved to: {csv_output_path}")


# %%
nest_asyncio.apply()
    grouped_yso_path = fr"grouped_ysos_{source_name}.csv"
    groups_summary_path = fr"group_summary_{source_name}.csv"
    grouped_targets = convert_grouped_targets_to_polar(grouped_yso_path, groups_summary_path)
    summary_df = pd.read_csv(groups_summary_path)
    for group_id in grouped_targets.keys():
        group_row = summary_df[summary_df['Group'] == group_id].iloc[0]
        median_jmag = group_row['Median_Jmag']
        await run_simulation(group_id, grouped_yso_path, groups_summary_path)

# %%

# %%


# nest_asyncio.apply()
# await run_simulation(
#     fr"C:\Users\Sanja\Downloads\grouped_ysos_{source_name}.csv",
#     fr"C:\Users\Sanja\Downloads\group_summary_{source_name}.csv",
#     group_to_test=int(input('Enter group to test: ')))


# %%


# nest_asyncio.apply()
# grouped_yso_path = fr"C:\Users\Sanja\Downloads\grouped_ysos_{source_name}.csv"
# groups_summary_path = fr"C:\Users\Sanja\Downloads\group_summary_{source_name}.csv"
# grouped_targets = convert_grouped_targets_to_polar(grouped_yso_path, groups_summary_path)
# for group_id in grouped_targets.keys():
#     await run_simulation(grouped_yso_path, groups_summary_path, group_id)


# %%





# %%





# %%





# %%





# %%




