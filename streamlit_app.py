import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import asyncio
import nest_asyncio
from math import radians, cos, sin, sqrt, ceil
import io

# --- App Configuration and Title ---
st.set_page_config(layout="wide", page_title="TA-MOONS Arm Simulator")
st.title("🛰️ TA-MOONS Robotic Arm Positioning Simulator")
st.write("This application runs a full simulation pipeline: it fetches live astronomical data, groups targets, and simulates the optimal robotic arm positions for observation.")

# --- Helper function to convert dataframe to CSV for download ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- The Entire Pipeline, Adapted for Streamlit ---

# MODULE 1: Fetch YSO Data
def run_module1(target_input, source_name, progress_bar):
    st.text(f"🔭 Module 1: Fetching YSOs for '{source_name}' around '{target_input}'...")
    
    @st.cache_data(ttl=3600) # Cache the data fetching for 1 hour
    def fetch_ysos_with_pm(target, radius_deg=0.1):
        Vizier.ROW_LIMIT = -1
        radius = radius_deg * u.deg
        try:
            coord = SkyCoord(target, unit=(u.hourangle, u.deg))
        except Exception:
            st.error("Invalid target coordinates format. Please use 'HH MM SS DD MM SS'.")
            return None

        v_yso = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "Jmag"])
        yso_result = v_yso.query_region(coord, radius=radius, catalog="II/360")
        if not yso_result:
            st.warning("No YSOs found in the primary catalog (II/360).")
            return pd.DataFrame()
        ysos_df = yso_result[0].to_pandas()

        source_ids = ysos_df["Source"].astype(str).tolist()
        v_gaia = Vizier(columns=["Source", "pmRA", "pmDE", "Epoch"])
        gaia_result = v_gaia.query_constraints(Source=source_ids, catalog="I/345/gaia2")
        if not gaia_result:
            st.warning("No proper motion data found in Gaia DR2 for these YSOs.")
            gaia_df = pd.DataFrame(columns=["Source", "pmRA", "pmDE", "Epoch"])
        else:
            gaia_df = gaia_result[0].to_pandas()

        combined_df = ysos_df.merge(gaia_df, on="Source", how="left")
        combined_df = combined_df.rename(columns={
            "Source": "GAIA_Source_ID", "RA_ICRS": "RA_deg", "DE_ICRS": "DEC_deg",
            "Jmag": "Jmag", "pmRA": "PMRA_masyr", "pmDE": "PMDEC_masyr", "Epoch": "Epoch_year"
        })
        return combined_df[["GAIA_Source_ID", "RA_deg", "DEC_deg", "Jmag", "PMRA_masyr", "PMDEC_masyr", "Epoch_year"]]

    df = fetch_ysos_with_pm(target_input)
    progress_bar.progress(25)
    if df is None or df.empty:
        st.error("Module 1 failed: No YSOs found.")
        return None
    
    st.success(f"✅ Module 1 complete: Found {len(df)} YSOs.")
    return df

# MODULE 2: Group Targets
def run_module2(ysos_df, progress_bar):
    st.text("\n🧩 Module 2: Grouping targets by brightness and position...")
    ysos_df["coord"] = [SkyCoord(ra, dec, unit="deg") for ra, dec in zip(ysos_df["RA_deg"], ysos_df["DEC_deg"])]

    ra_center_deg = ysos_df["RA_deg"].median()
    dec_center_deg = ysos_df["DEC_deg"].median()
    coord_center = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg)
    ysos_df["offset_arcmin"] = [c.separation(coord_center).arcminute for c in ysos_df["coord"]]

    inner_df = ysos_df[ysos_df["offset_arcmin"] <= 2.4].copy()
    outer_df = ysos_df[(ysos_df["offset_arcmin"] > 2.4) & (ysos_df["offset_arcmin"] <= 6)].copy()

    def process_zone(df_zone, min_sep):
        if df_zone.empty:
            return []
        df_zone["Jmag_bin"] = pd.cut(df_zone["Jmag"], bins=np.arange(df_zone["Jmag"].min(), df_zone["Jmag"].max() + 1, 1))
        
        def collision_check(sublist, min_sep_check):
            valid_stars = []
            if sublist.empty: return pd.DataFrame()
            coords = sublist["coord"].tolist()
            for i in range(len(coords)):
                is_valid = True
                for j in range(len(valid_stars)):
                    if coords[i].separation(valid_stars[j]["coord"]).arcmin < min_sep_check:
                        is_valid = False
                        break
                if is_valid:
                    valid_stars.append(sublist.iloc[i])
            return pd.DataFrame(valid_stars)

        final_groups = []
        # Group by brightness, then split into chunks of 8
        for _, group in df_zone.groupby("Jmag_bin"):
            group = group.sort_values("offset_arcmin")
            for i in range(0, len(group), 8):
                chunk = group.iloc[i:i+8]
                valid_chunk = collision_check(chunk, min_sep)
                if not valid_chunk.empty:
                    final_groups.append(valid_chunk)
        return final_groups

    final_inner_groups = process_zone(inner_df, min_sep=0.9)
    final_outer_groups = process_zone(outer_df, min_sep=0.3)
    final_groups = final_inner_groups + final_outer_groups

    if not final_groups:
        st.error("Module 2 Failed: No valid groups could be formed.")
        return None

    output = []
    for i, group in enumerate(final_groups, 1):
        for _, row in group.iterrows():
            output.append({
                "Group": i, "GAIA_Source_ID": row["GAIA_Source_ID"], "RA_deg": row["RA_deg"],
                "DEC_deg": row["DEC_deg"], "Jmag": row["Jmag"], "offset_arcmin": row["offset_arcmin"]
            })
    
    df_out = pd.DataFrame(output)
    progress_bar.progress(50)
    st.success(f"✅ Module 2 complete: Formed {len(final_groups)} groups.")
    return df_out

# MODULE 3: Create Group Summary
def run_module3(grouped_df, progress_bar):
    st.text("\n📊 Module 3: Creating group summary...")
    group_summaries = []
    for group_id, group_data in grouped_df.groupby('Group'):
        group_summaries.append({
            'Group': group_id, 'N_Targets': len(group_data),
            'RA_center': group_data['RA_deg'].mean(), 'DEC_center': group_data['DEC_deg'].mean(),
            'Median_Jmag': group_data['Jmag'].median()
        })
    summary_df = pd.DataFrame(group_summaries)
    progress_bar.progress(60)
    st.success("✅ Module 3 complete: Summary created.")
    return summary_df

# MODULE 4: Arm Positioning Simulation (UPDATED)
def run_module4(grouped_df, summary_df, source_name, progress_bar):
    st.text("\n🤖 Module 4: Simulating robotic arm positions for all groups...")
    
    # --- Constants and Helper functions for Module 4 ---
    NUM_ARMS = 8
    FOCAL_PLANE_RADIUS = 6.0
    ARM_RADIUS = 6.5
    pickup_arm_centers = [(i * (360 / NUM_ARMS), ARM_RADIUS) for i in range(NUM_ARMS)]
    
    def polar_to_cartesian(theta_deg, r):
        theta_rad = radians(theta_deg)
        return r * cos(theta_rad), r * sin(theta_rad)

    def euclidean_distance(p1, p2):
        return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    all_arm_positions = []
    num_groups = len(summary_df)
    if num_groups == 0:
        st.warning("Module 4: No groups to process.")
        return pd.DataFrame()

    # --- NEW: Create a subplot grid for all group plots ---
    st.subheader("Arm Position Simulation Plots")
    cols = 3  # Number of plots per row
    rows = ceil(num_groups / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), subplot_kw={'polar': True})
    axes = axes.flatten() # Make the axes array 1D for easy iteration

    # --- Main Simulation Loop ---
    for i, (_, group_row) in enumerate(summary_df.iterrows()):
        ax = axes[i]
        group_id = int(group_row['Group'])
        origin_coord = SkyCoord(group_row['RA_center'], group_row['DEC_center'], unit="deg")
        group_targets_df = grouped_df[grouped_df['Group'] == group_id]
        
        polar_targets = []
        for _, row in group_targets_df.iterrows():
            target_coord = SkyCoord(row['RA_deg'], row['DEC_deg'], unit="deg")
            sep = origin_coord.separation(target_coord).arcmin
            pa = origin_coord.position_angle(target_coord).deg
            if sep <= FOCAL_PLANE_RADIUS:
                polar_targets.append((pa, sep))
        
        if not polar_targets:
            ax.set_title(f"Group {group_id}\nNo valid targets", color='red')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            continue

        target_cartesian = [polar_to_cartesian(t, r) for t, r in polar_targets]
        arm_cartesian = [polar_to_cartesian(t, ARM_RADIUS) for t, _ in pickup_arm_centers]
        
        cost_matrix = np.full((NUM_ARMS, len(polar_targets)), 1e6)
        for arm_idx in range(NUM_ARMS):
            for target_idx in range(len(polar_targets)):
                cost_matrix[arm_idx, target_idx] = euclidean_distance(arm_cartesian[arm_idx], target_cartesian[target_idx])

        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        final_positions = [(t, FOCAL_PLANE_RADIUS) for t, _ in pickup_arm_centers]
        assigned_targets_count = 0
        for arm_i, target_i in zip(row_idx, col_idx):
            if cost_matrix[arm_i, target_i] < 1e5:
                final_positions[arm_i] = polar_targets[target_i]
                assigned_targets_count += 1
        
        for arm_idx in range(NUM_ARMS):
            all_arm_positions.append({
                "Group": group_id, "Arm_ID": f"A{arm_idx+1}",
                "Angle_deg": round(final_positions[arm_idx][0], 2),
                "Radius_arcmin": round(final_positions[arm_idx][1], 2)
            })

        # --- Plotting on the dedicated subplot (ax) ---
        ax.set_title(f"Group {group_id} ({assigned_targets_count}/{len(polar_targets)} assigned)")
        ax.fill_between(np.linspace(0, 2 * np.pi, 100), 0, FOCAL_PLANE_RADIUS, color='lightgray', alpha=0.3)

        for arm_idx, (theta_init, r_init) in enumerate(pickup_arm_centers):
            theta_final, r_final = final_positions[arm_idx]
            is_assigned = r_final <= FOCAL_PLANE_RADIUS
            color = 'blue' if is_assigned else 'black'
            linestyle = '-' if is_assigned else '--'
            ax.plot([radians(theta_init), radians(theta_final)], [r_init, r_final], color=color, linestyle=linestyle, marker='o', markersize=4, lw=1.5)
            ax.text(radians(theta_final), r_final + 0.8, f"A{arm_idx+1}", ha='center', va='center', fontsize=7)
        
        ax.set_rmax(ARM_RADIUS + 1)
        ax.set_theta_zero_location("N")
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # --- Finalize and display the plot grid ---
    # Hide any unused subplots
    for j in range(num_groups, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(f"Arm Positions for {source_name}", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    st.pyplot(fig)
    plt.close(fig)

    progress_bar.progress(100)
    st.success(f"✅ Module 4 complete: All {num_groups} groups simulated and plotted.")
    return pd.DataFrame(all_arm_positions)


# --- Streamlit UI Elements ---
with st.sidebar:
    st.header("Simulation Inputs")
    target_coords = st.text_input("Target Coordinates (HH MM SS DD MM SS)", "03 44 30.7 +32 00 17")
    source_name = st.text_input("Source Name (no spaces)", "IC_348")
    run_button = st.button("🚀 Run Full Simulation")

if run_button:
    if not target_coords or not source_name:
        st.warning("Please provide both target coordinates and a source name.")
    else:
        st.info("Simulation in progress... This may take a minute or two depending on the number of targets.")
        progress_bar = st.progress(0, text="Starting pipeline...")
        
        # Run pipeline
        df_mod1 = run_module1(target_coords, source_name, progress_bar)
        
        if df_mod1 is not None and not df_mod1.empty:
            df_mod2 = run_module2(df_mod1, progress_bar)
            
            if df_mod2 is not None and not df_mod2.empty:
                df_mod3 = run_module3(df_mod2, progress_bar)
                
                # The simulation is now synchronous, no asyncio needed for this part
                df_mod4 = run_module4(df_mod2, df_mod3, source_name, progress_bar)

                # --- Display Results and Download Buttons ---
                st.header("📂 Results and Downloads")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Fetched YSOs (Module 1)")
                    st.dataframe(df_mod1, height=200)
                    st.download_button("Download YSO Data (CSV)", convert_df_to_csv(df_mod1), f"selected_ysos_{source_name}.csv", "text/csv")

                    st.subheader("Group Summary (Module 3)")
                    st.dataframe(df_mod3, height=200)
                    st.download_button("Download Group Summary (CSV)", convert_df_to_csv(df_mod3), f"group_summary_{source_name}.csv", "text/csv")
                
                with col2:
                    st.subheader("Grouped Targets (Module 2)")
                    st.dataframe(df_mod2, height=200)
                    st.download_button("Download Grouped Data (CSV)", convert_df_to_csv(df_mod2), f"grouped_ysos_{source_name}.csv", "text/csv")

                    st.subheader("Final Arm Positions (Module 4)")
                    st.dataframe(df_mod4, height=200)
                    st.download_button("Download Arm Positions (CSV)", convert_df_to_csv(df_mod4), f"arm_positions_{source_name}.csv", "text/csv")
