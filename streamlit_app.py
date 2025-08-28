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
from astroquery.skyview import SkyView
from astropy.io import fits
from astropy.wcs import WCS
import io

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="TA-MOONS Arm Simulator")
st.title("🛰️ TA-MOONS Robotic Arm Positioning Simulator")
st.write("A direct web adaptation of the TA-MOONS pipeline. This version faithfully reproduces the logic, outputs, and plotting styles of the original script.")

# --- Helper function for downloads ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ==============================================================================
# MODULE 1: Fetch YSO Data
# ==============================================================================
def run_module1(target_input, source_name):
    st.header("Module 1: Fetching YSO Data")
    log_expander = st.expander("Show Module 1 Logs")

    @st.cache_data(ttl=3600)
    def fetch_ysos_with_pm(target, radius_deg=0.1):
        Vizier.ROW_LIMIT = -1
        radius = radius_deg * u.deg
        try:
            coord = SkyCoord(target, unit=(u.hourangle, u.deg))
        except Exception as e:
            return None, f"Invalid target coordinates format. Error: {e}"

        log_expander.text(f"Querying Vizier catalog II/360 around {coord.to_string('hmsdms')}...")
        v_yso = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "Jmag"])
        yso_result = v_yso.query_region(coord, radius=radius, catalog="II/360")
        if not yso_result:
            return None, "Module 1 Error: No YSOs found in the primary catalog (II/360)."
        ysos_df = yso_result[0].to_pandas()
        
        source_ids = ysos_df["Source"].astype(str).tolist()
        v_gaia = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS","Jmag","pmRA", "pmDE", "Epoch"])
        gaia_result = v_gaia.query_constraints(Source=source_ids, catalog="I/345/gaia2")
        if not gaia_result:
             return None, "Module 1 Error: No Gaia DR2 matches found for the YSOs."
        gaia_df = gaia_result[0].to_pandas()

        combined_df = ysos_df.merge(gaia_df, on="Source", suffixes=("", "_GAIA"))
        combined_df = combined_df.rename(columns={
            "Source": "GAIA_Source_ID", "RA_ICRS": "RA_deg", "DE_ICRS": "DEC_deg",
            "Jmag": "Jmag", "pmRA": "PMRA_masyr", "pmDE": "PMDEC_masyr", "Epoch": "Epoch_year"
        })
        columns_to_keep = ["GAIA_Source_ID", "RA_deg", "DEC_deg", "Jmag", "PMRA_masyr", "PMDEC_masyr", "Epoch_year"]
        return combined_df[columns_to_keep], "Success"

    df, message = fetch_ysos_with_pm(target_input)
    
    if df is None:
        st.error(message)
        return None
    
    log_expander.text(f"Query successful. Found {len(df)} YSOs.")
    st.success(f"✅ Module 1 Complete: YSO data for {source_name} fetched successfully.")
    return df

# ==============================================================================
# MODULE 2: Group Targets
# ==============================================================================
def run_module2(ysos_df, source_name, min_sep_inner, min_sep_outer):
    st.header("Module 2: Grouping Targets")
    log_expander = st.expander("Show Module 2 Logs")

    ysos_df["coord"] = [SkyCoord(ra, dec, unit="deg") for ra, dec in zip(ysos_df["RA_deg"], ysos_df["DEC_deg"])]
    ra_center_deg = ysos_df["RA_deg"].median()
    dec_center_deg = ysos_df["DEC_deg"].median()
    coord_center = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg)
    ysos_df["offset_arcmin"] = [c.separation(coord_center).arcminute for c in ysos_df["coord"]]
    
    inner_df = ysos_df[ysos_df["offset_arcmin"] <= 2.4].copy()
    outer_df = ysos_df[(ysos_df["offset_arcmin"] > 2.4) & (ysos_df["offset_arcmin"] <= 6)].copy()

    def process_zone(df_zone, min_sep):
        # This function preserves the original script's logic
        if df_zone.empty: return []
        df_zone["Jmag_bin"] = df_zone["Jmag"].apply(lambda x: round(x))
        initial_sublists = []
        for _, group in df_zone.groupby("Jmag_bin"):
            group = group.sort_values("offset_arcmin")
            for i in range(0, len(group), 8): initial_sublists.append(group.iloc[i:i + 8].copy())
        
        def collision_check_acco_offset(sublist, min_sep_check=min_sep):
            valid_stars, flagged_stars = [], []
            if sublist.empty: return pd.DataFrame(), pd.DataFrame()
            for i in range(len(sublist)):
                current_star = sublist.iloc[i]
                too_close = any(current_star["coord"].separation(vs["coord"]).arcminute < min_sep_check for vs in valid_stars)
                if not too_close: valid_stars.append(current_star)
                else: flagged_stars.append(current_star)
            return pd.DataFrame(valid_stars) if valid_stars else pd.DataFrame(), pd.DataFrame(flagged_stars) if flagged_stars else pd.DataFrame()

        valid_groups, flagged_list = [], []
        for sublist in initial_sublists:
            valid, flagged = collision_check_acco_offset(sublist)
            if not valid.empty: valid_groups.append(valid)
            if not flagged.empty: flagged_list.append(flagged)
        
        if flagged_list:
            retry_df = pd.concat(flagged_list).reset_index(drop=True)
            retry_df["Jmag_bin"] = retry_df["Jmag"].apply(lambda x: round(x))
            for _, group in retry_df.groupby("Jmag_bin"):
                group = group.sort_values("offset_arcmin")
                for i in range(0, len(group), 8):
                    valid, _ = collision_check_acco_offset(group.iloc[i:i+8])
                    if not valid.empty: valid_groups.append(valid)
        final_groups = []
        for group in valid_groups:
            if len(group) > 8:
                coords_array = np.array([[c.ra.degree, c.dec.degree] for c in group["coord"]])
                k = ceil(len(group) / 8)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(coords_array)
                group["cluster"] = kmeans.labels_
                for label in range(k): final_groups.append(group[group["cluster"] == label].drop(columns="cluster"))
            else: final_groups.append(group)
        return final_groups

    log_expander.text(f"Processing inner zone with {min_sep_inner}' separation...")
    final_inner_groups = process_zone(inner_df, min_sep=min_sep_inner)
    log_expander.text(f"Processing outer zone with {min_sep_outer}' separation...")
    final_outer_groups = process_zone(outer_df, min_sep=min_sep_outer)
    final_groups = final_inner_groups + final_outer_groups

    output = []
    for i, group in enumerate(final_groups, 1):
        for _, row in group.iterrows():
            coord = row["coord"]
            ra_hms, dec_dms = coord.to_string("hmsdms").split()
            output.append({"Group": i, "GAIA_Source_ID": row["GAIA_Source_ID"], "RA_HMS": ra_hms, "DEC_DMS": dec_dms, "RA_deg": coord.ra.degree, "DEC_deg": coord.dec.degree, "Jmag": row["Jmag"], "Offset_arcmin": row["offset_arcmin"]})
    df_out = pd.DataFrame(output)
    st.success(f"✅ Module 2 Complete: {len(final_groups)} groups formed.")
    return df_out

# ==============================================================================
# MODULE 3: Create Group Summary & FITS Previews (UPDATED with faster download)
# ==============================================================================
def run_module3(grouped_df, source_name):
    st.header("Module 3: Creating Group Summary & FITS Previews")
    log_expander = st.expander("Show Module 3 Logs")

    final_sublists = [group for _, group in grouped_df.groupby('Group')]
    group_summaries = []
    num_groups = len(final_sublists)
    if num_groups == 0: return pd.DataFrame()

    cols, rows = 3, ceil(num_groups / 3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()
    
    status_text = st.empty()
    for i, group in enumerate(final_sublists, 1):
        status_text.info(f"Downloading and plotting FITS preview {i}/{num_groups}...")
        ax = axes[i-1]
        ra_mean, dec_mean = group['RA_deg'].mean(), group['DEC_deg'].mean()
        mean_coord = SkyCoord(ra=ra_mean*u.deg, dec=dec_mean*u.deg)
        ra_used_hms, dec_used_dms = mean_coord.ra.to_string(unit=u.hour, sep=':'), mean_coord.dec.to_string(unit=u.deg, sep=':', alwayssign=True)
        median_jmag = group['Jmag'].median()
        group_summaries.append({'Group': i, 'N_Targets': len(group), 'RA_center': ra_mean, 'DEC_center': dec_mean, 'Median_Jmag': median_jmag})

        try:
            log_expander.text(f"Downloading 2MASS-J Group {i} image...")
            # --- SPEEDUP FIX: Request a smaller 300x300 pixel image for a faster preview ---
            images = SkyView.get_images(position=mean_coord, survey=['2MASS-J'], radius=0.1 * u.deg, pixels=300)
            if not images:
                log_expander.warning(f"No FITS found for Group {i}")
                ax.text(0.5, 0.5, f"Group {i}\nImage Not Found", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([])
                continue
            
            hdu, wcs, data = images[0][0], WCS(images[0][0].header), images[0][0].data
            try: vmin, vmax = np.nanpercentile(data, 5), np.nanpercentile(data, 95)
            except: vmin, vmax = np.nanmin(data), np.nanmax(data)
            ax.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            center_x, center_y = wcs.world_to_pixel(mean_coord)
            for _, star in group.iterrows(): ax.plot(*wcs.world_to_pixel(SkyCoord(star['RA_deg'], star['DEC_deg'], unit='deg')), 'ro', markersize=3)
            
            x_ticks, y_ticks = np.linspace(-6, 6, 5), np.linspace(-6, 6, 5)
            arcmin_per_pix_x, arcmin_per_pix_y = abs(wcs.wcs.cdelt[0]) * 60, abs(wcs.wcs.cdelt[1]) * 60
            xticks_pix, yticks_pix = center_x + x_ticks / arcmin_per_pix_x, center_y + y_ticks / arcmin_per_pix_y
            ax.set_xticks(xticks_pix, [f"{x:.1f}" for x in x_ticks]); ax.set_yticks(yticks_pix, [f"{y:.1f}" for y in y_ticks])
            ax.set_xlabel("ΔRA (arcmin)"); ax.set_ylabel("ΔDEC (arcmin)")
            ax.set_title(f"Group {i} RA={ra_used_hms}\nDEC={dec_used_dms}, Jmag={median_jmag:.2f}", fontsize=9)
            ax.grid(True, alpha=0.5)
        except Exception as e:
            log_expander.error(f"Error processing Group {i}: {e}")
            ax.text(0.5, 0.5, f"Group {i}\nDownload Failed", ha='center', va='center', color='red'); ax.set_xticks([]); ax.set_yticks([])

    status_text.empty()
    for j in range(num_groups, len(axes)): axes[j].set_visible(False)
    fig.suptitle(f"FITS Previews for {source_name}", fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.subheader("FITS Image Previews"); st.pyplot(fig); plt.close(fig)
    summary_df = pd.DataFrame(group_summaries)
    st.success("✅ Module 3 Complete.")
    return summary_df

# ==============================================================================
# MODULE 4: Arm Positioning Simulation
# ==============================================================================
async def run_module4(grouped_df, summary_df, source_name):
    st.header("Module 4: Simulating Robotic Arm Positions")
    NUM_ARMS, FOCAL_PLANE_RADIUS, ARM_RADIUS = 8, 6.0, 6.5
    THETA_STEP, OBS_EPOCH, PARKED_RADIUS, FIXED_END_RADIUS = 360/NUM_ARMS, 2025.5, 6.0, 6.5
    pickup_arm_centers = [(i * THETA_STEP, ARM_RADIUS) for i in range(NUM_ARMS)]

    def convert_grouped_targets_to_polar(grouped_yso_df, groups_summary_df):
        grouped_polar_targets = {}
        for _, group_row in groups_summary_df.iterrows():
            group_id = int(group_row['Group'])
            origin_coord = SkyCoord(group_row['RA_center'], group_row['DEC_center'], unit="deg")
            group_targets = grouped_yso_df[grouped_yso_df['Group'] == group_id]
            polar_targets = []
            for _, row in group_targets.iterrows():
                pm_ra, pm_dec, epoch = row.get('PMRA_masyr', 0.0), row.get('PMDEC_masyr', 0.0), row.get('Epoch_year', 2015.5)
                delta_t = OBS_EPOCH - epoch
                ra_corr, dec_corr = row['RA_deg'] + (pm_ra / 3.6e6) * delta_t, row['DEC_deg'] + (pm_dec / 3.6e6) * delta_t
                target_coord = SkyCoord(ra_corr, dec_corr, unit="deg")
                sep, pa = origin_coord.separation(target_coord).arcmin, origin_coord.position_angle(target_coord).deg % 360
                polar_targets.append((pa, sep))
            if polar_targets: grouped_polar_targets[group_id] = polar_targets
        return grouped_polar_targets

    def polar_to_cartesian(theta_deg, r): return r * cos(radians(theta_deg)), r * sin(radians(theta_deg))
    def euclidean_distance(p1, p2): return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    grouped_polar_targets = convert_grouped_targets_to_polar(grouped_df, summary_df)
    all_arm_positions_list, num_groups = [], len(grouped_polar_targets)
    if num_groups == 0: return pd.DataFrame()

    st.subheader("Arm Position Simulation Plots")
    cols, rows = 3, ceil(num_groups / 3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), subplot_kw={'polar': True})
    axes = axes.flatten()

    for i, group_id in enumerate(grouped_polar_targets.keys()):
        ax, targets = axes[i], grouped_polar_targets[group_id]
        arm_cartesian = [polar_to_cartesian(t, r) for t, r in pickup_arm_centers]
        target_cartesian = [polar_to_cartesian(t, r) for t, r in targets]
        cost_matrix = np.array([[euclidean_distance(ac, tc) for tc in target_cartesian] for ac in arm_cartesian])
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        assignments, assigned_arms = {}, set()
        for arm_i, target_i in zip(row_idx, col_idx): assignments[arm_i], assigned_arms.add(arm_i) = targets[target_i], True
        for arm_i in range(NUM_ARMS):
            if arm_i not in assigned_arms: assignments[arm_i] = (pickup_arm_centers[arm_i][0], PARKED_RADIUS)
        final_positions = [assignments.get(i, (pickup_arm_centers[i][0], PARKED_RADIUS)) for i in range(NUM_ARMS)]
        
        group_row = summary_df[summary_df['Group'] == group_id].iloc[0]
        median_jmag = group_row['Median_Jmag']
        origin_coord = SkyCoord(group_row['RA_center'], group_row['DEC_center'], unit="deg")
        origin_ra_hms, origin_dec_dms = origin_coord.ra.to_string(unit=u.hour, sep=':'), origin_coord.dec.to_string(unit=u.deg, sep=':', alwayssign=True)
        
        ax.set_title(f"Group {group_id}: RA={origin_ra_hms} DEC={origin_dec_dms}\nMedian Jmag={median_jmag:.2f}", va='bottom', fontsize=10)
        theta_grid = np.linspace(0, 2 * np.pi, 360)
        ax.fill_between(theta_grid, 0, 3, color='skyblue', alpha=0.3, label='Inner Zone (≤3\')')
        ax.fill_between(theta_grid, 3, FOCAL_PLANE_RADIUS, color='orange', alpha=0.15, label='Outer Zone (3\'–6\')')
        
        for arm_idx, (theta_init, _) in enumerate(pickup_arm_centers):
            theta_final, r_final = final_positions[arm_idx]
            color = 'black'
            if r_final <= FOCAL_PLANE_RADIUS: color = 'blue' if r_final <= 2.4 else 'green'
            ax.plot([radians(theta_init), radians(theta_final)], [FIXED_END_RADIUS, r_final], color=color, linewidth=1.5, marker='o', markersize=4)
            ax.text(radians(theta_final), r_final + 0.6, f"A{arm_idx + 1}", fontsize=7, ha='center', va='center')

        ax.plot(theta_grid, np.full(360, FOCAL_PLANE_RADIUS), 'k--', linewidth=2, alpha=0.5)
        ax.set_rmax(FOCAL_PLANE_RADIUS + 1); ax.set_rticks(np.arange(0, FOCAL_PLANE_RADIUS + 2, 2))
        ax.set_theta_zero_location("N"); ax.set_theta_direction(1); ax.grid(True, alpha=0.5)
        ax.tick_params(axis='x', labelsize=8); ax.tick_params(axis='y', labelsize=8)

        for arm_idx in range(NUM_ARMS):
            theta_global, r = final_positions[arm_idx]
            theta_relative = (theta_global - pickup_arm_centers[arm_idx][0] + 540) % 360 - 180
            all_arm_positions_list.append({"Group": group_id, "Arm_ID": f"A{arm_idx + 1}", "Angle_deg": round(theta_global, 2), "Radius_arcmin": round(r, 2), "Relative_Angle_deg": round(theta_relative, 2), "Median_Jmag": median_jmag, "Within_Reach": "Yes" if r <= FOCAL_PLANE_RADIUS else "No"})
            
    for j in range(num_groups, len(axes)): axes[j].set_visible(False)
    fig.suptitle(f"Arm Position Simulations for {source_name}", fontsize=20, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); st.pyplot(fig); plt.close(fig)
    st.success("✅ Module 4 Complete."); return pd.DataFrame(all_arm_positions_list)

# --- Streamlit UI and Execution Flow ---
with st.sidebar:
    st.header("Simulation Inputs")
    target_coords = st.text_input("Target Coordinates (HH MM SS DD MM SS)", "03 44 30.7 +32 00 17")
    source_name = st.text_input("Source Name (no spaces)", "IC_348")
    
    st.header("Grouping Parameters (Module 2)")
    sep_options = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    min_sep_inner = st.selectbox("Inner Zone Min Separation (arcmin)", options=sep_options, index=2, help="Default: 0.9")
    min_sep_outer = st.selectbox("Outer Zone Min Separation (arcmin)", options=sep_options, index=0, help="Default: 0.3")

    run_button = st.button("🚀 Run Full Simulation")

if run_button:
    if not target_coords or not source_name:
        st.warning("Please provide both target coordinates and a source name.")
    else:
        st.info("Simulation in progress... This may take a minute or two.")
        df_mod1 = run_module1(target_coords, source_name)
        if df_mod1 is not None:
            df_mod2 = run_module2(df_mod1, source_name, min_sep_inner, min_sep_outer)
            if not df_mod2.empty:
                df_mod3 = run_module3(df_mod2, source_name)
                nest_asyncio.apply()
                df_mod4 = asyncio.run(run_module4(df_mod2, df_mod3, source_name))

                st.header("📂 Final Results and Downloads")
                st.subheader("Fetched YSOs"); st.dataframe(df_mod1)
                st.download_button("Download Selected YSOs (CSV)", convert_df_to_csv(df_mod1), f"selected_ysos_{source_name}.csv", "text/csv")
                st.subheader("Grouped Targets"); st.dataframe(df_mod2)
                st.download_button("Download Grouped YSOs (CSV)", convert_df_to_csv(df_mod2), f"grouped_ysos_{source_name}.csv", "text/csv")
                st.subheader("Group Summary"); st.dataframe(df_mod3)
                st.download_button("Download Group Summary (CSV)", convert_df_to_csv(df_mod3), f"group_summary_{source_name}.csv", "text/csv")
                if not df_mod4.empty:
                    st.subheader("Final Arm Positions"); st.dataframe(df_mod4)
                    st.download_button("Download Arm Positions (CSV)", convert_df_to_csv(df_mod4), f"arm_positions_{source_name}.csv", "text/csv")
