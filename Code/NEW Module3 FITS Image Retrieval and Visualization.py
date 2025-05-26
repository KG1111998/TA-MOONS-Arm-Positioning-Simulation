#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
import numpy as np
import pandas as pd
from astroquery.skyview import SkyView
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
import matplotlib.pyplot as plt


# %%


source_name=input('Enter Source Name (format: IC_348): ')


# %%


grouped_path = fr"C:\Users\Sanja\Downloads\grouped_ysos_{source_name}.csv"
df = pd.read_csv(grouped_path)


# %%


final_sublists = []
grouped = df.groupby('Group')  
for name, group in grouped:
    group = group.reset_index(drop=True)  
    final_sublists.append(group)          


# %%


ra_global_median = df['RA_deg'].median()
dec_global_median = df['DEC_deg'].median()
median_coord = SkyCoord(ra=ra_global_median*u.deg, dec=dec_global_median*u.deg)


# %%


main_dir = os.getcwd()
fits_dir = os.path.join(main_dir, f"fits_images_{source_name}")
preview_dir = os.path.join(main_dir, f"fits_previews_{source_name}")
os.makedirs(fits_dir, exist_ok=True)
os.makedirs(preview_dir, exist_ok=True)
group_summaries = []


# %%
for i, group in enumerate(final_sublists, 1):
    group_coords = SkyCoord(ra=group['RA_deg'].values*u.deg, dec=group['DEC_deg'].values*u.deg)
    ra_mean = group['RA_deg'].mean()
    dec_mean = group['DEC_deg'].mean()
    mean_coord = SkyCoord(ra=ra_mean*u.deg, dec=dec_mean*u.deg)
    offsets = [mean_coord.separation(c).arcminute for c in group_coords]
    min_offset = min(offsets)
    ra_used, dec_used = ra_mean, dec_mean
    coord_used = mean_coord
    ra_used_hms=mean_coord.ra.to_string(unit=u.hour, sep=':')
    dec_used_dms=mean_coord.dec.to_string(unit=u.deg, sep=':', alwayssign=True)
    median_jmag = group['Jmag'].median()
    group_summaries.append({
        'Group': i,
        'N_Targets': len(group),
        'RA_center': ra_used,
        'DEC_center': dec_used,
        'Median_Jmag': median_jmag
    })
    try:
        print(f" Downloading 2MASS_J Group {i} image...")
        images = SkyView.get_images(position=coord_used, survey=['2MASS-J'], radius=0.1 * u.deg)
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
        plt.title(f"Source: {source_name}\n Group {i} RA={ra_used_hms}, DEC={dec_used_dms}, Median Jmag={median_jmag:.2f}")
        plt.grid(True)
        plt.tight_layout()
        preview_path = os.path.join(preview_dir, f"group_{i}_2MASS_preview_{source_name}.png")
        plt.savefig(preview_path, bbox_inches='tight')
        plt.close()

        print(f" Group {i} preview saved")

    except Exception as e:
        print(f" Error Group {i}: {e}")


# %%


summary_df = pd.DataFrame(group_summaries)
summary_out = os.path.join(main_dir, f"group_summary_{source_name}.csv")
summary_df.to_csv(summary_out, index=False)
print(f"\nGroup summary saved at: {summary_out}")
def display_previews_as_plots(preview_dir):
    png_files = sorted(glob.glob(os.path.join(preview_dir, "*.png")))
    print(f"Displaying {len(png_files)} 2MASS-J preview images from: {preview_dir}\n")
    for png in png_files:
        print(f"Showing: {os.path.basename(png)}")
        img = mpimg.imread(png)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Preview: {os.path.basename(png)}")
        plt.tight_layout()
        plt.show()
display_previews_as_plots(preview_dir)


# %%




