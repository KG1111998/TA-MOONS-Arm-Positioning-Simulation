# MODULE 1 – Target-Based YSO Selection Interface

* It input sky coordinates and searches for Young Stellar Objects (YSOs) around that region using Gaia catalog.

* Sky is mapped using celestial coordinates: RA and DEC in HH MM SS DD MM SS, which are converted to degrees using SkyCoord.

* Catalogs like MARTON+2019 (II/360) list star positions, brightness (Jmag), and more.

* Radius = 6 arcmin search area (~0.1° in the sky).

 Code Flow:
* Convert input to degrees.

* Query Vizier: "II/360" = Gaia DR2 YSO catalog.

* Display results in HTML table.

* Export to: selected_ysos_{source}.csv.

* Output File: CSV with: GAIA_Source_ID, RA_deg, DEC_deg, Jmag

![Slide2_module1](https://github.com/user-attachments/assets/12c79052-b4ba-477a-883f-b4ca1fa4e3ce)

# MODULE 2 – Target Grouping and Clustering

* Takes all YSOs found in Module 1 and groups them into valid batches of up to 8 per group (1 arm per target), avoiding collision.

* Collision check ensures robotic arms cannot be closer than 1 arcmin to each other.

* Group clustering based on:

  * offset_arcmin from center (distance)

  * Jmag (brightness binning)

  * Separation distance and angular offset

Code Logic:

* Calculate angular separation using SkyCoord.separation().

* Filter out stars too close or far from center (< 6 arcmin).

* Group by brightness (Jmag) and distance (offset_arcmin).

* Apply collision_check() and KMeans clustering if >8 stars.

* Compute θ (angle) and r (distance) from group center.

Output File: grouped_ysos_{source}.csv — includes each YSO’s group, RA/Dec, offsets.

# MODULE 3 – FITS Image Retrieval and WCS Annotation

* Retrieves DSS sky images for each group and annotates star positions on the image.

* Uses SkyView service to fetch DSS images.

* Uses WCS (World Coordinate System) headers in FITS to:

  * Convert RA/Dec → pixels

  * Annotate each star’s location

  * Measure angular offset between them

Code Flow:
* Read grouped_ysos.csv.

* For each group, calculate median center.

* Download DSS image via SkyView.get_images().

* Annotate star positions and save preview plots.

Output Files: group_summary_{source}.csv — Group-wise RA/Dec centers.

Folder: fits_images_* and fits_previews_* — plots per group.

# MODULE 4 – Robotic Arm Positioning & Simulation

Takes groups + images, and simulates 8 robotic arms to move safely toward their assigned target.

* 8 robotic arms = each placed at 45° apart (360°/8).

* Each arm is like a polar robotic link (like a clock hand).

* Arms must reach specific (θ, r) positions.

* Avoid collision using minimum distance check and safe parking.

Code Logic:
* Convert target coordinates (RA/Dec) to polar (θ, r) from group center.

* Adjust targets to avoid collision.

* Compute arm ↔ target distance matrix.

* Apply Hungarian algorithm for best match.

* Simulate motion using asyncio and animate movement.

* Plot final configuration using matplotlib polar plots.

Final Output: Polar plot for each group showing arm assignments, and interactive motion using async updates.

![Slide1_module4](https://github.com/user-attachments/assets/a39a4e63-514d-4947-adbd-9e1ed60d6c83)













