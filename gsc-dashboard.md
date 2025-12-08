
---

# **Fortinet GSC Dashboard — Monthly Data Update & Deployment Guide**

**Project folder:** `/root/apps/gsc-dashboard`
**Service name:** `gsc-dashboard`
**Virtual environment:** `venv/`
**Data folder:** `/root/apps/gsc-dashboard/Data/`
**Domain:** [https://fortinet-gsc-dashboard.leadwalnut.com/](https://fortinet-gsc-dashboard.leadwalnut.com/)
**Code to get the data:** Use this [script](https://colab.research.google.com/drive/1P6X63ivEGwkb0hdGy9Yix6OYm-Fb8xHC?usp=sharing) to obtain the updated data.
---

## **Step 1 — SSH into the server**

```bash
ssh root@<server-ip>
cd ~/apps/gsc-dashboard
```

---

## **Step 2 — Activate Python virtual environment**

```bash
source venv/bin/activate
```

---

## **Step 3 — Update / Replace Data Files**

* Place new CSV files in the `Data/` folder:

```bash
cd Data
# e.g., upload gsc_data_day_by_day_3.csv, efax_internal_html.csv
# Replace existing files with the updated versions
```

* **Important:** Keep filenames consistent.
* Example `scp` command from local to server:

```bash
scp -i "C:\Users\Naveen\.ssh\id_rsa" "C:\Users\Naveen\Downloads\gsc_data_day_by_day_3.csv" root@<server-ip>:/root/apps/gsc-dashboard/Data/
```

---

## **Step 4 — Optional: Test app locally**

```bash
cd ~/apps/gsc-dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

* Open: http://<server-ip>:8501
* Verify the new data is displayed correctly
* Stop the test with `CTRL+C`

---

## **Step 5 — Restart the service to load new data**

Streamlit loads data at startup, so restart systemd service:

```bash
sudo systemctl restart gsc-dashboard
sudo systemctl status gsc-dashboard  # verify status = active (running)
```

---

## **Step 6 — Verify the dashboard on your domain**

Open:

```
https://fortinet-gsc-dashboard.leadwalnut.com/
```

* New data should appear on the dashboard
* If the site shows **502**, check service status:

```bash
sudo systemctl status gsc-dashboard
sudo ss -lptn 'sport = :8501'
```
---




