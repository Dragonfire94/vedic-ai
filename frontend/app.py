import json
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import os
API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Vedic AI Dashboard", layout="wide")
st.title("Vedic AI Dashboard")

def api_get(path, params=None, timeout=60):
    url = f"{API}{path}"
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r

def show_error(e: Exception, context: str = ""):
    msg = f"{context}\n{str(e)}".strip()
    st.error(msg)
    with st.expander("Details"):
        st.exception(e)

def copy_to_clipboard(text: str, button_label: str = "Copy", key: str = "copy_btn"):
    if st.button(button_label, key=key):
        safe = json.dumps(text)
        components.html(f"<script>navigator.clipboard.writeText({safe});</script>", height=0, width=0)
        st.success("Copied")

# ---- sidebar inputs ----
st.sidebar.header("Inputs (KST)")

# Preset load
try:
    presets = api_get("/presets", timeout=5).json().get("presets", [])
except Exception:
    presets = [{"id": "my_birth", "label": "내 출생정보", "year": 1994, "month": 12, "day": 18, "hour": 23.75, "lat": 37.5665, "lon": 126.9780}]

if "inputs" not in st.session_state:
    st.session_state.inputs = dict(year=1994, month=12, day=18, hour=23, minute=45, lat=37.5665, lon=126.9780,
                                   house_system="P", include_nodes=True, include_d9=True, language="ko", gender="male")

def apply_preset(p):
    hour_float = float(p.get("hour", 0.0))
    hh = int(hour_float)
    mm = int(round((hour_float - hh) * 60))
    st.session_state.inputs.update(year=int(p["year"]), month=int(p["month"]), day=int(p["day"]),
                                   hour=hh, minute=mm, lat=float(p["lat"]), lon=float(p["lon"]))

st.sidebar.subheader("Preset")
if presets:
    preset_ids = [p["id"] for p in presets]
    preset_labels = {p["id"]: p.get("label", p["id"]) for p in presets}
    chosen = st.sidebar.selectbox("불러올 프리셋", preset_ids, format_func=lambda x: preset_labels.get(x, x))
    if st.sidebar.button("내 출생정보 불러오기"):
        p = next((x for x in presets if x["id"] == chosen), presets[0])
        apply_preset(p)
        st.sidebar.success("Preset applied")

st.sidebar.divider()

with st.sidebar.form("input_form"):
    year = st.number_input("Year", 1900, 2100, st.session_state.inputs["year"])
    month = st.number_input("Month", 1, 12, st.session_state.inputs["month"])
    day = st.number_input("Day", 1, 31, st.session_state.inputs["day"])
    hour = st.number_input("Hour", 0, 23, st.session_state.inputs["hour"])
    minute = st.number_input("Minute", 0, 59, st.session_state.inputs["minute"])
    lat = st.number_input("Latitude", value=float(st.session_state.inputs["lat"]), format="%.6f")
    lon = st.number_input("Longitude", value=float(st.session_state.inputs["lon"]), format="%.6f")
    house_system = st.selectbox("House System", ["P", "W"], index=0 if st.session_state.inputs["house_system"] == "P" else 1)
    include_nodes = st.checkbox("Include Nodes", bool(st.session_state.inputs["include_nodes"]))
    include_d9 = st.checkbox("Include D9", bool(st.session_state.inputs["include_d9"]))
    language = st.selectbox("Language", ["ko", "en"], index=0 if st.session_state.inputs["language"] == "ko" else 1)
    gender = st.selectbox("Gender", ["male","female","other"], index={"male":0,"female":1,"other":2}.get(st.session_state.inputs.get("gender","male"),0))

    if st.form_submit_button("Apply Inputs"):
        st.session_state.inputs.update(year=int(year), month=int(month), day=int(day),
                                       hour=int(hour), minute=int(minute),
                                       lat=float(lat), lon=float(lon),
                                       house_system=house_system, include_nodes=bool(include_nodes),
                                       include_d9=bool(include_d9), language=language, gender=gender)
        st.success("Inputs updated")

inputs = st.session_state.inputs
hour_float = float(inputs["hour"]) + float(inputs["minute"]) / 60.0
params = dict(year=inputs["year"], month=inputs["month"], day=inputs["day"], hour=hour_float,
              lat=inputs["lat"], lon=inputs["lon"], house_system=inputs["house_system"],
              include_nodes=int(inputs["include_nodes"]), include_d9=int(inputs["include_d9"]),
              include_interpretation=1, gender=inputs.get("gender","male"))

# ---- state ----
if "health" not in st.session_state: st.session_state.health = None
if "chart" not in st.session_state: st.session_state.chart = None
if "ai" not in st.session_state: st.session_state.ai = None
if "ai_cache_key" not in st.session_state: st.session_state.ai_cache_key = None
if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None

# ---- actions ----
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

with c1:
    if st.button("Health Check"):
        try:
            with st.spinner("Checking backend..."):
                st.session_state.health = api_get("/health", timeout=8).json()
            st.success("Backend OK")
        except Exception as e:
            show_error(e, "Health error")

with c2:
    if st.button("Run Chart"):
        prog = st.progress(0, text="Starting chart...")
        try:
            with st.spinner("Computing chart..."):
                prog.progress(20, text="Calling /chart")
                st.session_state.chart = api_get("/chart", params=params, timeout=45).json()
                prog.progress(100, text="Done")
            st.success("Chart OK")
        except Exception as e:
            prog.empty()
            show_error(e, "Chart error")

with c3:
    if st.button("Generate AI Reading"):
        prog = st.progress(0, text="Starting AI reading...")
        try:
            with st.spinner("Generating AI Reading..."):
                p = params.copy()
                p["language"] = inputs["language"]
                p["use_cache"] = 1
                prog.progress(20, text="Calling /ai_reading")
                st.session_state.ai = api_get("/ai_reading", params=p, timeout=120).json()
                st.session_state.ai_cache_key = st.session_state.ai.get("ai_cache_key")
                prog.progress(100, text="Done")
            st.success(f"AI OK (cached={st.session_state.ai.get('cached')}, fallback={st.session_state.ai.get('fallback')})")
        except Exception as e:
            prog.empty()
            show_error(e, "AI error")

with c4:
    if st.button("Generate PDF"):
        prog = st.progress(0, text="Starting PDF...")
        try:
            with st.spinner("Building PDF..."):
                p = params.copy()
                p["include_ai"] = 1
                p["language"] = inputs["language"]
                if st.session_state.ai_cache_key:
                    p["ai_cache_key"] = st.session_state.ai_cache_key
                    p["cache_only"] = 1
                else:
                    p["cache_only"] = 1
                prog.progress(30, text="Calling /report.pdf")
                r = api_get("/report.pdf", params=p, timeout=180)
                st.session_state.pdf_bytes = r.content
                prog.progress(100, text="Done")
            st.success("PDF generated")
        except Exception as e:
            prog.empty()
            show_error(e, "PDF error")

st.divider()

if st.session_state.pdf_bytes:
    st.download_button("Download PDF", data=st.session_state.pdf_bytes, file_name="vedic_report.pdf", mime="application/pdf")
else:
    st.info("PDF not generated yet.")

tabs = st.tabs(["Chart View", "AI Reading", "Raw JSON"])

with tabs[0]:
    chart = st.session_state.chart
    if not isinstance(chart, dict):
        st.info("Run Chart first.")
    else:
        houses = chart.get("houses", {})
        planets = chart.get("planets", {})
        feats = chart.get("features", {})

        lagna = houses.get("ascendant", {}).get("rasi", {}).get("name", "—")
        moon = planets.get("Moon", {})
        moon_sign = moon.get("rasi", {}).get("name", "—")
        moon_house = moon.get("house", "—")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lagna", lagna)
        m2.metric("Moon Sign", moon_sign)
        m3.metric("Moon House", str(moon_house))
        m4.metric("House System", chart.get("input", {}).get("house_system", "—"))

        rows = []
        for pn, p in planets.items():
            rasi = p.get("rasi", {})
            nak = p.get("nakshatra", {})
            f = p.get("features", {})
            rows.append({
                "Planet": pn,
                "Longitude": float(p.get("longitude", 0.0)),
                "Sign": rasi.get("name"),
                "Deg in Sign": rasi.get("deg_in_sign"),
                "House": p.get("house"),
                "Nakshatra": f"{nak.get('name')}({nak.get('pada')})",
                "Dignity": f.get("dignity"),
                "Retro": bool(f.get("retrograde")),
                "Combust": bool(f.get("combust")),
            })
        df = pd.DataFrame(rows).sort_values("Longitude")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Yogas (MVP)")
        yogas = feats.get("yogas", []) or []
        hits = [y for y in yogas if y.get("hit")]
        if hits:
            for y in hits:
                st.success(f"{y.get('name')} — {y.get('note','')}".strip())
        else:
            st.info("No hits")

with tabs[1]:
    ai = st.session_state.ai
    if not isinstance(ai, dict):
        st.info("Generate AI Reading first.")
    else:
        st.caption(f"model={ai.get('model')} cached={ai.get('cached')} fallback={ai.get('fallback')}")
        if ai.get("error"):
            st.warning(f"backend error: {ai.get('error')}")
        st.markdown("### Summary")
        st.json(ai.get("summary", {}))
        st.markdown("### Reading")
        reading = ai.get("reading", "")
        if reading:
            copy_to_clipboard(reading, "Copy Reading", key="copy_reading")
            st.markdown(reading)
        else:
            st.info("Empty reading")

with tabs[2]:
    if isinstance(st.session_state.health, dict):
        st.markdown("**health**")
        st.json(st.session_state.health)
    if isinstance(st.session_state.chart, dict):
        st.markdown("**chart**")
        st.json(st.session_state.chart)
    if isinstance(st.session_state.ai, dict):
        st.markdown("**ai**")
        st.json(st.session_state.ai)
