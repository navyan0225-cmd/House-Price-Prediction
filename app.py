import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Prediction", page_icon="🏠", layout="wide"
)

# ====================== STYLE ======================
st.markdown(
    """
<style>
.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #06b6d4 100%);
    padding: 34px 38px; border-radius: 22px; color: white; margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(79,70,229,0.35);
}
.hero h1 { margin: 0; font-size: 2.3rem; color: white; }
.hero p { margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.92; color: white; }
.result {
    background: linear-gradient(135deg, #059669, #10b981 60%, #34d399);
    padding: 26px 28px; border-radius: 20px; color: white;
    box-shadow: 0 10px 28px rgba(16,185,129,0.35); margin-bottom: 12px;
}
.result .label { font-size: 0.95rem; opacity: 0.9; letter-spacing: 1px; text-transform: uppercase; }
.result .big { font-size: 3rem; font-weight: 800; line-height: 1.1; margin: 4px 0; }
.result .sub { font-size: 1rem; opacity: 0.95; }
.kpi {
    padding: 20px 22px; border-radius: 18px; color: white; margin-bottom: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}
.kpi .t { font-size: 0.85rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }
.kpi .v { font-size: 1.9rem; font-weight: 800; margin-top: 4px; }
.k1 { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
.k2 { background: linear-gradient(135deg, #0ea5e9, #06b6d4); }
.k3 { background: linear-gradient(135deg, #f59e0b, #f97316); }
.k4 { background: linear-gradient(135deg, #ec4899, #f43f5e); }
.house {
    background: linear-gradient(160deg, #1e293b, #334155); color: white;
    padding: 20px; border-radius: 18px; margin-bottom: 16px;
    border-top: 5px solid #38bdf8; box-shadow: 0 6px 16px rgba(0,0,0,0.2);
}
.house .name { font-size: 1.25rem; font-weight: 700; }
.house .loc { font-size: 0.85rem; opacity: 0.75; margin-bottom: 8px; }
.house .price { font-size: 1.7rem; font-weight: 800; color: #4ade80; margin: 6px 0 10px 0; }
.house .row { font-size: 0.95rem; margin: 3px 0; }
.chip {
    display: inline-block; background: rgba(255,255,255,0.15); padding: 3px 10px;
    border-radius: 999px; font-size: 0.8rem; margin: 6px 6px 0 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def html(s):
  st.markdown(
      "".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True
  )


# ====================== DATA AND MODEL ======================
@st.cache_data
def load_data():
  return pd.read_csv("dataset.csv")


@st.cache_resource
def load_model():
  return joblib.load("models/house_model.pkl")


def money(x):
  x = float(x)
  if x >= 10000000:
    return f"₹{x / 10000000:.2f} Cr"
  return f"₹{x / 100000:.1f} L"


def predict(row):
  X = pd.DataFrame([row])

  # --- FIX FOR MISSING COLUMNS ERROR ---
  # If your model pipeline expects 'state' and 'district', inject them here
  if "state" not in X.columns:
    X["state"] = "Telangana"  # Change to match your training dataset state
  if "district" not in X.columns:
    X["district"] = "Hyderabad"  # Change to match your training dataset district
  # -------------------------------------

  price = float(model.predict(X)[0])
  prep = model.named_steps["prep"]
  rf = model.named_steps["rf"]
  Xt = prep.transform(X)
  if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()
  Xt = np.asarray(Xt, dtype=np.float32)
  trees = np.array([t.predict(Xt)[0] for t in rf.estimators_])
  low, high = np.percentile(trees, [10, 90])
  return price, float(low), float(high)


df = load_data()
model = load_model()

# ====================== HEADER ======================
html("""
<div class="hero">
<h1>🏠 House Price Prediction</h1>
<p>Pick a location, place and size. Get an instant price estimate and explore the housing market.</p>
</div>
""")

# ====================== SIDEBAR ======================
ALL_LOCS = sorted(df["location"].unique())
ALL_PLACES = sorted(df["place"].unique())
PMIN, PMAX = int(df["price"].min()), int(df["price"].max())


def reset_filters():
  st.session_state["f_loc"] = ALL_LOCS
  st.session_state["f_place"] = ALL_PLACES
  st.session_state["f_price"] = (PMIN, PMAX)
  st.session_state["f_bed"] = 1


if "f_loc" not in st.session_state:
  reset_filters()

st.sidebar.markdown("## 🔎 Explore filters")
st.sidebar.caption("These filters work on the Dashboard, Houses and Data tabs.")
st.sidebar.button("🔄 Reset filters", on_click=reset_filters)

sel_locations = st.sidebar.multiselect("Location", ALL_LOCS, key="f_loc")
sel_places = st.sidebar.multiselect("Place", ALL_PLACES, key="f_place")
price_range = st.sidebar.slider(
    "Price range (₹)", PMIN, PMAX, step=10000, key="f_price"
)
min_bed = st.sidebar.select_slider(
    "Minimum bedrooms", options=[1, 2, 3, 4, 5], key="f_bed"
)

result = df[
    df["location"].isin(sel_locations)
    & df["place"].isin(sel_places)
    & df["price"].between(price_range[0], price_range[1])
    & (df["bedrooms"] >= min_bed)
]

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict Price", "📊 Dashboard", "🏡 Houses", "📋 Data"]
)

# ---------------------- PREDICT ----------------------
with tab1:
  form_col, out_col = st.columns([1, 1.15], gap="large")

  with form_col:
    st.subheader("🏗️ House details")
    location = st.selectbox("📍 Location", sorted(df["location"].unique()))
    places_here = sorted(df[df["location"] == location]["place"].unique())
    place = st.selectbox("🏘️ Place", places_here)
    area = st.selectbox("🧭 Area (direction)", sorted(df["area"].unique()))

    d1, d2 = st.columns(2)
    width = d1.number_input(
        "Width (ft)", min_value=15, max_value=100, value=30, step=5
    )
    length = d2.number_input(
        "Length (ft)", min_value=15, max_value=120, value=40, step=5
    )
    square_feet = int(width * length)
    st.info(
        f"📐 Dimensions: {width} x {length} ft  =  **{square_feet:,} sq ft**"
    )

    b1, b2 = st.columns(2)
    bedrooms = b1.slider("🛏️ Bedrooms", 1, 5, 3)
    bathrooms = b2.slider("🛁 Bathrooms", 1, 3, 2)

    go = st.button("✨ Predict Price", type="primary")

  with out_col:
    if not go:
      st.subheader("💰 Your estimate")
      st.write("Fill in the details on the left and click **Predict Price**.")
    else:
      row = {
          "place": place,
          "location": location,
          "area": area,
          "square_feet": square_feet,
          "bedrooms": bedrooms,
          "bathrooms": bathrooms,
      }
      price, low, high = predict(row)

      html(f"""
            <div class="result">
            <div class="label">Estimated price</div>
            <div class="big">{money(price)}</div>
            <div class="sub">₹ {price:,.0f}</div>
            <div class="sub">Likely range: {money(low)} to {money(high)}</div>
            </div>
            """)

      pps = price / square_feet
      place_df = df[df["place"] == place]
      place_pps = (place_df["price"] / place_df["square_feet"]).mean()
      diff = (pps - place_pps) / place_pps * 100

      m1, m2, m3 = st.columns(3)
      m1.metric("Price per sq ft", f"₹{pps:,.0f}")
      m2.metric(
          f"{place} average", f"₹{place_pps:,.0f}", f"{diff:+.1f}% vs average"
      )
      m3.metric("Size", f"{square_feet:,} sq ft")

      if (
          square_feet < df["square_feet"].min()
          or square_feet > df["square_feet"].max()
      ):
        st.warning(
            f"Your size is outside the data range "
            f"({df['square_feet'].min():,} to {df['square_feet'].max():,} sq ft), "
            f"so this estimate may be less accurate."
        )

  if go:
    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")

    with c1:
      st.subheader("📈 Your house vs every place")
      comp = (
          df.assign(pps=df["price"] / df["square_feet"])
          .groupby("place", as_index=False)["pps"]
          .mean()
      )
      comp = pd.concat(
          [comp, pd.DataFrame([{"place": "⭐ Your house", "pps": pps}])],
          ignore_index=True,
      )
      comp["mine"] = comp["place"] == "⭐ Your house"
      chart = (
          alt.Chart(comp)
          .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
          .encode(
              x=alt.X("place:N", sort="-y", title=None),
              y=alt.Y("pps:Q", title="Price per sq ft (₹)"),
              color=alt.condition(
                  alt.datum.mine, alt.value("#10b981"), alt.value("#6366f1")
              ),
              tooltip=[
                  "place",
                  alt.Tooltip("pps:Q", format=",.0f", title="₹ per sq ft"),
              ],
          )
          .properties(height=320)
      )
      st.altair_chart(chart, use_container_width=True)

    with c2:
      st.subheader(f"🏘️ Similar houses in {place}")
      sim = df[
          (df["place"] == place)
          & (df["bedrooms"].between(bedrooms - 1, bedrooms + 1))
      ].copy()
      sim["gap"] = (sim["square_feet"] - square_feet).abs()
      sim = sim.sort_values("gap").head(5).drop(columns="gap")
      if sim.empty:
        st.write("No similar houses found.")
      else:
        st.dataframe(sim, use_container_width=True, hide_index=True)

# ---------------------- DASHBOARD ----------------------
with tab2:
  if result.empty:
    st.warning(
        "No houses match the filters on the left. Click **🔄 Reset filters**."
    )
  else:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
      html(
          f'<div class="kpi k1"><div class="t">Houses</div><div class="v">{len(result)}</div></div>'
      )
    with k2:
      html(
          f'<div class="kpi k2"><div class="t">Average price</div><div class="v">{money(result["price"].mean())}</div></div>'
      )
    with k3:
      html(
          f'<div class="kpi k3"><div class="t">Average size</div><div class="v">{result["square_feet"].mean():,.0f} sq ft</div></div>'
      )
    with k4:
      html(
          f'<div class="kpi k4"><div class="t">Price per sq ft</div><div class="v">₹{result["price"].sum() / result["square_feet"].sum():,.0f}</div></div>'
      )

    a, b = st.columns(2, gap="large")
    with a:
      st.subheader("💰 Average price by place")
      avg = result.groupby("place", as_index=False)["price"].mean()
      st.altair_chart(
          alt.Chart(avg)
          .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
          .encode(
              x=alt.X("place:N", sort="-y", title=None),
              y=alt.Y("price:Q", title="Average price (₹)"),
              color=alt.Color("place:N", legend=None),
              tooltip=["place", alt.Tooltip("price:Q", format=",.0f")],
          )
          .properties(height=320),
          use_container_width=True,
      )
    with b:
      st.subheader("📍 Houses by location")
      loc = result["location"].value_counts().reset_index()
      loc.columns = ["location", "count"]
      st.altair_chart(
          alt.Chart(loc)
          .mark_arc(innerRadius=70)
          .encode(
              theta="count:Q",
              color=alt.Color("location:N", title="Location"),
              tooltip=["location", "count"],
          )
          .properties(height=320),
          use_container_width=True,
      )

    a, b = st.columns(2, gap="large")
    with a:
      st.subheader("📐 Price per sq ft by place")
      pp = (
          result.assign(pps=result["price"] / result["square_feet"])
          .groupby("place", as_index=False)["pps"]
          .mean()
      )
      st.altair_chart(
          alt.Chart(pp)
          .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
          .encode(
              y=alt.Y("place:N", sort="-x", title=None),
              x=alt.X("pps:Q", title="₹ per sq ft"),
              color=alt.Color("place:N", legend=None),
              tooltip=["place", alt.Tooltip("pps:Q", format=",.0f")],
          )
          .properties(height=320),
          use_container_width=True,
      )
    with b:
      st.subheader("🛏️ Bedrooms")
      bd = result["bedrooms"].value_counts().reset_index()
      bd.columns = ["bedrooms", "count"]
      st.altair_chart(
          alt.Chart(bd)
          .mark_bar(
              cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#f59e0b"
          )
          .encode(
              x=alt.X("bedrooms:O", title="Bedrooms"),
              y=alt.Y("count:Q", title="Houses"),
              tooltip=["bedrooms", "count"],
          )
          .properties(height=320),
          use_container_width=True,
      )

    st.subheader("🔵 Size vs price")
    st.altair_chart(
        alt.Chart(result)
        .mark_circle(size=80, opacity=0.75)
        .encode(
            x=alt.X("square_feet:Q", title="Square feet"),
            y=alt.Y("price:Q", title="Price (₹)"),
            color=alt.Color("place:N", title="Place"),
            tooltip=[
                "place",
                "location",
                "area",
                "dimensions",
                "bedrooms",
                "bathrooms",
                alt.Tooltip("price:Q", format=",.0f"),
            ],
        )
        .properties(height=380)
        .interactive(),
        use_container_width=True,
    )

# ---------------------- HOUSES ----------------------
with tab3:
  if result.empty:
    st.warning(
        "No houses match the filters on the left. Click **🔄 Reset filters**."
    )
  else:
    sort_by = st.selectbox(
        "Sort by",
        ["Price: low to high", "Price: high to low", "Size: biggest first"],
    )
    if sort_by == "Price: low to high":
      shown = result.sort_values("price")
    elif sort_by == "Price: high to low":
      shown = result.sort_values("price", ascending=False)
    else:
      shown = result.sort_values("square_feet", ascending=False)

    shown = shown.head(12)
    st.caption(f"Showing {len(shown)} of {len(result)} houses")
    cols = st.columns(3)
    for i, (_, h) in enumerate(shown.iterrows()):
      with cols[i % 3]:
        html(f"""
                <div class="house">
                <div class="name">🏠 {h['place']}</div>
                <div class="loc">📍 {h['location']} · {h['area']} side · House #{h['house_id']}</div>
                <div class="price">{money(h['price'])}</div>
                <div class="row">📐 {h['dimensions']} ({h['square_feet']:,} sq ft)</div>
                <span class="chip">🛏️ {h['bedrooms']} bedrooms</span>
                <span class="chip">🛁 {h['bathrooms']} bathrooms</span>
                </div>
                """)

# ---------------------- DATA ----------------------
with tab4:
  if result.empty:
    st.warning(
        "No houses match the filters on the left. Click **🔄 Reset filters**."
    )
  else:
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download as CSV",
        result.to_csv(index=False),
        file_name="houses.csv",
        mime="text/csv",
    )