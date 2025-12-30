import streamlit as st
import fastf1
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# -------------------------------
# FIBONACCI HEAP IMPLEMENTATION
# -------------------------------
class FibNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.parent = None
        self.child = None
        self.left = self
        self.right = self
        self.degree = 0
        self.mark = False

class FibHeap:
    def __init__(self):
        self.min = None
        self.total_nodes = 0

    def insert(self, key, value):
        node = FibNode(key, value)
        self._merge_with_root_list(node)
        if not self.min or node.key < self.min.key:
            self.min = node
        self.total_nodes += 1
        return node

    def _merge_with_root_list(self, node):
        if self.min is None:
            node.left = node.right = node
            self.min = node
        else:
            node.left = self.min
            node.right = self.min.right
            self.min.right.left = node
            self.min.right = node

    def extract_min(self):
        z = self.min
        if z:
            if z.child:
                children = []
                child = z.child
                while True:
                    children.append(child)
                    child = child.right
                    if child == z.child:
                        break
                for c in children:
                    self._merge_with_root_list(c)
                    c.parent = None
            z.left.right = z.right
            z.right.left = z.left
            if z == z.right:
                self.min = None
            else:
                self.min = z.right
                self._consolidate()
            self.total_nodes -= 1
        return z

    def _consolidate(self):
        import math
        max_degree = int(math.log2(self.total_nodes or 1)) + 2
        A = [None] * (max_degree + 1)
        nodes = []
        x = self.min
        if x:
            nodes.append(x)
            x = x.right
            while x != self.min:
                nodes.append(x)
                x = x.right
        for w in nodes:
            d = w.degree
            while A[d]:
                y = A[d]
                if w.key > y.key:
                    w, y = y, w
                self._link(y, w)
                A[d] = None
                d += 1
            A[d] = w
        self.min = None
        for i in A:
            if i:
                if not self.min or i.key < self.min.key:
                    self.min = i

    def _link(self, y, x):
        y.left.right = y.right
        y.right.left = y.left
        y.parent = x
        if not x.child:
            x.child = y
            y.left = y.right = y
        else:
            y.left = x.child
            y.right = x.child.right
            x.child.right.left = y
            x.child.right = y
        x.degree += 1
        y.mark = False

    def decrease_key(self, node, new_key):
        if new_key > node.key:
            raise ValueError("new key is greater than current key")
        node.key = new_key
        y = node.parent
        if y and node.key < y.key:
            self._cut(node, y)
            self._cascading_cut(y)
        if node.key < self.min.key:
            self.min = node

    def _cut(self, x, y):
        if y.child == x:
            if x.right != x:
                y.child = x.right
            else:
                y.child = None
        x.left.right = x.right
        x.right.left = x.left
        y.degree -= 1
        self._merge_with_root_list(x)
        x.parent = None
        x.mark = False

    def _cascading_cut(self, y):
        z = y.parent
        if z:
            if not y.mark:
                y.mark = True
            else:
                self._cut(y, z)
                self._cascading_cut(z)

    def is_empty(self):
        return self.min is None

# -------------------------------
# Driver & Simulator
# -------------------------------
class Driver:
    def __init__(self, name, fuel, tire_wear, weather_risk=0):
        self.name = name
        self.fuel = fuel
        self.tire_wear = tire_wear
        self.weather_risk = weather_risk
        self.priority = self.compute_priority()
        self.node = None
        self.progress = random.random()
        self.speed = 0.0
        self.pit_count = 0
        self.pit_history = []

    def compute_priority(self):
        return (1.0 / max(0.1, self.fuel)) + self.tire_wear / 100.0 + self.weather_risk

class PitStopSimulator:
    def __init__(self):
        self.heap = FibHeap()
        self.driver_nodes = {}

    def add_driver(self, driver):
        driver.node = self.heap.insert(driver.priority, driver)
        self.driver_nodes[driver.name] = driver.node

    def update_driver(self, driver):
        node = driver.node
        new_priority = driver.priority
        if node is None:
            self.add_driver(driver)
            return
        if new_priority < node.key:
            self.heap.decrease_key(node, new_priority)
        else:
            self.add_driver(driver)

    def next_in_pit(self):
        if self.heap.is_empty():
            return None
        node = self.heap.extract_min()
        driver = node.value
        if driver.name in self.driver_nodes:
            del self.driver_nodes[driver.name]
        return driver

# -------------------------------
# Track Visualization
# -------------------------------
def generate_track_shape(gp_name, num_points=500):
    theta = np.linspace(0, 2*np.pi, num_points)
    
    if gp_name == "Monaco":
        r = 1.0 + 0.05 * np.sin(5*theta) + 0.02 * np.sin(10*theta)  # uske i zakrivljene ulice
    elif gp_name == "Bahrain":
        r = 1.0 + 0.1 * np.sin(3*theta) + 0.05 * np.sin(7*theta)     # brza staza sa pravcima
    elif gp_name == "Monza":
        r = 1.0 + 0.3 * np.sin(2*theta) + 0.1 * np.sin(5*theta)      # dugi pravci i par krivina
    elif gp_name == "Spa-Francorchamps":
        r = 1.0 + 0.2 * np.sin(4*theta) + 0.15 * np.sin(7*theta)     # duga staza, blage krivine
    elif gp_name == "Brazil":
        r = 1.0 + 0.15 * np.sin(3*theta) + 0.1 * np.sin(6*theta)     # kombinacija krivina i pravaca
    else:
        r = 1.0 + 0.2 * np.sin(3*theta) + 0.1 * np.sin(5*theta)      # default
    
    track_x = r * np.cos(theta)
    track_y = r * np.sin(theta)
    return track_x, track_y


def position_on_track(progress):
    idx = int(progress * len(track_x)) % len(track_x)
    return track_x[idx], track_y[idx]

def generate_track_figure_icons(drivers, size=800):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=track_x,
            y=track_y,
            mode="lines",
            line=dict(width=4, color="gray"),
            hoverinfo="none",
            showlegend=False,
        ))

    if drivers:
        priorities = np.array([d.priority for d in drivers])
        pmin, pmax = priorities.min(), priorities.max()
        rng = (pmax - pmin) if (pmax - pmin) > 0 else 1.0
        norm_colors = [(d.priority - pmin) / rng for d in drivers]
        viridis_map = px.colors.sequential.Viridis_r

        for idx, d in enumerate(drivers):
            x, y = position_on_track(d.progress)
            color_idx = int(norm_colors[idx] * (len(viridis_map) - 1))
            car_color = viridis_map[color_idx]

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(size=40, color=car_color, line=dict(color="black", width=1)),
                    text=["🏎️"],
                    textfont=dict(size=30),
                    textposition="middle center",
                    showlegend=False,
                ))

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y + 0.12],
                    text=[f"<b>{d.name}</b>"],
                    mode="text",
                    textposition="top center",
                    textfont=dict(color="black", size=14),
                    showlegend=False,
                ))

            trail_x, trail_y = [], []
            for t in range(5):
                trail_progress = (d.progress - 0.01 * (t + 1)) % 1.0
                tx, ty = position_on_track(trail_progress)
                trail_x.append(tx)
                trail_y.append(ty)

            fig.add_trace(
                go.Scatter(
                    x=trail_x,
                    y=trail_y,
                    mode="lines",
                    line=dict(width=3, color=car_color),
                    showlegend=False,
                ))

    fig.update_layout(
        autosize=True,
        xaxis=dict(visible=False, scaleanchor="y", autorange=True),
        yaxis=dict(visible=False, autorange=True),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig

def generate_priority_bar_figure(drivers):
    if not drivers:
        fig = go.Figure()
        fig.update_layout(title="No drivers to display", height=400)
        return fig

    sorted_drivers = sorted(drivers, key=lambda d: d.priority)
    df = pd.DataFrame({"Driver": [d.name for d in sorted_drivers], "Priority": [d.priority for d in sorted_drivers]})

    priorities = np.array([d.priority for d in sorted_drivers])
    pmin, pmax = priorities.min(), priorities.max()
    rng = (pmax - pmin) if (pmax - pmin) > 0 else 1.0
    norm_colors = [(p - pmin) / rng for p in priorities]
    viridis_map = px.colors.sequential.Viridis_r
    bar_colors = [viridis_map[int(nc * (len(viridis_map) - 1))] for nc in norm_colors]

    fig = px.bar(
        df,
        x="Priority",
        y="Driver",
        orientation="h",
        range_x=[0, max(1.0, df["Priority"].max() * 1.2)],
        color=df["Driver"],
        color_discrete_map={d.name: c for d, c in zip(sorted_drivers, bar_colors)},
    )

    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig

def generate_pit_timeline(drivers, total_laps):
    fig = go.Figure()
    for idx, d in enumerate(drivers):
        fig.add_trace(
            go.Scatter(x=[0, total_laps], y=[idx, idx], mode="lines", line=dict(width=1, color="lightgray"), showlegend=False, hoverinfo="none")
        )
    for idx, d in enumerate(drivers):
        if d.pit_history:
            fig.add_trace(
                go.Scatter(
                    x=d.pit_history,
                    y=[idx] * len(d.pit_history),
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    showlegend=False,
                    hovertemplate=f"{d.name} — Lap" + "%{x}<extra></extra>",
                )
            )
    fig.update_layout(
        height=300,
        width=900,
        yaxis=dict(tickvals=list(range(len(drivers))), ticktext=[d.name for d in drivers], autorange="reversed"),
        xaxis=dict(range=[0, total_laps], title="Lap"),
        margin=dict(l=100, r=20, t=20, b=20),
    )
    return fig

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(layout="wide", page_title="F1 Pit Stop Simulator")
st.title("🏁 F1 Pit Stop Simulator")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("Session / Load")
    year = st.selectbox("Year", list(range(2018, 2025)), index=6)
    gp_name = st.selectbox("Grand Prix", ["Monaco", "Bahrain", "Monza", "Spa-Francorchamps", "Brazil"])
    session_type = st.selectbox("Session", ["R", "Q", "FP1", "FP2", "FP3"])
    cache_dir = st.text_input("FastF1 cache dir", "./ff1_cache")
    load_btn = st.button("Load FastF1 session")

    st.markdown("---")
    st.header("Simulation controls")
    num_laps = st.number_input("Simulation laps", min_value=1, max_value=200, value=20)
    lap_delay = st.slider("Simulation speed (s per lap)", 0.0, 2.0, 0.25, 0.05)
    autopace = st.checkbox("Auto-update speeds from priority", value=False)

    # -------------------------------
    # Safety Car
    st.markdown("---")
    st.header("Safety Car")
    safety_car_enabled = st.checkbox("Enable Safety Car", value=False)
    sc_start = st.number_input("Start lap", min_value=1, max_value=num_laps, value=5)
    sc_end = st.number_input("End lap", min_value=1, max_value=num_laps, value=10)

    track_x, track_y = generate_track_shape(gp_name)


# -------------------------------
# Session state initialization
# -------------------------------
if "drivers" not in st.session_state: st.session_state["drivers"] = []
if "simulator" not in st.session_state: st.session_state["simulator"] = PitStopSimulator()
if "session_loaded" not in st.session_state: st.session_state["session_loaded"] = False
if "running" not in st.session_state: st.session_state["running"] = False
if "lap" not in st.session_state: st.session_state["lap"] = 0
if "pit_messages" not in st.session_state: st.session_state["pit_messages"] = []
if "lap_pit_messages" not in st.session_state: st.session_state["lap_pit_messages"] = []

# -------------------------------
# Load FastF1 session
# -------------------------------
if load_btn:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        with st.spinner(f"Loading {gp_name} {session_type} {year}..."):
            session = fastf1.get_session(year, gp_name, session_type)
            session.load()
        st.success("Session loaded!")

        drivers_list = []
        for drv in session.drivers:
            laps = session.laps.pick_drivers(drv)
            if laps.empty:
                continue
            info = session.get_driver(drv)
            full_name = info.get('FullName', drv)
            last_lap = laps.iloc[-1]
            fuel = max(0.1, 100 - int(last_lap['LapNumber'])) if 'LapNumber' in last_lap else random.uniform(20, 80)
            tire_wear = float(last_lap['TyreLife']) if 'TyreLife' in last_lap and not pd.isna(last_lap['TyreLife']) else random.uniform(20, 80)
            driver = Driver(full_name, fuel, tire_wear)
            drivers_list.append(driver)

        if not drivers_list:
            for i in range(8):
                drivers_list.append(Driver(f"Driver_{i+1}", random.uniform(20, 80), random.uniform(10, 60)))

        st.session_state["drivers"] = drivers_list
        st.session_state["simulator"] = PitStopSimulator()
        for d in st.session_state["drivers"]:
            st.session_state["simulator"].add_driver(d)
        st.session_state["session_loaded"] = True
        st.session_state["lap"] = 0
        st.session_state["pit_messages"] = []
        st.session_state["lap_pit_messages"] = []
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load session: {e}")

#Simulation Fragment
@st.fragment(run_every=lap_delay if st.session_state.get("running", False) else None)
def simulation_display():
    if st.session_state.get("running", False) and st.session_state.get("session_loaded", False):
        if st.session_state["lap"] < num_laps:
            st.session_state["lap"] += 1
            lap_pit_messages = []

            # Check if Safety Car is active
            safety_car_active = safety_car_enabled and sc_start <= st.session_state["lap"] <= sc_end

            if safety_car_active:
                #  Sortiraj vozače po prioritetu
                sorted_drivers = sorted(st.session_state["drivers"], key=lambda d: d.priority)
                
                # 3 najmanja prioritet -> pit stop
                low_priority_drivers = sorted_drivers[:3]
                
                # 3 najveća prioritet -> “preuzimaju mjesta”
                high_priority_drivers = sorted_drivers[-3:]
                
                # Pošalji najsporije vozače u pit stop i daj im nakon izlaska najveći prioritet
                for d in low_priority_drivers:
                    d.fuel = 0.1
                    d.tire_wear = 0.0
                    d.pit_count += 1
                    d.pit_history.append(st.session_state["lap"])
                    # povećaj prioritet na max+1 da preuzmu pozicije
                    d.priority = max([drv.priority for drv in st.session_state["drivers"]]) + 1
                    st.session_state["simulator"].update_driver(d)
                    msg = f"🏁 Safety Car: {d.name} enters pit stop & gains top priority!"
                    st.session_state["pit_messages"].append(msg)
                    lap_pit_messages.append(msg)
                
                # Smanji prioritet prethodnih top vozača da zamijene mjesta
                for d in high_priority_drivers:
                    d.priority = min([drv.priority for drv in st.session_state["drivers"]]) - 1
                    st.session_state["simulator"].update_driver(d)
                    msg = f"⚠️ {d.name} loses priority during Safety Car!"
                    st.session_state["pit_messages"].append(msg)
                    lap_pit_messages.append(msg)

            else:
                # Normal lap
                for d in st.session_state["drivers"]:
                    burn = random.uniform(0.5, 2.0)
                    wear = random.uniform(0.5, 2.0)
                    d.fuel = max(0.1, d.fuel - burn)
                    d.tire_wear = min(100.0, d.tire_wear + wear)
                    d.priority = d.compute_priority()

                    if d.fuel < 10 or d.tire_wear > 90:
                        d.fuel = 100.0
                        d.tire_wear = 0.0
                        d.pit_count += 1
                        d.pit_history.append(st.session_state["lap"])
                        msg = f"🏁 {d.name} enters pit stop! (Pit #{d.pit_count})"
                        st.session_state["pit_messages"].append(msg)
                        lap_pit_messages.append(msg)

                    priorities = np.array([dr.priority for dr in st.session_state["drivers"]])
                    pmin, pmax = priorities.min(), priorities.max()
                    rng = (pmax - pmin) if (pmax - pmin) > 0 else 1.0
                    norm = (d.priority - pmin) / rng
                    d.speed = 0.6 + (1.4 - norm)
                    d.progress = (d.progress + 0.02 * (d.speed / 1.5)) % 1.0
                    st.session_state["simulator"].update_driver(d)

            st.session_state["lap_pit_messages"] = lap_pit_messages

    # Display visualizations
    if st.session_state.get("session_loaded", False):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Track & Simulation")
            st.plotly_chart(generate_track_figure_icons(st.session_state["drivers"], size=600), width="content")
            st.plotly_chart(generate_pit_timeline(st.session_state["drivers"], num_laps), width="content")
        with col2:
            st.subheader(f"Lap {st.session_state['lap']} / {num_laps}")
            for msg in st.session_state.get("lap_pit_messages", []):
                st.info(msg)
            st.subheader("Priority Chart")
            st.plotly_chart(generate_priority_bar_figure(st.session_state["drivers"]), width="content")


# -------------------------------
# Main Layout (controls)
# -------------------------------
if st.session_state.get("session_loaded", False):
    # Start / Pause / Reset buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("Start"):
            st.session_state["running"] = True
            st.rerun()
    with btn_col2:
        if st.button("Pause"):
            st.session_state["running"] = False
            st.rerun()
    with btn_col3:
        if st.button("Reset position"):
            st.session_state["running"] = False
            for d in st.session_state["drivers"]:
                d.progress = random.random()
                d.pit_history = []
                d.pit_count = 0
            st.session_state["lap"] = 0
            st.session_state["pit_messages"] = []
            st.session_state["lap_pit_messages"] = []
            st.rerun()

    # Run simulation fragment
    simulation_display()
    st.markdown("---")

    # Driver editor section
    st.markdown("### Edit Drivers")
    col_edit1, col_edit2 = st.columns(2)
    drivers_per_col = (len(st.session_state["drivers"]) + 1) // 2

    for col, driver_slice, start_idx in zip(
        [col_edit1, col_edit2],
        [st.session_state["drivers"][:drivers_per_col], st.session_state["drivers"][drivers_per_col:]],
        [0, drivers_per_col]
    ):
        for idx, d in enumerate(driver_slice, start=start_idx):
            with col.expander(f"{d.name} — fuel {d.fuel:.1f}, tyre {d.tire_wear:.1f}"):
                new_fuel = st.slider(f"Fuel — {d.name}", 0.1, 100.0, float(d.fuel), key=f"fuel_{idx}")
                new_tyre = st.slider(f"Tire wear — {d.name}", 0.0, 100.0, float(d.tire_wear), key=f"tyre_{idx}")
                cola, colb = st.columns(2)

                if cola.button("Apply", key=f"apply_{idx}"):
                    d.fuel = new_fuel
                    d.tire_wear = new_tyre
                    d.priority = d.compute_priority()
                    st.session_state["simulator"].update_driver(d)
                    st.rerun()

                if colb.button("Force pit", key=f"pit_{idx}"):
                    # Direktno forsiraj pit stop za ovog vozača
                    d.fuel = 0.1
                    d.tire_wear = 0.0
                    d.pit_count += 1
                    d.pit_history.append(st.session_state["lap"])
                    d.priority = max([drv.priority for drv in st.session_state["drivers"]]) + 1  # opcionalno da dobije top priority
                    st.session_state["simulator"].update_driver(d)

                    st.info(f"🏁 {d.name} enters pit stop! (Pit #{d.pit_count})")
                    st.session_state["lap_pit_messages"].append(f"🏁 {d.name} enters pit stop! (Pit #{d.pit_count})")
                    st.rerun()  # refresh prikaz

    if st.button("Add random driver"):
        name = f"Driver_{len(st.session_state['drivers']) + 1}"
        d = Driver(name, random.uniform(20, 80), random.uniform(10, 60))
        st.session_state["drivers"].append(d)
        st.session_state["simulator"].add_driver(d)
        st.rerun()
    
else:
    st.title("🏎️ Welcome to F1 Pit Stop Simulator")
    st.write("Please select a race and click **Load FastF1 session** to start the simulation.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>© 2025 Naida & Lana - FastF1</div>", unsafe_allow_html=True)