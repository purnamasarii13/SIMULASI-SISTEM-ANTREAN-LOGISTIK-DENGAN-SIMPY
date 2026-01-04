import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Core simulation functions
# =========================

def service_process(env, entity_id, server, records, min_service, max_service):
    """Datang -> Request -> Layanan -> Selesai -> Pergi + logging"""
    arrival_time = env.now

    with server.request() as req:
        yield req
        start_service = env.now
        queue_time = start_service - arrival_time

        service_time = random.uniform(min_service, max_service)
        yield env.timeout(service_time)

        finish_time = env.now

    records.append({
        "entity_id": entity_id,
        "arrival_time": arrival_time,
        "start_service": start_service,
        "finish_time": finish_time,
        "queue_time": queue_time,
        "service_time": service_time,
        "system_time": finish_time - arrival_time
    })


def entity_generator(env, server, records, mean_interarrival, min_service, max_service):
    """Generator kedatangan entitas (Eksponensial)."""
    i = 0
    while True:
        i += 1
        env.process(service_process(env, f"Entity {i}", server, records, min_service, max_service))
        interarrival = random.expovariate(1.0 / mean_interarrival)
        yield env.timeout(interarrival)


def monitor_queue(env, server, trace, capacity, sample_every=1):
    """Mencatat time series: queue_length, in_service, utilization."""
    while True:
        in_service = server.count
        trace.append({
            "time": env.now,
            "queue_length": len(server.queue),
            "in_service": in_service,
            "utilization": (in_service / capacity) if capacity > 0 else np.nan
        })
        yield env.timeout(sample_every)


def run_simulation(
    capacity: int,
    sim_time: float,
    seed: int,
    mean_interarrival: float,
    min_service: float,
    max_service: float,
    queue_sample_every: int,
):
    """Run SimPy simulation and return df_log, df_trace."""
    random.seed(seed)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=capacity)

    records = []
    trace = []

    env.process(entity_generator(env, server, records, mean_interarrival, min_service, max_service))
    env.process(monitor_queue(env, server, trace, capacity, sample_every=queue_sample_every))

    env.run(until=sim_time)

    df_log = pd.DataFrame(records)
    df_trace = pd.DataFrame(trace)
    return df_log, df_trace


def make_summary(df_A, q_A, df_B, q_B):
    summary = pd.DataFrame([
        {
            "scenario": "A (cap=1)",
            "avg_queue_time": df_A["queue_time"].mean(),
            "max_queue_time": df_A["queue_time"].max(),
            "avg_system_time": df_A["system_time"].mean(),
            "n_entities": len(df_A),
            "avg_queue_length": q_A["queue_length"].mean(),
            "max_queue_length": q_A["queue_length"].max(),
            "avg_utilization": q_A["utilization"].mean(),
        },
        {
            "scenario": "B (cap=2)",
            "avg_queue_time": df_B["queue_time"].mean(),
            "max_queue_time": df_B["queue_time"].max(),
            "avg_system_time": df_B["system_time"].mean(),
            "n_entities": len(df_B),
            "avg_queue_length": q_B["queue_length"].mean(),
            "max_queue_length": q_B["queue_length"].max(),
            "avg_utilization": q_B["utilization"].mean(),
        }
    ])
    return summary


# ===========
# Streamlit UI
# ===========

st.set_page_config(page_title="SimPy Queue Simulation", layout="wide")
st.title("SIMULASI SISTEM ANTREAN LOGISTIK DENGAN SIMPY")
st.caption("Baseline (capacity=1) vs Perbaikan (capacity=2) — Data logging, What-If analysis, Visualisasi")

with st.sidebar:
    st.header("Parameter Simulasi")

    seed = st.number_input("RANDOM_SEED", value=42, min_value=0, step=1)
    sim_time = st.number_input("SIM_TIME (menit)", value=1000, min_value=10, step=10)

    st.subheader("Kedatangan (Eksponensial)")
    mean_interarrival = st.number_input("MEAN_INTERARRIVAL (menit)", value=8.0, min_value=0.1, step=0.5)

    st.subheader("Layanan (Uniform)")
    min_service = st.number_input("MIN_SERVICE (menit)", value=5.0, min_value=0.0, step=0.5)
    max_service = st.number_input("MAX_SERVICE (menit)", value=12.0, min_value=0.0, step=0.5)

    st.subheader("Monitoring antrean")
    queue_sample_every = st.number_input("QUEUE_SAMPLE_EVERY (menit)", value=1, min_value=1, step=1)

    st.subheader("Skenario")
    cap_A = st.number_input("Capacity Skenario A (Baseline)", value=1, min_value=1, step=1)
    cap_B = st.number_input("Capacity Skenario B (Perbaikan)", value=2, min_value=1, step=1)

    run_btn = st.button("Jalankan Simulasi", type="primary")

# Validasi kecil
if max_service < min_service:
    st.error("MAX_SERVICE harus >= MIN_SERVICE.")
    st.stop()

# Cache agar tidak selalu menghitung ulang saat interaksi UI
@st.cache_data(show_spinner=True)
def cached_run(capacity, sim_time, seed, mean_interarrival, min_service, max_service, queue_sample_every):
    return run_simulation(
        capacity=capacity,
        sim_time=sim_time,
        seed=seed,
        mean_interarrival=mean_interarrival,
        min_service=min_service,
        max_service=max_service,
        queue_sample_every=queue_sample_every,
    )

if run_btn:
    df_A, q_A = cached_run(cap_A, sim_time, seed, mean_interarrival, min_service, max_service, queue_sample_every)
    df_B, q_B = cached_run(cap_B, sim_time, seed, mean_interarrival, min_service, max_service, queue_sample_every)

    summary = make_summary(df_A, q_A, df_B, q_B)

    # =======
    # Output
    # =======
    st.subheader("Ringkasan Perbandingan Skenario")
    st.dataframe(summary, use_container_width=True)

    improvement = summary.loc[0, "avg_queue_time"] - summary.loc[1, "avg_queue_time"]
    base = summary.loc[0, "avg_queue_time"]
    improvement_pct = (improvement / base * 100) if base and base > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Δ Avg Queue Time (A→B)", f"{improvement:.2f} menit")
    col2.metric("Penurunan relatif", f"{improvement_pct:.1f}%")
    col3.metric("Entitas tercatat (A / B)", f"{len(df_A)} / {len(df_B)}")

    if improvement_pct >= 30:
        st.success("Interpretasi cepat: Penambahan kapasitas berdampak besar (≥30%).")
    elif improvement_pct >= 10:
        st.info("Interpretasi cepat: Dampak penambahan kapasitas sedang (10–30%).")
    else:
        st.warning("Interpretasi cepat: Dampak penambahan kapasitas kecil (<10%).")

    st.divider()

    # =================
    # Visualizations
    # =================
    st.subheader("Visualisasi")

    left, right = st.columns(2)

    with left:
        st.markdown(f"**Histogram Waktu Tunggu — Skenario A (capacity={cap_A})**")
        fig = plt.figure(figsize=(8, 4))
        plt.hist(df_A["queue_time"], bins=20, edgecolor="black")
        plt.xlabel("Queue Time (menit)")
        plt.ylabel("Frekuensi")
        plt.grid(axis="y")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown(f"**Histogram Waktu Tunggu — Skenario B (capacity={cap_B})**")
        fig = plt.figure(figsize=(8, 4))
        plt.hist(df_B["queue_time"], bins=20, edgecolor="black")
        plt.xlabel("Queue Time (menit)")
        plt.ylabel("Frekuensi")
        plt.grid(axis="y")
        st.pyplot(fig)
        plt.close(fig)

    left2, right2 = st.columns(2)

    with left2:
        st.markdown(f"**Panjang Antrean vs Waktu — Skenario A (capacity={cap_A})**")
        fig = plt.figure(figsize=(8, 4))
        plt.plot(q_A["time"], q_A["queue_length"])
        plt.xlabel("Waktu simulasi (menit)")
        plt.ylabel("Queue length")
        plt.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    with right2:
        st.markdown(f"**Panjang Antrean vs Waktu — Skenario B (capacity={cap_B})**")
        fig = plt.figure(figsize=(8, 4))
        plt.plot(q_B["time"], q_B["queue_length"])
        plt.xlabel("Waktu simulasi (menit)")
        plt.ylabel("Queue length")
        plt.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # =================
    # Data download
    # =================
    st.subheader("Unduh Data (CSV)")
    c1, c2, c3, c4 = st.columns(4)

    c1.download_button(
        "Download log A",
        data=df_A.to_csv(index=False).encode("utf-8"),
        file_name="log_skenario_A.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Download log B",
        data=df_B.to_csv(index=False).encode("utf-8"),
        file_name="log_skenario_B.csv",
        mime="text/csv",
    )
    c3.download_button(
        "Download trace A",
        data=q_A.to_csv(index=False).encode("utf-8"),
        file_name="queue_trace_A.csv",
        mime="text/csv",
    )
    c4.download_button(
        "Download trace B",
        data=q_B.to_csv(index=False).encode("utf-8"),
        file_name="queue_trace_B.csv",
        mime="text/csv",
    )

else:
    st.info("Atur parameter di sidebar, lalu klik **Jalankan Simulasi**.")
    st.markdown(
        """
        **Catatan deploy:**
        - Aplikasi ini menjalankan simulasi SimPy untuk 2 skenario (A dan B).
        - Anda bisa ubah kapasitas A/B, SIM_TIME, dan distribusi parameter lewat sidebar.
        """
    )
