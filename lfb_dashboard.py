
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

st.set_page_config(layout="wide")
st.title("🚒 London Fire Brigade Incident & Response Time Analysis")

#######################################################################################
#######################################################################################

@st.cache_data
def load_data():
    return pd.read_parquet("lfb_streamlit.parquet")

df = load_data()

# Feature Engineering

# Convert to datetime 
df["CallDate"] = pd.to_datetime(df["CallDate"])

# Create time features (needed for Daily and Hourly Incident Heatmap)
df["HourOfCall"] = pd.to_datetime(df["TimeOfCall"]).dt.hour
df["CallWeekday"] = pd.to_datetime(df["CallDate"]).dt.day_name()

# Extract year and month
df["Year"] = df["CallDate"].dt.year
df["Month"] = df["CallDate"].dt.month
df["MonthName"] = df["CallDate"].dt.month_name()
df["CallMonth"] = df["CallDate"].dt.month

# Identify incidents where the first pump arrived within the 6-minute response target
df["FirstPump_Within_6min"] = df["FirstPumpArriving_AttendanceTime"] <= 360

#######################################################################################
#######################################################################################

st.sidebar.header("Filters")

# Available years
available_years = ["All"] + sorted(df["Year"].unique())

# Available months (mit All Option)
available_months = ["All"] + [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# Year filter
selected_year = st.sidebar.selectbox(
    "Select Year",
    options=available_years
)

# Month filter
selected_month = st.sidebar.selectbox(
    "Select Month",
    options=available_months
)

# Apply Filters
if selected_year == "All" and selected_month == "All":
    filtered_df = df.copy()

elif selected_year == "All":
    filtered_df = df[df["MonthName"] == selected_month]

elif selected_month == "All":
    filtered_df = df[df["Year"] == selected_year]

else:
    filtered_df = df[
        (df["Year"] == selected_year) &
        (df["MonthName"] == selected_month)
    ]

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

if selected_year == "All":
    year_text = "All Years"
else:
    year_text = selected_year

if selected_month == "All":
    month_text = "All Months"
else:
    month_text = selected_month

st.caption(f"Data shown: {year_text} | {month_text}")

#######################################################################################
#######################################################################################

# KPI Calculations
total_incidents = len(filtered_df)
median_response = filtered_df["FirstPumpArriving_AttendanceTime"].median() / 60
response_within_6min = ((filtered_df["FirstPumpArriving_AttendanceTime"] <= 360).mean() * 100)
false_alarm_rate = (filtered_df["IncidentGroup"] == "False Alarm").mean() * 100
fire_rate = (filtered_df["IncidentGroup"] == "Fire").mean() * 100
special_service_rate = (filtered_df["IncidentGroup"] == "Special Service").mean() * 100
p90_response = filtered_df["FirstPumpArriving_AttendanceTime"].quantile(0.90) / 60
avg_response = filtered_df["FirstPumpArriving_AttendanceTime"].mean() / 60
second_pump_rate = filtered_df["SecondPumpArriving_AttendanceTime"].notna().mean() * 100
avg_pumps = filtered_df["NumPumpsAttending"].mean()
   
#######################################################################################
#######################################################################################

# Display KPIs
st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Total Incidents", f"{total_incidents:,}")
col2.metric("Median Response Time (min)", f"{median_response:.2f}")
col3.metric("Response within 6 min (%)", f"{response_within_6min:.1f}")

col4, col5, col6 = st.columns(3)

col4.metric("False Alarm Rate (%)", f"{false_alarm_rate:.1f}")
col5.metric("Fire Rate (%)", f"{fire_rate:.1f}")
col6.metric("Special Service Rate (%)", f"{special_service_rate:.1f}")

st.subheader("Operational KPIs")

col7, col8, col9, col10 = st.columns(4)

col7.metric("90th Percentile Response Time (min)", f"{p90_response:.2f}")
col8.metric("Average Response Time (min)", f"{avg_response:.2f}")
col9.metric("Second Pump Deployment Rate (%)", f"{second_pump_rate:.1f}")
col10.metric("Average Pumps Attending", f"{avg_pumps:.2f}")

#######################################################################################
#######################################################################################

st.subheader("Monthly Incident Trends by Incident Type")

# Monthly unique incident counts by incident type
monthly_incidents_by_type = (
    filtered_df
    .groupby(["CallMonth", "IncidentGroup"])["IncidentNumber"]
    .nunique()
    .reset_index(name="IncidentCount")
)

# Monthly unique incident counts across all incident types
monthly_incidents_total = (
    filtered_df
    .groupby("CallMonth")["IncidentNumber"]
    .nunique()
    .reset_index(name="IncidentCount")
)

# Label totals so they can be plotted together with incident types
monthly_incidents_total["IncidentGroup"] = "All Incidents"

# Combine into long format
monthly_incident_counts_long = pd.concat(
    [monthly_incidents_by_type, monthly_incidents_total],
    ignore_index=True
)

palette = {
    "All Incidents": "black",
    "False Alarm": sns.color_palette("colorblind")[0],
    "Fire": sns.color_palette("colorblind")[1],
    "Special Service": sns.color_palette("colorblind")[2],
}

sns.set_theme(style="white")  # removes grid automatically

fig, ax = plt.subplots(figsize=(12, 6))

hue_order = [
    "All Incidents",
    "False Alarm",
    "Special Service",
    "Fire"
]

# Plot all incident types EXCEPT totals
sns.lineplot(
    data=monthly_incident_counts_long[monthly_incident_counts_long["IncidentGroup"] != "All Incidents"],
    x="CallMonth",
    y="IncidentCount",
    hue="IncidentGroup",
    hue_order=hue_order[1:],  # exclude All Incidents
    palette=palette,
    linewidth=2.5,
    marker="o",
    ax=ax
)

# Plot ALL INCIDENTS separately with thicker line
sns.lineplot(
    data=monthly_incident_counts_long[monthly_incident_counts_long["IncidentGroup"] == "All Incidents"],
    x="CallMonth",
    y="IncidentCount",
    color="black",
    linewidth=4,
    marker="o",
    label="All Incidents",
    ax=ax
)

ax.set_title("Monthly Incident Trends by Incident Type (2021–2025)", weight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Incidents")

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

# Get current handles and labels
handles, labels = ax.get_legend_handles_labels()

# Desired order
desired_order = [
    "All Incidents",
    "False Alarm",
    "Special Service",
    "Fire"
]

# Reorder legend
ordered_handles = [handles[labels.index(label)] for label in desired_order]

ax.legend(
    ordered_handles,
    desired_order,
    title="Incident Group",
    frameon=False
)

sns.despine()

fig.tight_layout()

st.pyplot(fig)

#######################################################################################
#######################################################################################

st.subheader("Daily and Hourly Incident Heatmap")

# Pivot table: weekday x hour
daily_hourly_incidents = filtered_df.pivot_table(
    index="HourOfCall",
    columns="CallWeekday",
    values="IncidentNumber",
    aggfunc="nunique"
)

# Order by Weekday
weekday_order = [
"Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

daily_hourly_incidents = (
    daily_hourly_incidents
    .reindex(index=range(24))          # hours 0–23
    .reindex(columns=weekday_order)    # Monday → Sunday
)

fig, ax = plt.subplots(figsize=(9, 11))

sns.heatmap(
    daily_hourly_incidents,
    cmap="coolwarm",
    square=True,
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Number of Incidents"},
    ax=ax
)

ax.invert_yaxis()  # 0 at bottom, 23 at top

ax.set_title("Daily and Hourly Incident Heatmap (2021–2025)", weight="bold")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Hour of Call")

fig.tight_layout()

st.pyplot(fig)

#######################################################################################
#######################################################################################

st.subheader("Monthly Response Performance by Incident Type")

avg_firstpump_attendance_by_type = (
    filtered_df
    .groupby(["CallMonth", "IncidentGroup"])["FirstPumpArriving_AttendanceTime"]
    .mean()
    .div(60)
    .reset_index(name="AvgFirstPumpMinutes")
)

avg_firstpump_attendance_total = (
    filtered_df
    .groupby("CallMonth")["FirstPumpArriving_AttendanceTime"]
    .mean()
    .div(60)
    .reset_index(name="AvgFirstPumpMinutes")
)

avg_firstpump_attendance_total["IncidentGroup"] = "All Incidents"

avg_firstpump_attendance_long = pd.concat(
    [avg_firstpump_attendance_by_type, avg_firstpump_attendance_total],
    ignore_index=True
)

palette = {
    "All Incidents": "black",
    "False Alarm": sns.color_palette("colorblind")[0],
    "Fire": sns.color_palette("colorblind")[1],
    "Special Service": sns.color_palette("colorblind")[2],
}

hue_order = ["Fire", "Special Service", "False Alarm", "All Incidents"]

sns.set_theme(style="white")  # removes background grid

fig, ax = plt.subplots(figsize=(12, 6))

# Plot incident types
sns.lineplot(
    data=avg_firstpump_attendance_long[
        avg_firstpump_attendance_long["IncidentGroup"] != "All Incidents"
    ],
    x="CallMonth",
    y="AvgFirstPumpMinutes",
    hue="IncidentGroup",
    hue_order=hue_order[:-1],
    palette=palette,
    linewidth=2.5,
    marker="o",
    ax=ax
)

# Plot ALL incidents separately thicker
sns.lineplot(
    data=avg_firstpump_attendance_long[
        avg_firstpump_attendance_long["IncidentGroup"] == "All Incidents"
    ],
    x="CallMonth",
    y="AvgFirstPumpMinutes",
    color="black",
    linewidth=4,
    marker="o",
    label="All Incidents",
    ax=ax
)

ax.set_title("Average Monthly First Pump Attendance Time by Incident Type (2021–2025)", weight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Average First Pump Attendance Time (minutes)")

ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

# Get current legend handles and labels
handles, labels = ax.get_legend_handles_labels()

# Desired order
desired_order = [
    "All Incidents",
    "Special Service",
    "Fire",
    "False Alarm"
]

# Keep only labels that are actually present
ordered_labels = [label for label in desired_order if label in labels]
ordered_handles = [handles[labels.index(label)] for label in ordered_labels]

ax.legend(
    ordered_handles,
    ordered_labels,
    title="Incident Group",
    frameon=False
)
sns.despine()

fig.tight_layout()

st.pyplot(fig)

#######################################################################################
#######################################################################################



#######################################################################################
#######################################################################################

st.subheader("Response Performance by Borough")

# --- Median response time by borough ---
median_response_by_borough = (
    filtered_df
    .groupby("IncGeo_BoroughName", observed=True)["FirstPumpArriving_AttendanceTime"]
    .median()
    .div(60)
    .reset_index(name="MedianResponseMinutes")
)

# Remove potential NaNs
median_response_by_borough = median_response_by_borough.dropna()

# Sort once (ascending)
median_response_by_borough = median_response_by_borough.sort_values(
    "MedianResponseMinutes",
    ascending=True
)

# Select top and bottom safely
top10_fastest = median_response_by_borough.nsmallest(
    10, "MedianResponseMinutes"
)

top10_slowest = median_response_by_borough.nlargest(
    10, "MedianResponseMinutes"
)

sns.set_theme(style="white")

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 12),
    sharex=True
)


# Top 10 Fastest Boroughs


fast_palette = sns.color_palette("YlGn_r", n_colors=len(top10_fastest))

sns.barplot(
    data=top10_fastest.sort_values("MedianResponseMinutes", ascending=False),
    y="IncGeo_BoroughName",
    x="MedianResponseMinutes",
    palette=fast_palette,
    ax=ax1
)

ax1.axvline(
    x=6,
    color="black",
    linestyle="--",
    linewidth=2
)

ax1.set_title(
    "Top 10 Fastest Boroughs (Median Response Time)",
    weight="bold"
)
ax1.set_xlabel("")
ax1.set_ylabel("")


# Top 10 Slowest Boroughs


slow_palette = sns.color_palette("YlOrRd", n_colors=len(top10_slowest))

sns.barplot(
    data=top10_slowest.sort_values("MedianResponseMinutes", ascending=False),
    y="IncGeo_BoroughName",
    x="MedianResponseMinutes",
    palette=slow_palette,
    ax=ax2
)

ax2.axvline(
    x=6,
    color="black",
    linestyle="--",
    linewidth=2
)

ax2.set_title(
    "Top 10 Slowest Boroughs (Median Response Time)",
    weight="bold"
)
ax2.set_xlabel("Median Response Time (minutes)")
ax2.set_ylabel("")

sns.despine()
fig.tight_layout()

st.pyplot(fig)

#######################################################################################
#######################################################################################

st.subheader("First Pump Response Performance Against the 6-Minute Target")

# Calculate the rate of incidents meeting the response target by incident type
compliance_by_borough = (
    filtered_df.groupby("IncGeo_BoroughName")["FirstPump_Within_6min"]
    .mean()
    .mul(100)
    .reset_index(name="CompliancePercent")
)

# Sort ascending once
compliance_sorted = compliance_by_borough.sort_values("CompliancePercent")

# Select extremes
bottom10_compliance = compliance_sorted.head(10)
top10_compliance = compliance_sorted.tail(10)

# Sort both subsets descending (best at top)
top10_compliance = top10_compliance.sort_values("CompliancePercent", ascending=False)
bottom10_compliance = bottom10_compliance.sort_values("CompliancePercent", ascending=False)

sns.set_theme(style="white")

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 12),
    sharex=True
)

# Top 10 Highest Compliance

high_palette = sns.color_palette("YlGn_r", n_colors=len(top10_compliance))

sns.barplot(
    data=top10_compliance,
    y="IncGeo_BoroughName",
    x="CompliancePercent",
    palette=high_palette,
    ax=ax1
)

ax1.set_title("Top 10 Boroughs — Highest Compliance (%)", weight="bold")
ax1.set_xlabel("")
ax1.set_ylabel("")


# Bottom 10 Lowest Compliance


low_palette = sns.color_palette("YlOrRd", n_colors=len(bottom10_compliance))

sns.barplot(
    data=bottom10_compliance,
    y="IncGeo_BoroughName",
    x="CompliancePercent",
    palette=low_palette,
    ax=ax2
)

ax2.set_title("Top 10 Boroughs — Lowest Compliance (%)", weight="bold")
ax2.set_xlabel("Compliance Rate (%)")
ax2.set_ylabel("")

sns.despine()
fig.tight_layout()

st.pyplot(fig)


#######################################################################################
#######################################################################################

st.subheader("Response Time Bands Distribution")

# Create response time in minutes
filtered_df["ResponseMinutes"] = (
    filtered_df["FirstPumpArriving_AttendanceTime"] / 60
)

# Extreme delay KPI (GLOBAL)
extreme_delay_rate = (
    (filtered_df["ResponseMinutes"] > 10).mean() * 100
)

# Define bands
bins = [0, 6, 8, 10, float("inf")]
labels = ["≤ 6 min", "6–8 min", "8–10 min", "> 10 min"]

filtered_df["ResponseBand"] = pd.cut(
    filtered_df["ResponseMinutes"],
    bins=bins,
    labels=labels,
    right=True
)

# Count incidents per band & type
band_counts = (
    filtered_df
    .groupby(["IncidentGroup", "ResponseBand"])
    .size()
    .reset_index(name="Count")
)

# Calculate percentage within each IncidentGroup
band_counts["Percent"] = (
    band_counts.groupby("IncidentGroup")["Count"]
    .transform(lambda x: 100 * x / x.sum())
)

# Pivot for stacked bar
band_pivot = band_counts.pivot(
    index="IncidentGroup",
    columns="ResponseBand",
    values="Percent"
).fillna(0)

band_pivot = band_pivot[labels]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

colors = ["#2ca02c", "#ffdd57", "#ff8c42", "#d62728"]

left = None

for i, band in enumerate(labels):
    ax.barh(
        band_pivot.index,
        band_pivot[band],
        left=left,
        color=colors[i],
        label=band
    )
    if left is None:
        left = band_pivot[band]
    else:
        left += band_pivot[band]

ax.set_xlim(0, 100)
ax.set_xlabel("Percentage of Incidents (%)")
ax.set_title("Response Time Distribution by Incident Type", weight="bold")

ax.legend(
    title="Response Band",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=4,
    frameon=False
)

sns.despine()
fig.tight_layout()
st.pyplot(fig)


extreme_delay_rate = (
    filtered_df["ResponseMinutes"] > 10
).mean() * 100

st.markdown(f"""
**Extreme Delays**
            : **Incidents exceeding 10 minutes:** {extreme_delay_rate:.2f}%"
""")

#######################################################################################
#######################################################################################






#######################################################################################
#######################################################################################

st.subheader("Distribution of First Pump Attendance Time")

response_minutes = filtered_df["FirstPumpArriving_AttendanceTime"] / 60

median = response_minutes.median()
mean = response_minutes.mean()
p90 = response_minutes.quantile(0.90)

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    response_minutes,
    bins=60,
    kde=True,
    ax=ax
)

# Reference lines
ax.axvline(6, color="red", linestyle="--", linewidth=2, label="6-min target")
ax.axvline(median, color="black", linewidth=2, label=f"Median ({median:.2f})")
ax.axvline(mean, color="blue", linestyle="--", label=f"Mean ({mean:.2f})")
ax.axvline(p90, color="purple", linestyle=":", label=f"P90 ({p90:.2f})")

ax.set_title("Distribution of First Pump Attendance Time", weight="bold")
ax.set_xlabel("Attendance Time (minutes)")
ax.set_ylabel("Frequency")

ax.legend(frameon=False)

sns.despine()
fig.tight_layout()

st.pyplot(fig)

st.markdown(f"""
- Median response time: **{median:.2f} minutes**
- 90% of incidents are handled within **{p90:.2f} minutes**
- The gap between mean and median indicates a right-skewed distribution driven by extreme delays.
""")

#######################################################################################
#######################################################################################

bins = [0, 4, 6, 8, 20]
labels = ["<4 min", "4–6 min", "6–8 min", ">8 min"]

band_distribution = (
    pd.cut(response_minutes, bins=bins, labels=labels)
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(1)
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    x=band_distribution.index,
    y=band_distribution.values,
    palette="YlGnBu",
    ax=ax
)

ax.set_title("Response Time Distribution Bands (%)", weight="bold")
ax.set_ylabel("Percentage of Incidents")
ax.set_xlabel("Response Time Band")

sns.despine()
fig.tight_layout()

st.pyplot(fig)

extreme_delay_share = (
    (response_minutes > 10).mean() * 100
)

st.markdown(f"""
**Extreme Delays**
            : **Incidents exceeding 10 minutes: {extreme_delay_share:.2f}%**
""")

#######################################################################################
#######################################################################################

st.subheader("Response Time Distribution by Incident Type")

incident_types = ["Fire", "Special Service", "False Alarm"]

fig, axes = plt.subplots(
    3, 1,
    figsize=(10, 14),   # deutlich höher
    sharex=True
)

for ax, incident in zip(axes, incident_types):

    subset = filtered_df[
        filtered_df["IncidentGroup"] == incident
    ]

    response_minutes = subset["FirstPumpArriving_AttendanceTime"] / 60

    sns.histplot(
        response_minutes,
        bins=50,
        kde=True,
        ax=ax
    )

    ax.axvline(6, color="red", linestyle="--", linewidth=2)

    ax.set_title(incident, weight="bold")
    ax.set_ylabel("Frequency")

axes[-1].set_xlabel("Attendance Time (minutes)")

sns.despine()
fig.tight_layout()

st.pyplot(fig)

#######################################################################################
#######################################################################################

st.subheader("Response Time Decomposition: Turnout vs Travel")

# Calculate average turnout & travel per Incident Type
decomposition = (
    filtered_df
    .groupby("IncidentGroup")[["TurnoutTimeSeconds", "TravelTimeSeconds"]]
    .mean()
    .div(60)  # convert to minutes
    .reset_index()
)

# Calculate total
decomposition["TotalMinutes"] = (
    decomposition["TurnoutTimeSeconds"] +
    decomposition["TravelTimeSeconds"]
)

# Calculate percentage contribution
decomposition["TurnoutPercent"] = (
    decomposition["TurnoutTimeSeconds"] /
    decomposition["TotalMinutes"] * 100
)

decomposition["TravelPercent"] = (
    decomposition["TravelTimeSeconds"] /
    decomposition["TotalMinutes"] * 100
)

order = ["Fire", "Special Service", "False Alarm"]
decomposition = decomposition.set_index("IncidentGroup").loc[order].reset_index()

sns.set_theme(style="white")

# Prepare stacked bar
fig, ax = plt.subplots(figsize=(10, 6))

colorblind = sns.color_palette("colorblind")

# Turnout
ax.barh(
    decomposition["IncidentGroup"],
    decomposition["TurnoutPercent"],
    color=colorblind[0],
    label="Turnout Time"
)

# Travel
ax.barh(
    decomposition["IncidentGroup"],
    decomposition["TravelPercent"],
    left=decomposition["TurnoutPercent"],
    color=colorblind[1],
    label="Travel Time"
)

ax.set_xlim(0, 100)
ax.set_xlabel("Percentage of Total Response Time (%)")
ax.set_title("Turnout vs Travel Contribution by Incident Type", weight="bold")

ax.legend(
    title="Component",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False
)

sns.despine()
fig.tight_layout()
st.pyplot(fig)

#######################################################################################
#######################################################################################

st.subheader("Extreme Delays (>10 minutes): Pareto Analysis")

filtered_df["ResponseMinutes"] = (
    filtered_df["FirstPumpArriving_AttendanceTime"] / 60
)

extreme_df = filtered_df[filtered_df["ResponseMinutes"] > 10]

delay_counts_extreme = (
    extreme_df
    .groupby("DelayCode_Description")
    .size()
    .reset_index(name="IncidentCount")
    .sort_values("IncidentCount", ascending=False)
)

if delay_counts_extreme.empty:
    st.warning("No extreme delays found for selected filters.")
    st.stop()

total_extreme = delay_counts_extreme["IncidentCount"].sum()

delay_counts_extreme["Percent"] = (
    delay_counts_extreme["IncidentCount"] / total_extreme * 100
)

delay_counts_extreme["CumulativePercent"] = (
    delay_counts_extreme["Percent"].cumsum()
)

pareto_df = delay_counts_extreme.head(10).copy()

pareto_df["ShortLabel"] = (
    pareto_df["DelayCode_Description"]
    .str.slice(0, 35)
)

sns.set_theme(style="white")

fig, ax1 = plt.subplots(figsize=(16, 8))

bars = sns.barplot(
    data=pareto_df,
    x="ShortLabel",
    y="Percent",
    palette="Reds_r",
    ax=ax1
)

ax1.set_ylabel("Share of Extreme Delays (%)", fontsize=13)
ax1.set_xlabel("")
ax1.set_title(
    "Pareto Analysis of Extreme Delay Drivers (>10 minutes)",
    fontsize=16,
    weight="bold"
)

ax1.tick_params(axis='x', labelsize=11)
ax1.tick_params(axis='y', labelsize=11)

plt.xticks(rotation=45, ha="right")

# Add percentage labels on bars
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.1f%%", padding=3, fontsize=10)

# Cumulative line
ax2 = ax1.twinx()

ax2.plot(
    pareto_df["ShortLabel"],
    pareto_df["CumulativePercent"],
    color="black",
    marker="o",
    linewidth=2
)

ax2.set_ylabel("Cumulative Share (%)", fontsize=13)
ax2.set_ylim(0, 100)
ax2.tick_params(axis='y', labelsize=11)

# 80% reference
ax2.axhline(80, linestyle="--", color="gray", alpha=0.6)

sns.despine()
fig.tight_layout()

st.pyplot(fig)

# Calculate Top 3 cumulative share
top3_share = delay_counts_extreme.head(3)["Percent"].sum()

st.markdown(f"""
Extreme Delays: 
  **Top 3 delay codes explain {top3_share:.1f}% of extreme response delays (>10 minutes).**
""")


















#######################################################################################
#######################################################################################






#######################################################################################
#######################################################################################

tab1, tab2, tab3 = st.tabs([
    "Operational Demand",
    "Response Performance",
    "Geographic Performance"
])

with tab1:

    st.subheader("Key Performance Indicators")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Incidents", f"{total_incidents:,}")
    col2.metric("Median Response Time (min)", f"{median_response:.2f}")
    col3.metric("Response within 6 min (%)", f"{response_within_6min:.1f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("False Alarm Rate (%)", f"{false_alarm_rate:.1f}")
    col5.metric("Fire Rate (%)", f"{fire_rate:.1f}")
    col6.metric("Special Service Rate (%)", f"{special_service_rate:.1f}")

    st.subheader("Monthly Incident Trends by Incident Type")
    # <- hier kommt dein kompletter Monthly Plot Code rein

    st.subheader("Daily and Hourly Incident Heatmap")
    # <- hier kommt dein Heatmap Code rein

with tab2:

    st.subheader("Operational KPIs")

    col7, col8, col9, col10 = st.columns(4)
    col7.metric("90th Percentile Response Time (min)", f"{p90_response:.2f}")
    col8.metric("Average Response Time (min)", f"{avg_response:.2f}")
    col9.metric("Second Pump Deployment Rate (%)", f"{second_pump_rate:.1f}")
    col10.metric("Average Pumps Attending", f"{avg_pumps:.2f}")

    st.subheader("Response Performance Over Time")

with tab3:

    st.subheader("Response Performance by Borough")
    st.subheader("First Pump Response Performance Against the 6-Minute Target")
   
#######################################################################################
#######################################################################################

