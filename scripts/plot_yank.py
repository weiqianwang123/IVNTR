import plotly.graph_objects as go
import numpy as np

labels = [
    "Total Trials", "Successfully Start", "Communication Error",
    "Navigation Success", "Navigation Failure", "Hit", "Trapped",
    "Searching Success", "Searching Failure", "Detection Error", "Stopped Too Far","Timeout"
]

source = [0, 1, 1, 0, 3, 3, 4, 4, 8, 8,8]
target = [1, 4, 3, 2, 7, 8, 5, 6, 9, 10,11]
values = [46, 4, 42, 2, 26, 16, 2, 2, 11,2,3]

# Compute totals per node
n_nodes = len(labels)
inflow = np.zeros(n_nodes)
outflow = np.zeros(n_nodes)

for s, t, v in zip(source, target, values):
    outflow[s] += v
    inflow[t] += v

# Optionally, total can be max(inflow, outflow)
totals = [int(max(inflow[i], outflow[i])) for i in range(n_nodes)]

# Append totals to labels
labels_with_totals = [f"{label} ({total})" for label, total in zip(labels, totals)]

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=5, thickness=10,
        line=dict(color="black", width=0.5),
        label=labels_with_totals
    ),
    link=dict(
        source=source, target=target, value=values
    )
)])

fig.update_layout(
    title=dict(text="Trial Result Flow", font=dict(size=24)),
    font=dict(size=24),
    width=1400,     # ↓ Smaller width = thinner figure
    height=600,    # Keep height same or adjust as needed
    margin=dict(l=20, r=20, t=40, b=20)
)
fig.update_layout(title_text="Trial Result Flow", font_size=35)
fig.show()
fig.write_image("trial_result_sankey.png", width=1000, height=600)