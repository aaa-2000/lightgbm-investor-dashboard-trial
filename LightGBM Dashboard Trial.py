# 1. Import libraries

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# 2. Page configuration

st.set_page_config(
    page_title="LightGBM Investor Dashboard",
    layout="wide"
)


# 2.1 Dashboard styling

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0E1117;
}

/* Main page width and spacing */
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1500px;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #262730;
}

/* Main heading */
h1 {
    font-size: 40px !important;
    font-weight: 700 !important;
}

/* Section headings */
h2, h3 {
    font-weight: 600 !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    font-size: 14px;
}

/* Metric values */
[data-testid="stMetricValue"] {
    font-size: 30px;
}

/* Metric spacing */
[data-testid="stMetric"] {
    padding-top: 5px;
    padding-bottom: 5px;
}

/* Selectbox background */
div[data-baseweb="select"] > div {
    background-color: #1E2128;
}

/* Chart spacing */
[data-testid="stPlotlyChart"] {
    margin-top: 0px;
}

</style>
""", unsafe_allow_html=True)


# 3. Load LightGBM prediction data

sp_predictions = pd.read_csv(
    "sp500_lightgbm_predictions.csv",
    parse_dates=["Date"]
)

ftse_predictions = pd.read_csv(
    "ftse100_lightgbm_predictions.csv",
    parse_dates=["Date"]
)


# 4. Dashboard heading

st.title("Market Signal Comparison")

st.caption(
    "Stage 2 LightGBM market-direction predictions using technical "
    "indicators, FinBERT sentiment and VIX."
)


# 5. Sidebar filters

st.sidebar.subheader("Filters")

market = st.sidebar.selectbox(
    "Market",
    ["S&P 500", "FTSE 100"]
)


# Choose the correct dataset

if market == "S&P 500":
    predictions = sp_predictions.copy()

else:
    predictions = ftse_predictions.copy()


# Reset row numbers
predictions = predictions.reset_index(drop=True)


# 6. Find the selected prediction

# Default to the latest prediction
selected_index = len(predictions) - 1


# Check if a point was previously selected on the graph
if "prediction_chart" in st.session_state:

    saved_chart = st.session_state["prediction_chart"]

    if saved_chart.selection.points:

        selected_index = (
            saved_chart.selection.points[0]["point_number"]
        )


# Get selected row
selected_row = predictions.iloc[selected_index]


# Get selected values
selected_date = selected_row["Date"]

actual_class = int(
    selected_row["Actual_Class"]
)

predicted_class = int(
    selected_row["Predicted_Class"]
)

probability_up = float(
    selected_row["Probability_Up"]
)


# Change actual class into readable text

if actual_class == 1:
    actual_direction = "UP"

else:
    actual_direction = "DOWN"


# Change predicted class into readable text

if predicted_class == 1:

    predicted_direction = "UP"
    prediction_probability = probability_up

else:

    predicted_direction = "DOWN"
    prediction_probability = 1 - probability_up


# Check if prediction was correct

if actual_class == predicted_class:
    prediction_result = "Correct"

else:
    prediction_result = "Incorrect"


# 7. Selected investor signal

with st.container(border=True):

    st.subheader("LightGBM Signal")
    st.title(predicted_direction)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "5-Day Direction",
            predicted_direction
        )

    with col2:
        st.metric(
            "Prediction Probability",
            f"{selected_probability:.1%}"
        )

    with col3:
        st.metric(
            "Prediction Date",
            selected_date.strftime("%d %b %Y")
        )

    st.caption(
        "Stage 2 model using technical indicators, "
        "FinBERT sentiment and VIX."
    )

# 8. Graph and evaluation metrics

graph_col, metric_col = st.columns([3, 1])


# 8.1 Probability graph

with graph_col:

    st.subheader("5-Day Market Direction")

    st.write(
        "Click a point on the graph to explore the prediction "
        "and model performance at that stage."
    )


    # Create chart
    fig = go.Figure()


    # Add probability line
    fig.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Probability_Up"],
            mode="lines+markers",
            name="Probability of Up",

            line=dict(
                color="#7CC7F5",
                width=2
            ),

            marker=dict(
                size=4,
                color="#7CC7F5",
                opacity=0.5
            ),

            fill="tozeroy",

            fillcolor="rgba(124, 199, 245, 0.20)"
        )
    )


    # Add 50% reference line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#666666",
        line_width=1
    )


    # Chart layout
    fig.update_layout(
        height=460,

        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",

        margin=dict(
            l=40,
            r=20,
            t=20,
            b=40
        ),

        xaxis_title="Date",
        yaxis_title="Probability of Up",

        showlegend=False,

        hovermode="x unified"
    )


    # X-axis style
    fig.update_xaxes(
        showgrid=False,
        zeroline=False
    )


    # Y-axis style
    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
        showgrid=True,
        gridcolor="rgba(255,255,255,0.12)",
        zeroline=False
    )


    # Display interactive chart
    chart_event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="prediction_chart",
        on_select="rerun",
        selection_mode="points"
    )


# 8.2 Evaluation metrics

with metric_col:

    st.subheader("Model Evaluation")

    st.caption(
        "Performance from the start of the test period "
        "up to the selected date."
    )


    # Only use predictions up to selected date
    evaluation_data = predictions.iloc[
        :selected_index + 1
    ]


    # Count true positives
    true_positive = len(
        evaluation_data[
            (evaluation_data["Actual_Class"] == 1)
            &
            (evaluation_data["Predicted_Class"] == 1)
        ]
    )


    # Count true negatives
    true_negative = len(
        evaluation_data[
            (evaluation_data["Actual_Class"] == 0)
            &
            (evaluation_data["Predicted_Class"] == 0)
        ]
    )


    # Count false positives
    false_positive = len(
        evaluation_data[
            (evaluation_data["Actual_Class"] == 0)
            &
            (evaluation_data["Predicted_Class"] == 1)
        ]
    )


    # Count false negatives
    false_negative = len(
        evaluation_data[
            (evaluation_data["Actual_Class"] == 1)
            &
            (evaluation_data["Predicted_Class"] == 0)
        ]
    )


    # Total predictions so far
    total_predictions = len(
        evaluation_data
    )


    # Calculate accuracy
    accuracy = (
        true_positive + true_negative
    ) / total_predictions


    # Calculate precision
    if true_positive + false_positive > 0:

        precision = true_positive / (
            true_positive + false_positive
        )

    else:

        precision = 0


    # Calculate recall
    if true_positive + false_negative > 0:

        recall = true_positive / (
            true_positive + false_negative
        )

    else:

        recall = 0


    # Calculate F1 score
    if precision + recall > 0:

        f1_score = (
            2 * precision * recall
        ) / (
            precision + recall
        )

    else:

        f1_score = 0


    # Display evaluation metrics

    st.metric(
        "Accuracy",
        f"{accuracy:.1%}"
    )

    st.metric(
        "Precision",
        f"{precision:.1%}"
    )

    st.metric(
        "Recall",
        f"{recall:.1%}"
    )

    st.metric(
        "F1 Score",
        f"{f1_score:.1%}"
    )


    st.caption(
        f"{total_predictions} test observations included."
    )