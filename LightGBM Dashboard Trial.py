import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# Page configuration

st.set_page_config(
    page_title="LightGBM Investor Dashboard",
    page_icon="📈",
    layout="wide"
)


# Dashboard styling

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1.2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background-color: #171922;
}

h1 {
    font-size: 36px !important;
    font-weight: 700 !important;
}

h2, h3 {
    font-weight: 600 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 13px;
}

[data-testid="stMetricValue"] {
    font-size: 26px;
}

[data-testid="stMetric"] {
    padding-top: 2px;
    padding-bottom: 2px;
}

div[data-baseweb="select"] > div {
    background-color: #1E2128;
}

</style>
""", unsafe_allow_html=True)


# Load LightGBM prediction datasets

sp500_stage1 = pd.read_csv(
    "sp500_lightgbm_stage1_predictions.csv",
    parse_dates=["Date"]
)

sp500_stage2 = pd.read_csv(
    "sp500_lightgbm_stage2_predictions.csv",
    parse_dates=["Date"]
)

ftse_stage1 = pd.read_csv(
    "ftse100_lightgbm_stage1_predictions.csv",
    parse_dates=["Date"]
)

ftse_stage2 = pd.read_csv(
    "ftse100_lightgbm_stage2_predictions.csv",
    parse_dates=["Date"]
)


# Load historical market datasets

sp500_market = pd.read_csv(
    "clean_sp500.csv"
)

ftse_market = pd.read_csv(
    "clean_ftse_100.csv"
)


# Rename first market column to Date

sp500_market.rename(
    columns={
        sp500_market.columns[0]: "Date"
    },
    inplace=True
)

ftse_market.rename(
    columns={
        ftse_market.columns[0]: "Date"
    },
    inplace=True
)


# Convert market dates

sp500_market["Date"] = pd.to_datetime(
    sp500_market["Date"]
)

ftse_market["Date"] = pd.to_datetime(
    ftse_market["Date"]
)


# Load VIX dataset

vix_data = pd.read_csv(
    "clean_vix.csv"
)


# Convert VIX Price column into Date

vix_data["Price"] = pd.to_datetime(
    vix_data["Price"]
)

vix_data.rename(
    columns={
        "Price": "Date"
    },
    inplace=True
)


# Load sentiment datasets

sp500_sentiment = pd.read_csv(
    "daily_sp500_sentiment.csv",
    parse_dates=["Date"]
)

guardian_sentiment = pd.read_csv(
    "daily_guardian_sentiment.csv",
    parse_dates=["Date"]
)


# Sort prediction datasets

prediction_datasets = [
    sp500_stage1,
    sp500_stage2,
    ftse_stage1,
    ftse_stage2
]

for dataset in prediction_datasets:

    dataset.sort_values(
        "Date",
        inplace=True
    )

    dataset.reset_index(
        drop=True,
        inplace=True
    )


# Sort historical datasets

historical_datasets = [
    sp500_market,
    ftse_market,
    vix_data,
    sp500_sentiment,
    guardian_sentiment
]

for dataset in historical_datasets:

    dataset.sort_values(
        "Date",
        inplace=True
    )

    dataset.reset_index(
        drop=True,
        inplace=True
    )


# Sidebar navigation

st.sidebar.title(
    "LightGBM Investor Dashboard"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Investor Decision Support",
        "Market & Behaviour"
    ]
)


# Investor Decision Support page

if page == "Investor Decision Support":

    st.title(
        "Investor Decision Support"
    )

    st.caption(
        "Compare technical-only and technical + behavioural "
        "LightGBM predictions for five-day market direction."
    )


    # Select market

    market_col, date_col = st.columns(
        [1, 2]
    )

    with market_col:

        market = st.selectbox(
            "Market",
            [
                "S&P 500",
                "FTSE 100"
            ]
        )


    # Select market prediction datasets

    if market == "S&P 500":

        stage1 = sp500_stage1.copy()
        stage2 = sp500_stage2.copy()

    else:

        stage1 = ftse_stage1.copy()
        stage2 = ftse_stage2.copy()


    # Get actual prediction dates

    prediction_dates = sorted(
        set(stage1["Date"]).union(
            set(stage2["Date"])
        )
    )


    prediction_dates = [
        pd.Timestamp(date).date()
        for date in prediction_dates
    ]


    # Select prediction date

    with date_col:

        selected_date = st.selectbox(
            "Prediction Date",
            prediction_dates,
            index=len(prediction_dates) - 1,
            format_func=lambda date: date.strftime(
                "%d %b %Y"
            )
        )


    selected_date = pd.Timestamp(
        selected_date
    )


    # Find latest available Stage 1 prediction

    stage1_available = stage1[
        stage1["Date"] <= selected_date
    ]


    if len(stage1_available) > 0:

        stage1_row = (
            stage1_available.iloc[-1]
        )

    else:

        stage1_row = None


    # Find latest available Stage 2 prediction

    stage2_available = stage2[
        stage2["Date"] <= selected_date
    ]


    if len(stage2_available) > 0:

        stage2_row = (
            stage2_available.iloc[-1]
        )

    else:

        stage2_row = None


    # Prepare Stage 1 prediction

    if stage1_row is not None:

        stage1_class = int(
            stage1_row["Predicted_Class"]
        )

        stage1_probability_up = float(
            stage1_row["Probability_Up"]
        )


        if stage1_class == 1:

            stage1_direction = "UP"

            stage1_probability = (
                stage1_probability_up
            )

        else:

            stage1_direction = "DOWN"

            stage1_probability = (
                1 - stage1_probability_up
            )

    else:

        stage1_direction = "N/A"

        stage1_probability = 0


    # Prepare Stage 2 prediction

    if stage2_row is not None:

        stage2_class = int(
            stage2_row["Predicted_Class"]
        )

        stage2_probability_up = float(
            stage2_row["Probability_Up"]
        )


        if stage2_class == 1:

            stage2_direction = "UP"

            stage2_probability = (
                stage2_probability_up
            )

        else:

            stage2_direction = "DOWN"

            stage2_probability = (
                1 - stage2_probability_up
            )

    else:

        stage2_direction = "N/A"

        stage2_probability = 0


    # Determine investor instruction

    if stage2_row is not None:

        final_direction = stage2_direction

        final_probability = stage2_probability

    elif stage1_row is not None:

        final_direction = stage1_direction

        final_probability = stage1_probability

    else:

        final_direction = "N/A"

        final_probability = 0


    if final_direction == "UP":

        if final_probability >= 0.65:

            investor_instruction = "BUY"

            investor_explanation = (
                "Strong upward probability. An investor could "
                "consider increasing exposure, subject to "
                "risk tolerance and wider market conditions."
            )

        else:

            investor_instruction = "HOLD"

            investor_explanation = (
                "Upward direction is indicated, but confidence "
                "is not sufficiently strong for a stronger action."
            )


    elif final_direction == "DOWN":

        if final_probability >= 0.65:

            investor_instruction = "SELL"

            investor_explanation = (
                "Strong downward probability. An investor could "
                "consider reducing exposure or adopting a more "
                "defensive position."
            )

        else:

            investor_instruction = "HOLD"

            investor_explanation = (
                "Downward direction is indicated, but confidence "
                "is not sufficiently strong for a stronger action."
            )

    else:

        investor_instruction = "HOLD"

        investor_explanation = (
            "No model prediction is available."
        )


    # Display model results and investor action

    st.subheader(
        f"{market} — Model Comparison"
    )


    stage1_col, stage2_col, action_col = st.columns(
        3
    )


    # Stage 1 card

    with stage1_col:

        with st.container(border=True):

            st.markdown(
                "### Stage 1"
            )

            st.caption(
                "Technical indicators only"
            )

            st.metric(
                "Direction",
                stage1_direction
            )

            if stage1_row is not None:

                st.metric(
                    "Probability",
                    f"{stage1_probability:.1%}"
                )

                st.caption(
                    "Prediction: "
                    +
                    stage1_row["Date"].strftime(
                        "%d %b %Y"
                    )
                )

            else:

                st.caption(
                    "No prediction available"
                )


    # Stage 2 card

    with stage2_col:

        with st.container(border=True):

            st.markdown(
                "### Stage 2"
            )

            st.caption(
                "Technical + behavioural"
            )

            st.metric(
                "Direction",
                stage2_direction
            )

            if stage2_row is not None:

                st.metric(
                    "Probability",
                    f"{stage2_probability:.1%}"
                )

                st.caption(
                    "Prediction: "
                    +
                    stage2_row["Date"].strftime(
                        "%d %b %Y"
                    )
                )

            else:

                st.caption(
                    "No prediction available"
                )


    # Investor action card

    with action_col:

        with st.container(border=True):

            st.markdown(
                "### Investor Action"
            )

            st.metric(
                "Suggested Action",
                investor_instruction
            )

            st.caption(
                investor_explanation
            )


    # Behavioural impact

    if (
        stage1_row is not None
        and stage2_row is not None
    ):

        probability_difference = (
            stage2_probability
            -
            stage1_probability
        )

        direction_changed = (
            stage1_direction
            !=
            stage2_direction
        )


        impact_col1, impact_col2 = st.columns(
            2
        )


        with impact_col1:

            st.metric(
                "Behavioural Probability Impact",
                f"{probability_difference:+.1%}"
            )


        with impact_col2:

            st.metric(
                "Model Direction",
                (
                    "Changed"
                    if direction_changed
                    else "Unchanged"
                )
            )


    # Probability chart

    st.subheader(
        "Five-Day Market Direction Probability"
    )


    fig_probability = go.Figure()


    fig_probability.add_trace(
        go.Scatter(
            x=stage1["Date"],
            y=stage1["Probability_Up"],
            mode="lines",
            name="Stage 1 — Technical",
            line=dict(
                color="#7CC7F5",
                width=2
            )
        )
    )


    fig_probability.add_trace(
        go.Scatter(
            x=stage2["Date"],
            y=stage2["Probability_Up"],
            mode="lines",
            name="Stage 2 — Technical + Behavioural",
            line=dict(
                color="#A78BFA",
                width=2
            )
        )
    )


    fig_probability.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="#777777",
        line_width=1
    )


    fig_probability.add_vline(
        x=selected_date,
        line_dash="dot",
        line_color="#FFFFFF",
        line_width=1
    )


    fig_probability.update_layout(
        height=320,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(
            l=40,
            r=20,
            t=10,
            b=30
        ),
        xaxis_title="Date",
        yaxis_title="Probability",
        yaxis=dict(
            tickformat=".0%",
            range=[0, 1]
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )


    fig_probability.update_xaxes(
        showgrid=False
    )


    fig_probability.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.12)"
    )


    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )


    # Recent prediction history

    st.subheader(
        "Recent Prediction History"
    )


    recent_stage1 = stage1[
        stage1["Date"] <= selected_date
    ].tail(5).copy()


    recent_stage2 = stage2[
        stage2["Date"] <= selected_date
    ].tail(5).copy()


    # Format Stage 1 history

    recent_stage1["Direction"] = (
        recent_stage1["Predicted_Class"]
        .map({
            1: "UP",
            0: "DOWN"
        })
    )


    recent_stage1["Probability"] = (
        recent_stage1["Probability_Up"]
        .where(
            recent_stage1["Predicted_Class"] == 1,
            1 - recent_stage1["Probability_Up"]
        )
    )


    recent_stage1["Probability"] = (
        recent_stage1["Probability"]
        .apply(
            lambda x: f"{x:.1%}"
        )
    )


    recent_stage1["Date"] = (
        recent_stage1["Date"]
        .dt.strftime("%d %b %Y")
    )


    # Format Stage 2 history

    recent_stage2["Direction"] = (
        recent_stage2["Predicted_Class"]
        .map({
            1: "UP",
            0: "DOWN"
        })
    )


    recent_stage2["Probability"] = (
        recent_stage2["Probability_Up"]
        .where(
            recent_stage2["Predicted_Class"] == 1,
            1 - recent_stage2["Probability_Up"]
        )
    )


    recent_stage2["Probability"] = (
        recent_stage2["Probability"]
        .apply(
            lambda x: f"{x:.1%}"
        )
    )


    recent_stage2["Date"] = (
        recent_stage2["Date"]
        .dt.strftime("%d %b %Y")
    )


    # Display prediction history

    history_col1, history_col2 = st.columns(
        2
    )


    with history_col1:

        st.caption(
            "Stage 1 — Technical Only"
        )

        st.dataframe(
            recent_stage1[
                [
                    "Date",
                    "Direction",
                    "Probability"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


    with history_col2:

        st.caption(
            "Stage 2 — Technical + Behavioural"
        )

        st.dataframe(
            recent_stage2[
                [
                    "Date",
                    "Direction",
                    "Probability"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


    # Explain investor interpretation

    st.caption(
        "The suggested investor action is a dashboard "
        "interpretation of model probability, not a class "
        "directly trained by LightGBM. It is intended as "
        "decision-support information rather than financial advice."
    )


# Market and Behaviour page

else:

    st.title(
        "Market & Behaviour"
    )

    st.caption(
        "Explore historical price trends, volatility and "
        "behavioural sentiment alongside technical conditions."
    )


    # Select market

    market = st.selectbox(
        "Market",
        [
            "S&P 500",
            "FTSE 100"
        ]
    )


    # Select market-specific datasets

    if market == "S&P 500":

        market_data = sp500_market.copy()

        sentiment_data = sp500_sentiment.copy()

        sentiment_name = (
            "S&P 500 News Sentiment"
        )

    else:

        market_data = ftse_market.copy()

        sentiment_data = guardian_sentiment.copy()

        sentiment_name = (
            "Guardian News Sentiment"
        )


    # Historical date range

    historical_min = (
        market_data["Date"]
        .min()
        .date()
    )

    historical_max = (
        market_data["Date"]
        .max()
        .date()
    )


    # Select historical dates

    date_range = st.date_input(
        "Historical Date Range",
        value=(
            historical_min,
            historical_max
        ),
        min_value=historical_min,
        max_value=historical_max
    )


    if isinstance(
        date_range,
        tuple
    ):

        start_date = pd.Timestamp(
            date_range[0]
        )

        end_date = pd.Timestamp(
            date_range[1]
        )

    else:

        start_date = pd.Timestamp(
            date_range
        )

        end_date = start_date


    # Filter market data

    filtered_market = market_data[
        (
            market_data["Date"] >= start_date
        )
        &
        (
            market_data["Date"] <= end_date
        )
    ].copy()


    # Calculate moving average

    filtered_market["SMA_20"] = (
        filtered_market["Close"]
        .rolling(20)
        .mean()
    )


    # Calculate Bollinger Bands

    filtered_market["STD_20"] = (
        filtered_market["Close"]
        .rolling(20)
        .std()
    )


    filtered_market["Upper_Band"] = (
        filtered_market["SMA_20"]
        +
        2 * filtered_market["STD_20"]
    )


    filtered_market["Lower_Band"] = (
        filtered_market["SMA_20"]
        -
        2 * filtered_market["STD_20"]
    )


    # Market price chart

    st.subheader(
        "Market Price and Bollinger Bands"
    )


    fig_price = go.Figure()


    fig_price.add_trace(
        go.Scatter(
            x=filtered_market["Date"],
            y=filtered_market["Close"],
            mode="lines",
            name="Closing Price",
            line=dict(
                color="#7CC7F5",
                width=2
            )
        )
    )


    fig_price.add_trace(
        go.Scatter(
            x=filtered_market["Date"],
            y=filtered_market["SMA_20"],
            mode="lines",
            name="20-Day SMA",
            line=dict(
                color="#F2B134",
                width=1.5
            )
        )
    )


    fig_price.add_trace(
        go.Scatter(
            x=filtered_market["Date"],
            y=filtered_market["Upper_Band"],
            mode="lines",
            name="Upper Bollinger Band",
            line=dict(
                color="#888888",
                width=1,
                dash="dash"
            )
        )
    )


    fig_price.add_trace(
        go.Scatter(
            x=filtered_market["Date"],
            y=filtered_market["Lower_Band"],
            mode="lines",
            name="Lower Bollinger Band",
            line=dict(
                color="#888888",
                width=1,
                dash="dash"
            )
        )
    )


    fig_price.update_layout(
        height=400,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(
            l=40,
            r=20,
            t=15,
            b=30
        ),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )


    fig_price.update_xaxes(
        showgrid=False
    )


    fig_price.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.12)"
    )


    st.plotly_chart(
        fig_price,
        use_container_width=True
    )


    # VIX and sentiment charts

    left_col, right_col = st.columns(
        2
    )


    # VIX chart

    with left_col:

        st.subheader(
            "VIX Volatility"
        )


        filtered_vix = vix_data[
            (
                vix_data["Date"] >= start_date
            )
            &
            (
                vix_data["Date"] <= end_date
            )
        ].copy()


        fig_vix = go.Figure()


        fig_vix.add_trace(
            go.Scatter(
                x=filtered_vix["Date"],
                y=filtered_vix["Close"],
                mode="lines",
                name="VIX",
                line=dict(
                    color="#E76F51",
                    width=2
                )
            )
        )


        fig_vix.update_layout(
            height=300,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            margin=dict(
                l=40,
                r=20,
                t=10,
                b=30
            ),
            xaxis_title="Date",
            yaxis_title="VIX",
            showlegend=False
        )


        st.plotly_chart(
            fig_vix,
            use_container_width=True
        )


    # Sentiment chart

    with right_col:

        st.subheader(
            sentiment_name
        )


        filtered_sentiment = sentiment_data[
            (
                sentiment_data["Date"] >= start_date
            )
            &
            (
                sentiment_data["Date"] <= end_date
            )
        ].copy()


        filtered_sentiment["Sentiment_MA"] = (
            filtered_sentiment["daily_sentiment"]
            .rolling(30)
            .mean()
        )


        fig_sentiment = go.Figure()


        fig_sentiment.add_trace(
            go.Scatter(
                x=filtered_sentiment["Date"],
                y=filtered_sentiment["daily_sentiment"],
                mode="lines",
                name="Daily Sentiment",
                line=dict(
                    color="#A78BFA",
                    width=1
                )
            )
        )


        fig_sentiment.add_trace(
            go.Scatter(
                x=filtered_sentiment["Date"],
                y=filtered_sentiment["Sentiment_MA"],
                mode="lines",
                name="30-Day Sentiment",
                line=dict(
                    color="#7CC7F5",
                    width=2
                )
            )
        )


        fig_sentiment.add_hline(
            y=0,
            line_dash="dash",
            line_color="#777777",
            line_width=1
        )


        fig_sentiment.update_layout(
            height=300,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            margin=dict(
                l=40,
                r=20,
                t=10,
                b=30
            ),
            xaxis_title="Date",
            yaxis_title="Sentiment",
            hovermode="x unified"
        )


        st.plotly_chart(
            fig_sentiment,
            use_container_width=True
        )


    # Technical and behavioural context

    st.subheader(
        "Technical vs Behavioural Context"
    )


    # Determine technical context

    if len(filtered_market) > 0:

        latest_market = (
            filtered_market.iloc[-1]
        )

        latest_close = float(
            latest_market["Close"]
        )

        latest_sma = (
            latest_market["SMA_20"]
        )


        if pd.notna(latest_sma):

            if latest_close >= latest_sma:

                technical_context = (
                    "Price above 20-day SMA"
                )

            else:

                technical_context = (
                    "Price below 20-day SMA"
                )

        else:

            technical_context = (
                "Insufficient SMA data"
            )

    else:

        technical_context = (
            "No market data"
        )


    # Determine VIX context

    if len(filtered_vix) > 0:

        latest_vix = float(
            filtered_vix.iloc[-1]["Close"]
        )


        if latest_vix >= 25:

            volatility_context = (
                "High volatility"
            )

        elif latest_vix >= 20:

            volatility_context = (
                "Elevated volatility"
            )

        else:

            volatility_context = (
                "Lower volatility"
            )

    else:

        volatility_context = (
            "No VIX data"
        )


    # Determine sentiment context

    if len(filtered_sentiment) > 0:

        latest_sentiment = float(
            filtered_sentiment.iloc[-1]["daily_sentiment"]
        )


        if latest_sentiment > 0.10:

            behavioural_context = (
                "Positive sentiment"
            )

        elif latest_sentiment < -0.10:

            behavioural_context = (
                "Negative sentiment"
            )

        else:

            behavioural_context = (
                "Neutral sentiment"
            )

    else:

        behavioural_context = (
            "No sentiment data"
        )


    # Display context

    context_col1, context_col2, context_col3 = st.columns(
        3
    )


    with context_col1:

        st.metric(
            "Technical",
            technical_context
        )


    with context_col2:

        st.metric(
            "Market Volatility",
            volatility_context
        )


    with context_col3:

        st.metric(
            "Behaviour",
            behavioural_context
        )


    st.caption(
        "Technical indicators describe price and trend "
        "conditions, while VIX and news sentiment provide "
        "additional behavioural context. Behavioural indicators "
        "are presented as complementary information rather "
        "than a replacement for technical analysis."
    )


    # Recent historical data

    st.subheader(
        "Recent Historical Data"
    )


    display_data = filtered_market[
        [
            "Date",
            "Close",
            "SMA_20",
            "Upper_Band",
            "Lower_Band"
        ]
    ].tail(10).copy()


    display_data["Date"] = (
        display_data["Date"]
        .dt.strftime("%d %b %Y")
    )


    display_data = display_data.round(2)


    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        "Historical market and behavioural indicators are "
        "provided to support exploration of market conditions."
    )