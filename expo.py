import pandas as pd
import pickle
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
import dash
from dash import dcc, html
from dash.dependencies import State
from dash.dependencies import Input, Output
import plotly.express as px

# Load your trained LinearRegression model
with open("modelExpo.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("Exports.csv")
df["Product Group"] = df["Product Group"].str.strip()
df["Partner Name"] = df["Partner Name"].str.strip()

# --- Initial/Default Data Calculations (Used when filters are NOT applied) ---
exports_by_year = df.groupby("Year")["Export (US$ Thousand)"].sum().reset_index()
df_products = df[df["Product Group"] != "All Products"]
top_products_all = (
    df_products.groupby("Product Group")["Export (US$ Thousand)"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

regions = [
    "World",
    "East Asia & Pacific",
    "Europe & Central Asia",
    "Latin America & Caribbean",
    "Middle East & North Africa",
    "North America",
    "South Asia",
    "Sub-Saharan Africa",
]
df["Partner Type"] = df["Partner Name"].apply(
    lambda x: "Region" if x in regions else "Country"
)
regions_df = df[df["Partner Type"] == "Region"].copy()
countries_df = df[df["Partner Type"] == "Country"].copy()

regions_only = regions_df[regions_df["Partner Name"] != "World"]
top_regions_all = (
    regions_only.groupby("Partner Name")["Export (US$ Thousand)"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
top_countries_all = (
    countries_df.groupby("Partner Name")["Export (US$ Thousand)"]
    .sum()
    .nlargest(10)
    .reset_index()
)
top_rca_all = (
    df_products.groupby("Product Group")["Revealed comparative advantage"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

# Recalculate top_growth_products_all for the filtered version
df_products_growth = df_products.sort_values(["Product Group", "Partner Name", "Year"])
df_products_growth["Export Growth Share"] = (
    df_products_growth.groupby(["Product Group", "Partner Name"])[
        "Export (US$ Thousand)"
    ].pct_change()
    * 100
)
df_products_growth = df_products_growth.dropna(
    subset=["Export Product Share (%)", "Export Growth Share"]
)
df_products_growth = df_products_growth[df_products_growth["Export Growth Share"] > 0]
top_growth_products_all = (
    df_products_growth.groupby("Product Group", as_index=False)["Export Growth Share"]
    .mean()
    .sort_values("Export Growth Share", ascending=True)
    .tail(10)
)

# --- Metric Card Default Values (using 'all' data) ---
sorted_exports = exports_by_year.sort_values("Year")
latest_year_row = sorted_exports.iloc[-1] if not sorted_exports.empty else pd.Series({'Export (US$ Thousand)': 0, 'Year': None})
prev_year_value = (
    sorted_exports.iloc[-2]["Export (US$ Thousand)"]
    if len(sorted_exports) > 1
    else None
)
latest_total = latest_year_row["Export (US$ Thousand)"]
yoy = None
if prev_year_value and prev_year_value != 0:
    yoy = (latest_total - prev_year_value) / prev_year_value * 100

top_product_all = top_products_all.iloc[0] if not top_products_all.empty else None
top_region_all = top_regions_all.iloc[0] if not top_regions_all.empty else None
unique_partners_all = countries_df["Partner Name"].nunique()

if yoy is not None:
    yoy_text_all = f"{yoy:+.1f}% vs previous year"
    yoy_class_all = "delta positive" if yoy >= 0 else "delta negative"
else:
    yoy_text_all = "No prior year data"
    yoy_class_all = "delta neutral"

# --- Palette and App setup (unchanged) ---
palette = {
    "background": "#f5f8ff",
    "surface": "#ffffff",
    "primary": "#1f3c88",
    "accent": "#3a5fcd",
    "accent_soft": "#91c7ff",
    "muted": "#5c6f91",
    "gradient": ["#1e3a8a", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"],
}

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# --- CORRECTED LAYOUT (Metric cards reordered to match request) ---
app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="hero",
            children=[
                html.Div(
                    className="hero__header",
                    children=[
                        html.Img(src="/assets/nisr-logo.png", className="hero__logo"),
                        html.Div(
                            className="hero__text",
                            children=[
                                html.H1(
                                    "Navigate Rwanda's Export", className="hero__title"
                                ),
                                html.P(
                                    "Explore long-term growth, standout products, and strategic partners",
                                    className="hero__subtitle",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="metric-grid",
            children=[
                # 1. EXPORTS
                html.Div(
                    className="metric-card",
                    children=[
                        html.Span("Total Exports", className="card-title"),
                        html.Span(id="metric-total", children=f"${latest_total:,.0f}K", className="metric"),
                        html.Span(id="metric-yoy", children=yoy_text_all, className=yoy_class_all),
                    ],
                ),
                # 2. TOP PRODUCT
                html.Div(
                    className="metric-card",
                    children=[
                        html.Span("Top Product", className="card-title"),
                        html.Span(
                            id="metric-top-product-name",
                            children=(top_product_all["Product Group"] if top_product_all is not None else "—"),
                            className="metric-small",
                        ),
                        html.Span(
                            id="metric-top-product-value",
                            children=(f"${top_product_all['Export (US$ Thousand)']:,.0f}K" if top_product_all is not None else "No data"),
                            className="delta neutral",
                        ),
                    ],
                ),
                # 3. RCA PRODUCT
                html.Div(
                    className="metric-card",
                    children=[
                        html.Span("Top RCA Product", className="card-title"),
                        html.Span(
                            id="metric-rca-product",
                            children=(top_rca_all.iloc[0]["Product Group"] if not top_rca_all.empty else "—"),
                            className="metric-small",
                        ),
                        html.Span(
                            id="metric-rca-value",
                            children=(f"{top_rca_all.iloc[0]['Revealed comparative advantage']:.2f}" if not top_rca_all.empty else "No data"),
                            className="delta neutral",
                        ),
                    ],
                ),
                # 4. REGION
                html.Div(
                    className="metric-card",
                    children=[
                        html.Span("Top Region", className="card-title"),
                        html.Span(
                            id="metric-top-region-name",
                            children=(top_region_all["Partner Name"] if top_region_all is not None else "—"),
                            className="metric-small",
                        ),
                        html.Span(
                            id="metric-top-region-value",
                            children=(f"${top_region_all['Export (US$ Thousand)']:,.0f}K" if top_region_all is not None else "No data"),
                            className="delta neutral",
                        ),
                    ],
                ),
                # 5. PARTNERS
                html.Div(
                    className="metric-card",
                    children=[
                        html.Span("Partners", className="card-title"),
                        html.Span(id="metric-partners", children=f"{unique_partners_all}", className="metric"),
                        html.Span("countries", className="delta neutral"),
                    ],
                ),
                # 6. GDP
                html.Div(
                    className="metric-card",
                    children=[
                        html.Span("Rwanda GDP", className="card-title"),
                        html.Span(
                            id="metric-gdp-current",
                            children=f"${10_500_000_000/1_000_000_000:.1f}B",
                            className="metric",
                        ),
                        html.Span("Nominal USD", className="delta neutral"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="main-content",
            children=[
                html.Div(
                    className="sidebar",
                    children=[
                        html.Div(className="sidebar-title", children="Filters"),
                        html.Label("Select Year"),
                        dcc.Dropdown(
                            id="year-filter",
                            options=[{"label": y, "value": y} for y in sorted(df["Year"].unique())],
                            value=None,
                            placeholder="All Years",
                            clearable=True,
                        ),
                        html.Hr(),
                        html.Div(
                            style={"margin-bottom": "30px"},
                            children=[
                                html.Div("Analytics 📊", className="sidebar-card-title"),
                                html.Div(
                                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                                    children=[
                                        html.Div(id="nav-export-trends", className="nav-item nav-item-active", children="📊 Trends", n_clicks=0),
                                        html.Div(id="nav-top5-trend", className="nav-item", children="⭐ Top 5 Trend", n_clicks=0),
                                        html.Div(id="nav-top-products", className="nav-item", children="🛍️ Top Products", n_clicks=0),
                                        html.Div(id="nav-top-regions", className="nav-item", children="🌍 Top Regions", n_clicks=0),
                                        html.Div(id="nav-top-countries", className="nav-item", children="🏳️ Top Countries", n_clicks=0),
                                        html.Div(id="nav-rca-analysis", className="nav-item", children="📈 RCA Analysis", n_clicks=0),
                                        html.Div(id="nav-growth-products", className="nav-item", children="📈 Growth Products", n_clicks=0),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style={"margin-bottom": "30px"},
                            children=[
                                html.Div("Forecast & Growth 📈", className="sidebar-card-title"),
                                html.Div(
                                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                                    children=[
                                        html.Div(id="nav-predict-exports", className="nav-item", children="🔮 Predict Exports", n_clicks=0),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style={"margin-bottom": "30px"},
                            children=[
                                html.Div("Recommendations 💡", className="sidebar-card-title"),
                                html.Div(
                                    style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "10px"},
                                    children=[
                                        html.Div(id="nav-recommendation", className="nav-item", children="💡 Recommendations", n_clicks=0),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="content-area",
                    children=[
                        html.Div(id="chart-content", className="chart-container"),
                    ],
                ),
            ],
        ),
        dcc.Store(id="active-tab", data="nav-export-trends"),
    ],
)


def stylize(fig):
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=palette["surface"],
        paper_bgcolor=palette["surface"],
        font=dict(
            color=palette["primary"], family="Inter, 'Segoe UI', sans-serif", size=11
        ),
        title_font=dict(color=palette["primary"], size=16, weight=600),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
            linecolor=palette["muted"],
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
            linecolor=palette["muted"],
        ),
        margin=dict(l=50, r=30, t=50, b=40),
        height=450,
    )
    return fig

# --- CALLBACK (Output/Return Order already fixed and remains correct) ---
@app.callback(
[
Output("metric-total", "children"),          # 0. EXPORTS (Value)
Output("metric-yoy", "children"),            # 1. EXPORTS (YoY Text)
Output("metric-yoy", "className"),           # 2. EXPORTS (YoY Class)
Output("metric-top-product-name", "children"),# 3. TOP PRODUCT (Name)
Output("metric-top-product-value", "children"),# 4. TOP PRODUCT (Value)
Output("metric-rca-product", "children"),    # 5. RCA (Name)
Output("metric-rca-value", "children"),      # 6. RCA (Value)
Output("metric-top-region-name", "children"),# 7. REGION (Name)
Output("metric-top-region-value", "children"),# 8. REGION (Value)
Output("metric-partners", "children"),       # 9. PARTNERS (Count)
Output("metric-gdp-current", "children"),    # 10. GDP (Value)
],
[
Input("year-filter", "value"),
],
)
def update_metrics(year_value):
    metric_df = df.copy()

    if year_value is not None and year_value != "Overall":
        metric_df = metric_df[metric_df["Year"] == year_value]

    # --- SETUP CALCULATION DATA FRAMES ---
    products_df = metric_df[metric_df["Product Group"] != "All Products"]
    countries_df = metric_df[metric_df["Partner Type"] == "Country"]
    regions_df_filtered = metric_df[metric_df["Partner Type"] == "Region"].copy()
    regions_df_filtered = regions_df_filtered[regions_df_filtered["Partner Name"] != "World"]


    # 1. Exports Calculation
    exports_by_year = metric_df.groupby("Year")["Export (US$ Thousand)"].sum().reset_index()
    sorted_exports = exports_by_year.sort_values("Year")
    latest_total = 0
    yoy_text = "No data"
    yoy_class = "delta neutral"

    if not sorted_exports.empty:
        latest_total = sorted_exports.iloc[-1]["Export (US$ Thousand)"]
        if len(sorted_exports) > 1:
            prev_year = sorted_exports.iloc[-2]["Export (US$ Thousand)"]
            if prev_year != 0:
                yoy = (latest_total - prev_year) / prev_year * 100
                yoy_text = f"{yoy:+.1f}% vs previous year"
                yoy_class = "delta positive" if yoy >= 0 else "delta negative"
            else:
                yoy_text = "No prior year data"
        else:
            yoy_text = f"Data for {year_value}" if year_value else "Single year data"

    # 2. Top Product Calculation
    top_product = (
        products_df.groupby("Product Group")["Export (US$ Thousand)"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
        .reset_index()
    )
    top_product_name = top_product.iloc[0]["Product Group"] if not top_product.empty else "—"
    top_product_value = f"${top_product.iloc[0]['Export (US$ Thousand)']:,.0f}K" if not top_product.empty else "No data"

    # 3. RCA Calculation
    top_rca = (
        products_df.groupby("Product Group")["Revealed comparative advantage"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
        .reset_index()
    )
    rca_product_name = top_rca.iloc[0]["Product Group"] if not top_rca.empty else "—"
    rca_value = f"{top_rca.iloc[0]['Revealed comparative advantage']:.2f}" if not top_rca.empty else "No data"

    # 4. Region Calculation
    top_region = (
        regions_df_filtered.groupby("Partner Name")["Export (US$ Thousand)"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
        .reset_index()
    )
    top_region_name = top_region.iloc[0]["Partner Name"] if not top_region.empty else "—"
    top_region_value = f"${top_region.iloc[0]['Export (US$ Thousand)']:,.0f}K" if not top_region.empty else "No data"

    # 5. Partners Calculation
    unique_partners_count = countries_df['Partner Name'].nunique()

    # 6. GDP Calculation
    gdp_current = 0
    if 'GDP_Current_USD' in df.columns:
        if year_value is None or year_value == "Overall":
            if not df.empty:
                latest_gdp_year = df['Year'].max()
                gdp_current_series = df[df['Year'] == latest_gdp_year]['GDP_Current_USD'].dropna()
                gdp_current = gdp_current_series.iloc[0] if not gdp_current_series.empty else 0
        else:
            filtered_gdp = df[df['Year'] == year_value]
            gdp_current_series = filtered_gdp['GDP_Current_USD'].dropna()
            gdp_current = gdp_current_series.iloc[0] if not gdp_current_series.empty else 0

    if gdp_current == 0:
        gdp_display = f"${10_500_000_000/1_000_000_000:.1f}B"
        if 'GDP_Current_USD' in df.columns:
             gdp_display = "No data"
    else:
        gdp_display = f"${gdp_current / 1_000_000_000:.1f}B"


    # --- FINAL RETURN STATEMENT (Order matches the Output list: Exports, Top Product, RCA, Region, Partner, GDP) ---
    return (
        # EXPORTS (3)
        f"${latest_total:,.0f}K",         # 0. metric-total
        yoy_text,                         # 1. metric-yoy (text)
        yoy_class,                        # 2. metric-yoy (class)

        # TOP PRODUCT (2)
        top_product_name,                 # 3. metric-top-product-name
        top_product_value,                # 4. metric-top-product-value

        # RCA (2)
        rca_product_name,                 # 5. metric-rca-product
        rca_value,                        # 6. metric-rca-value

        # REGION (2)
        top_region_name,                  # 7. metric-top-region-name
        top_region_value,                 # 8. metric-top-region-value

        # PARTNERS (1)
        f"{unique_partners_count}",       # 9. metric-partners

        # GDP (1)
        gdp_display,                      # 10. metric-gdp-current
    )
# --- CALLBACK TO UPDATE CHART (Conditional Filter Logic Implemented) ---
@app.callback(
    [
        Output("chart-content", "children"),
        Output("active-tab", "data"),
        Output("nav-export-trends", "className"),
        Output("nav-top5-trend", "className"),
        Output("nav-top-products", "className"),
        Output("nav-top-regions", "className"),
        Output("nav-top-countries", "className"),
        Output("nav-rca-analysis", "className"),
        Output("nav-growth-products", "className"),
        Output("nav-predict-exports", "className"),
        Output("nav-recommendation", "className"),
    ],
    [
        Input("nav-export-trends", "n_clicks"),
        Input("nav-top5-trend", "n_clicks"),
        Input("nav-top-products", "n_clicks"),
        Input("nav-top-regions", "n_clicks"),
        Input("nav-top-countries", "n_clicks"),
        Input("nav-rca-analysis", "n_clicks"),
        Input("nav-growth-products", "n_clicks"),
        Input("nav-predict-exports", "n_clicks"),
        Input("nav-recommendation", "n_clicks"),
        Input("year-filter", "value"),
    ],
    [State("active-tab", "data")]
)
def update_chart(
    trends_clicks, trend_clicks, products_clicks, regions_clicks, countries_clicks,
    rca_clicks, growth_clicks, predict_clicks, recommendation_clicks, year_value, active_tab_state
):
    ctx = dash.callback_context
    filter_tabs = ["nav-top-products", "nav-top-regions", "nav-top-countries", "nav-rca-analysis", "nav-growth-products"]
    unfiltered_tabs = ["nav-export-trends", "nav-top5-trend", "nav-predict-exports"]
    
    # Ensure button_id is valid
    button_id = active_tab_state if isinstance(active_tab_state, str) else "nav-export-trends"
    
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id.startswith("nav-"):
            button_id = prop_id
        elif prop_id == "year-filter":
            button_id = active_tab_state if isinstance(active_tab_state, str) else "nav-export-trends"
    
    # Keep predict tab active if year filter changes
    if button_id == "nav-predict-exports" and prop_id == "year-filter":
        button_id = "nav-predict-exports"
    
    new_active_tab_state = button_id
    nav_ids = [
        "nav-export-trends", "nav-top5-trend", "nav-top-products", "nav-top-regions",
        "nav-top-countries", "nav-rca-analysis", "nav-growth-products", "nav-predict-exports",
        "nav-recommendation",
    ]
    classes = ["nav-item"] * 9
    try:
        classes[nav_ids.index(button_id)] = "nav-item nav-item-active"
    except ValueError:
        button_id = "nav-export-trends"
        classes[0] = "nav-item nav-item-active"
    
    df_chart = df.copy()
    if button_id in filter_tabs and year_value is not None:
        df_chart = df_chart[df_chart["Year"] == year_value]
    
    if button_id == "nav-top5-trend":
        df_top = df[df["Product Group"] != "All Products"]
        top_products_list = df_top.groupby("Product Group")["Export (US$ Thousand)"].sum().sort_values(ascending=False).head(5).index.tolist()
        df_top5 = df_top[df_top["Product Group"].isin(top_products_list)]
        exports_top = df_top5.groupby(["Year", "Product Group"])["Export (US$ Thousand)"].sum().reset_index()
        fig = px.line(
            exports_top, x="Year", y="Export (US$ Thousand)", color="Product Group", markers=True,
            title="Top 5 Export Products Trend (Overall Data)"
        )
        fig.update_layout(legend_title_text="Product Group", xaxis_title="Year", yaxis_title="Export (US$ Thousand)", template="plotly_white")
        fig = stylize(fig)
        chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-top-products":
        df_products_filtered = df_chart[df_chart["Product Group"] != "All Products"]
        top_products_filtered = df_products_filtered.groupby("Product Group")["Export (US$ Thousand)"].sum().sort_values(ascending=False).head(10).reset_index()
        if top_products_filtered.empty:
            chart = html.Div("No data available for selected year", style={"textAlign": "center", "color": palette["primary"]})
        else:
            fig = px.bar(
                top_products_filtered, x="Export (US$ Thousand)", y="Product Group", orientation="h",
                title="Top 10 Export Products", color_discrete_sequence=[palette["accent"]]
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            fig.update_traces(marker_line_color=palette["accent_soft"], marker_line_width=1.5)
            fig = stylize(fig)
            chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-top-regions":
        regions_df_filtered = df_chart[df_chart["Partner Type"] == "Region"].copy()
        regions_only_filtered = regions_df_filtered[regions_df_filtered["Partner Name"] != "World"]
        top_regions_filtered = regions_only_filtered.groupby("Partner Name")["Export (US$ Thousand)"].sum().sort_values(ascending=False).reset_index()
        if top_regions_filtered.empty:
            chart = html.Div("No data available for selected year", style={"textAlign": "center", "color": palette["primary"]})
        else:
            fig = px.bar(
                top_regions_filtered, y="Export (US$ Thousand)", x="Partner Name", title="Rwanda Exports by Region",
                color_discrete_sequence=[palette["accent"]]
            )
            fig.update_layout(xaxis=dict(categoryorder="total ascending"))
            fig.update_traces(marker_line_color=palette["accent_soft"], marker_line_width=1.5)
            fig = stylize(fig)
            chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-top-countries":
        countries_df_filtered = df_chart[df_chart["Partner Type"] == "Country"].copy()
        top_countries_filtered = countries_df_filtered.groupby("Partner Name")["Export (US$ Thousand)"].sum().nlargest(10).reset_index()
        if top_countries_filtered.empty:
            chart = html.Div("No data available for selected year", style={"textAlign": "center", "color": palette["primary"]})
        else:
            fig = px.bar(
                top_countries_filtered, x="Export (US$ Thousand)", y="Partner Name", title="Top 10 Export Partner Countries",
                color_discrete_sequence=[palette["accent"]]
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            fig.update_traces(marker_line_color=palette["accent_soft"], marker_line_width=1.5)
            fig = stylize(fig)
            chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-rca-analysis":
        df_products_filtered = df_chart[df_chart["Product Group"] != "All Products"]
        top_rca_filtered = df_products_filtered.groupby("Product Group")["Revealed comparative advantage"].mean().sort_values(ascending=False).head(10).reset_index()
        if top_rca_filtered.empty:
            chart = html.Div("No data available for selected year", style={"textAlign": "center", "color": palette["primary"]})
        else:
            fig = px.bar(
                top_rca_filtered, x="Revealed comparative advantage", y="Product Group", orientation="h",
                title="Top 10 Products by Revealed Comparative Advantage", color_discrete_sequence=[palette["accent"]]
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            fig.update_traces(marker_line_color=palette["accent_soft"], marker_line_width=1.5)
            fig = stylize(fig)
            chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-growth-products":
        df_growth_calc = df_chart.copy()
        df_products_growth_filtered = df_growth_calc[df_growth_calc["Product Group"] != "All Products"]
        if year_value is not None:
            top_products_filtered = df_products_growth_filtered.groupby("Product Group")["Export (US$ Thousand)"].sum().sort_values(ascending=False).head(10).reset_index()
            if top_products_filtered.empty:
                chart = html.Div("No data available for selected year", style={"textAlign": "center", "color": palette["primary"]})
            else:
                fig = px.bar(
                    top_products_filtered, x="Export (US$ Thousand)", y="Product Group", orientation="h",
                    title=f"Top 10 Products by Export Value ({year_value}) (Growth requires multiple years)",
                    color_discrete_sequence=[palette["accent"]]
                )
                fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                fig.update_traces(marker_line_color=palette["accent_soft"], marker_line_width=1.5)
                fig = stylize(fig)
                chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
        else:
            df_products_growth_filtered = df_products_growth_filtered.sort_values(["Product Group", "Partner Name", "Year"])
            df_products_growth_filtered["Export Growth Share"] = df_products_growth_filtered.groupby(["Product Group", "Partner Name"])["Export (US$ Thousand)"].pct_change() * 100
            df_products_growth_filtered = df_products_growth_filtered.dropna(subset=["Export Product Share (%)", "Export Growth Share"])
            df_products_growth_filtered = df_products_growth_filtered[df_products_growth_filtered["Export Growth Share"] > 0]
            top_growth_products_filtered = df_products_growth_filtered.groupby("Product Group", as_index=False)["Export Growth Share"].mean().sort_values("Export Growth Share", ascending=True).tail(10)
            if top_growth_products_filtered.empty:
                chart = html.Div("No growth data available", style={"textAlign": "center", "color": palette["primary"]})
            else:
                fig = px.bar(
                    top_growth_products_filtered, x="Export Growth Share", y="Product Group", orientation="h",
                    title="Top 10 Products by Average Export Growth (%)", color_discrete_sequence=[palette["accent"]]
                )
                fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                fig.update_traces(marker_line_color=palette["accent_soft"], marker_line_width=1.5)
                fig = stylize(fig)
                chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-predict-exports":
        chart = html.Div(
            [
                html.H2("📈 EXPORT PREDICTION - RWANDA", style={"textAlign": "center", "color": palette["primary"], "margin": "5px 0 20px 0", "fontSize": "26px", "fontWeight": "700"}),
                html.Div(
                    [
                        html.H4("1️⃣ Select Time & Product", style={"color": palette["primary"], "fontSize": "14px"}),
                        html.Label("Year for Prediction:"),
                        dcc.Dropdown(id="year-dropdown", options=[{"label": str(y), "value": y} for y in range(2022, 2027)], value=2025, clearable=False),
                        html.Br(),
                        html.Label("Product Category:"),
                        dcc.Dropdown(id="product-dropdown", options=[{"label": x, "value": x} for x in sorted(df["Product Group"].unique())], value=sorted(df["Product Group"].unique())[0]),
                    ],
                    className="prediction-card", style={"marginBottom": "15px"}
                ),
                html.Div(
                    [
                        html.H4("2️⃣ Economic Indicators (Rwanda-specific)", style={"color": palette["primary"], "fontSize": "14px"}),
                        html.Label("Global Market Growth (%)"),
                        dcc.Slider(id="world-growth-input", min=0, max=6, step=0.1, value=3.5, marks={i: f"{i}%" for i in range(0, 7)}),
                        html.Small("Expected growth of the global market relevant to this product. Higher values indicate stronger international demand.", style={"color": "gray", "fontSize": 12, "display": "block", "marginTop": "4px"}),
                        html.Br(),
                        html.Label("Rwanda Country Economy Growth (%)"),
                        dcc.Slider(id="country-growth-input", min=5, max=10, step=0.1, value=8.0, marks={i: f"{i}%" for i in range(5, 11)}),
                        html.Small("Annual GDP growth rate of Rwanda. Higher growth may positively impact exports.", style={"color": "gray", "fontSize": 12, "display": "block", "marginTop": "4px"}),
                    ],
                    className="prediction-card", style={"marginBottom": "15px"}
                ),
                html.Div(
                    [
                        html.H4("3️⃣ Rwanda GDP Information", style={"color": palette["primary"], "fontSize": "14px"}),
                        html.Label("Nominal GDP (USD)"),
                        dcc.Input(id="gdp-current-input", type="number", value=14_300_000_000, step=100_000_000, style={"width": "100%"}),
                        html.Div("Example: 2023 ($14.1B), 2024 ($14.3B est.)", style={"fontSize": 12, "color": "gray"}),
                        html.Br(),
                        html.Label("GDP Growth Rate (%)"),
                        dcc.Slider(id="gdp-growth-input", min=5, max=10, step=0.1, value=7.0, marks={i: f"{i}%" for i in range(5, 11)}),
                    ],
                    className="prediction-card"
                ),
                html.Br(),
                html.Div(
                    html.Button(
                        "🔮 Predict Export Value for Rwanda", id="predict-button", n_clicks=0,
                        style={"backgroundColor": palette["accent"], "color": "white", "padding": "12px 25px", "fontWeight": "bold", "border": "none", "borderRadius": "10px", "cursor": "pointer", "fontSize": "16px", "boxShadow": "0px 2px 5px rgba(0,0,0,0.2)"}
                    ),
                    style={"textAlign": "center", "marginTop": "20px"}
                ),
                html.Div(id="prediction-output", style={"marginTop": "20px", "textAlign": "center", "fontSize": "22px", "fontWeight": "600", "color": palette["primary"], "padding": "15px", "backgroundColor": palette["surface"], "borderRadius": "12px", "boxShadow": "0px 3px 8px rgba(0,0,0,0.1)", "width": "70%", "margin": "20px auto"}),
            ],
            style={"padding": "15px", "maxHeight": "calc(100vh - 80px)", "overflow": "hidden", "display": "flex", "flexDirection": "column", "justifyContent": "flex-start"}
        )
    
    elif button_id == "nav-recommendation":
        chart = html.Div(
            [
                html.H2("💡 Strategic Recommendations", style={"textAlign": "center", "color": palette["primary"], "marginBottom": "30px"}),
                html.Div(
                    [
                        html.Div([html.Span("🚀", style={"fontSize": "24px", "marginRight": "10px"}), html.Span("Focus on top growing products")], className="recommendation-card"),
                        html.Div([html.Span("🌍", style={"fontSize": "24px", "marginRight": "10px"}), html.Span("Explore new markets in Sub-Saharan Africa")], className="recommendation-card"),
                        html.Div([html.Span("💰", style={"fontSize": "24px", "marginRight": "10px"}), html.Span("Invest in value addition and processing")], className="recommendation-card"),
                        html.Div([html.Span("📊", style={"fontSize": "24px", "marginRight": "10px"}), html.Span("Monitor RCA and market share regularly")], className="recommendation-card"),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))", "gap": "20px", "marginTop": "20px"}
                ),
                html.Div(
                    [
                        html.P(
                            "These recommendations are designed to help optimize Rwanda's export strategy. "
                            "Focus on products with high growth potential, explore new markets, invest in value addition, "
                            "and regularly monitor performance metrics such as RCA and market share.",
                            style={"fontSize": "16px", "color": palette["muted"], "marginTop": "30px"}
                        )
                    ]
                ),
            ],
            style={"padding": "30px"}
        )
    
    else:
        fig = px.line(
            exports_by_year, x="Year", y="Export (US$ Thousand)", markers=True,
            title="Rwanda Export Trend Over Time (Overall Data)"
        )
        fig.update_traces(line=dict(color=palette["accent"], width=3), marker=dict(size=8, color=palette["accent_soft"], line=dict(color=palette["accent"], width=2)))
        fig = stylize(fig)
        chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    return (
        chart,
        new_active_tab_state,
        classes[0],
        classes[1],
        classes[2],
        classes[3],
        classes[4],
        classes[5],
        classes[6],
        classes[7],
        classes[8],
    )

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("year-dropdown", "value"),
    State("product-dropdown", "value"),
    State("gdp-current-input", "value"),
    State("gdp-growth-input", "value"),
    State("world-growth-input", "value"),
    State("country-growth-input", "value"),
)
def predict_exports(n_clicks, year, product, gdp_current, gdp_growth, world_growth, country_growth):
    if n_clicks == 0:
        return ""

    # Validate inputs
    if None in [year, product, gdp_current, gdp_growth, world_growth, country_growth]:
        return "⚠️ Please fill in all the fields before predicting."
    
    # Prepare your input features in the same order your model expects
    X_new = pd.DataFrame([{
        "Year": year,
        "Product Group": product,
        "World Growth (%)": world_growth,
        "Country Growth (%)": country_growth,
        "GDP_Current_USD": gdp_current,
        "GDP_Growth_Percent": gdp_growth,
    }])
    
    # Make prediction
    prediction = model.predict(X_new)[0]
    
    # Return nicely formatted output
    return f"🔮 Predicted Export Value for {product} in {year}: ${prediction:,.0f}K"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  # Use Render's assigned port
    app.run(debug=False, host='0.0.0.0', port=port) # Or simply app.run(debug=False)