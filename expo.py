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
from dash.dependencies import State, Input, Output
import plotly.express as px

# Load your trained LinearRegression model
with open("modelExpo.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("Exports.csv")
df["Product Group"] = df["Product Group"].str.strip()
df["Partner Name"] = df["Partner Name"].str.strip()

# --- Initial/Default Data Calculations ---
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

# Calculate growth products
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

# --- Metric Card Default Values ---
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

# --- App Setup ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# --- LAYOUT ---
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
                        html.Div(className="metric-icon", children="💰"),
                        html.Span("Total Exports", className="card-title"),
                        html.Span(id="metric-total", children=f"${latest_total:,.0f}K", className="metric"),
                        html.Span(id="metric-yoy", children=yoy_text_all, className=yoy_class_all),
                    ],
                ),
                # 2. TOP PRODUCT
                html.Div(
                    className="metric-card",
                    children=[
                        html.Div(className="metric-icon", children="📦"),
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
                        html.Div(className="metric-icon", children="⭐"),
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
                        html.Div(className="metric-icon", children="🌍"),
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
                        html.Div(className="metric-icon", children="🤝"),
                        html.Span("Partners", className="card-title"),
                        html.Span(id="metric-partners", children=f"{unique_partners_all}", className="metric"),
                        html.Span("countries", className="delta neutral"),
                    ],
                ),
                # 6. GDP
                html.Div(
                    className="metric-card",
                    children=[
                        html.Div(className="metric-icon", children="📊"),
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
                        html.Div(className="sidebar-title", children="Filters & Navigation"),
                        html.Label("Select Year", className="filter-label"),
                        dcc.Dropdown(
                            id="year-filter",
                            options=[{"label": y, "value": y} for y in sorted(df["Year"].unique())],
                            value=None,
                            placeholder="All Years",
                            clearable=True,
                            className="year-dropdown"
                        ),
                        html.Hr(className="sidebar-divider"),
                        html.Div(
                            className="nav-section",
                            children=[
                                html.Div("Analytics 📊", className="sidebar-section-title"),
                                html.Div(
                                    className="nav-grid",
                                    children=[
                                        html.Div(id="nav-export-trends", className="nav-item nav-item-active", children="📈 Trends", n_clicks=0),
                                        html.Div(id="nav-top5-trend", className="nav-item", children="⭐ Top 5", n_clicks=0),
                                        html.Div(id="nav-top-products", className="nav-item", children="🛍 Products", n_clicks=0),
                                        html.Div(id="nav-top-regions", className="nav-item", children="🌍 Regions", n_clicks=0),
                                        html.Div(id="nav-top-countries", className="nav-item", children="🏳 Countries", n_clicks=0),
                                        html.Div(id="nav-rca-analysis", className="nav-item", children="📊 RCA", n_clicks=0),
                                        html.Div(id="nav-growth-products", className="nav-item", children="📈 Growth", n_clicks=0),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="nav-section",
                            children=[
                                html.Div("Forecast 🔮", className="sidebar-section-title"),
                                html.Div(
                                    className="nav-grid",
                                    children=[
                                        html.Div(id="nav-predict-exports", className="nav-item", children="🔮 Predict", n_clicks=0),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="nav-section",
                            children=[
                                html.Div("Insights 💡", className="sidebar-section-title"),
                                html.Div(
                                    className="nav-grid-single",
                                    children=[
                                        html.Div(id="nav-recommendation", className="nav-item", children="💡 SME Strategy", n_clicks=0),
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
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="Inter, 'Segoe UI', sans-serif", size=12),
        title_font=dict(color="#0F172A", size=18, weight=600),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(15, 23, 42, 0.08)",
            zeroline=False,
            linecolor="#CBD5E1",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(15, 23, 42, 0.08)",
            zeroline=False,
            linecolor="#CBD5E1",
        ),
        margin=dict(l=50, r=30, t=60, b=40),
        height=480,
    )
    return fig


@app.callback(
    [
        Output("metric-total", "children"),
        Output("metric-yoy", "children"),
        Output("metric-yoy", "className"),
        Output("metric-top-product-name", "children"),
        Output("metric-top-product-value", "children"),
        Output("metric-rca-product", "children"),
        Output("metric-rca-value", "children"),
        Output("metric-top-region-name", "children"),
        Output("metric-top-region-value", "children"),
        Output("metric-partners", "children"),
        Output("metric-gdp-current", "children"),
    ],
    [Input("year-filter", "value")],
)
def update_metrics(year_value):
    metric_df = df.copy()

    if year_value is not None and year_value != "Overall":
        metric_df = metric_df[metric_df["Year"] == year_value]

    products_df = metric_df[metric_df["Product Group"] != "All Products"]
    countries_df = metric_df[metric_df["Partner Type"] == "Country"]
    regions_df_filtered = metric_df[metric_df["Partner Type"] == "Region"].copy()
    regions_df_filtered = regions_df_filtered[regions_df_filtered["Partner Name"] != "World"]

    # Exports Calculation
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

    # Top Product
    top_product = (
        products_df.groupby("Product Group")["Export (US$ Thousand)"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
        .reset_index()
    )
    top_product_name = top_product.iloc[0]["Product Group"] if not top_product.empty else "—"
    top_product_value = f"${top_product.iloc[0]['Export (US$ Thousand)']:,.0f}K" if not top_product.empty else "No data"

    # RCA
    top_rca = (
        products_df.groupby("Product Group")["Revealed comparative advantage"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
        .reset_index()
    )
    rca_product_name = top_rca.iloc[0]["Product Group"] if not top_rca.empty else "—"
    rca_value = f"{top_rca.iloc[0]['Revealed comparative advantage']:.2f}" if not top_rca.empty else "No data"

    # Region
    top_region = (
        regions_df_filtered.groupby("Partner Name")["Export (US$ Thousand)"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
        .reset_index()
    )
    top_region_name = top_region.iloc[0]["Partner Name"] if not top_region.empty else "—"
    top_region_value = f"${top_region.iloc[0]['Export (US$ Thousand)']:,.0f}K" if not top_region.empty else "No data"

    # Partners
    unique_partners_count = countries_df['Partner Name'].nunique()

    # GDP
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

    return (
        f"${latest_total:,.0f}K",
        yoy_text,
        yoy_class,
        top_product_name,
        top_product_value,
        rca_product_name,
        rca_value,
        top_region_name,
        top_region_value,
        f"{unique_partners_count}",
        gdp_display,
    )


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
    
    button_id = active_tab_state if isinstance(active_tab_state, str) else "nav-export-trends"
    
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id.startswith("nav-"):
            button_id = prop_id
        elif prop_id == "year-filter":
            button_id = active_tab_state if isinstance(active_tab_state, str) else "nav-export-trends"
    
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
    
    # CHARTS GENERATION
    if button_id == "nav-top5-trend":
        df_top = df[df["Product Group"] != "All Products"]
        top_products_list = df_top.groupby("Product Group")["Export (US$ Thousand)"].sum().sort_values(ascending=False).head(5).index.tolist()
        df_top5 = df_top[df_top["Product Group"].isin(top_products_list)]
        exports_top = df_top5.groupby(["Year", "Product Group"])["Export (US$ Thousand)"].sum().reset_index()
        fig = px.line(
            exports_top, x="Year", y="Export (US$ Thousand)", color="Product Group", markers=True,
            title="Top 5 Export Products Trend"
        )
        fig = stylize(fig)
        chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-top-products":
        df_products_filtered = df_chart[df_chart["Product Group"] != "All Products"]
        top_products_filtered = df_products_filtered.groupby("Product Group")["Export (US$ Thousand)"].sum().sort_values(ascending=False).head(10).reset_index()
        if top_products_filtered.empty:
            chart = html.Div("No data available for selected year", className="no-data-message")
        else:
            fig = px.bar(
                top_products_filtered, x="Export (US$ Thousand)", y="Product Group", orientation="h",
                title="Top 10 Export Products", color_discrete_sequence=["#3B82F6"]
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            fig = stylize(fig)
            chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-top-regions":
        regions_df_filtered = df_chart[df_chart["Partner Type"] == "Region"].copy()
        regions_only_filtered = regions_df_filtered[regions_df_filtered["Partner Name"] != "World"]
        top_regions_filtered = regions_only_filtered.groupby("Partner Name")["Export (US$ Thousand)"].sum().sort_values(ascending=False).reset_index()
        if top_regions_filtered.empty:
            chart = html.Div("No data available for selected year", className="no-data-message")
        else:
            fig = px.bar(
                top_regions_filtered, y="Export (US$ Thousand)", x="Partner Name", title="Rwanda Exports by Region",
                color_discrete_sequence=["#3B82F6"]
            )
            fig = stylize(fig)
            chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-top-countries":
        countries_df_filtered = df_chart[df_chart["Partner Type"] == "Country"].copy()
        top_countries_filtered = countries_df_filtered.groupby("Partner Name")["Export (US$ Thousand)"].sum().nlargest(10).reset_index()
        if top_countries_filtered.empty:
            chart = html.Div("No data available for selected year", className="no-data-message")
        else:
            fig = px.bar(
                top_countries_filtered, x="Export (US$ Thousand)", y="Partner Name", title="Top 10 Export Partner Countries",
                color_discrete_sequence=["#3B82F6"]
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            fig = stylize(fig)
            chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-rca-analysis":
        df_products_filtered = df_chart[df_chart["Product Group"] != "All Products"]
        top_rca_filtered = df_products_filtered.groupby("Product Group")["Revealed comparative advantage"].mean().sort_values(ascending=False).head(10).reset_index()
        if top_rca_filtered.empty:
            chart = html.Div("No data available for selected year", className="no-data-message")
        else:
            fig = px.bar(
                top_rca_filtered, x="Revealed comparative advantage", y="Product Group", orientation="h",
                title="Top 10 Products by RCA", color_discrete_sequence=["#3B82F6"]
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            fig = stylize(fig)
            chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-growth-products":
        df_growth_calc = df_chart.copy()
        df_products_growth_filtered = df_growth_calc[df_growth_calc["Product Group"] != "All Products"]
        if year_value is not None:
            top_products_filtered = df_products_growth_filtered.groupby("Product Group")["Export (US$ Thousand)"].sum().sort_values(ascending=False).head(10).reset_index()
            if top_products_filtered.empty:
                chart = html.Div("No data available for selected year", className="no-data-message")
            else:
                fig = px.bar(
                    top_products_filtered, x="Export (US$ Thousand)", y="Product Group", orientation="h",
                    title=f"Top 10 Products by Export Value ({year_value})",
                    color_discrete_sequence=["#3B82F6"]
                )
                fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                fig = stylize(fig)
                chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
        else:
            df_products_growth_filtered = df_products_growth_filtered.sort_values(["Product Group", "Partner Name", "Year"])
            df_products_growth_filtered["Export Growth Share"] = df_products_growth_filtered.groupby(["Product Group", "Partner Name"])["Export (US$ Thousand)"].pct_change() * 100
            df_products_growth_filtered = df_products_growth_filtered.dropna(subset=["Export Product Share (%)", "Export Growth Share"])
            df_products_growth_filtered = df_products_growth_filtered[df_products_growth_filtered["Export Growth Share"] > 0]
            top_growth_products_filtered = df_products_growth_filtered.groupby("Product Group", as_index=False)["Export Growth Share"].mean().sort_values("Export Growth Share", ascending=True).tail(10)
            if top_growth_products_filtered.empty:
                chart = html.Div("No growth data available", className="no-data-message")
            else:
                fig = px.bar(
                    top_growth_products_filtered, x="Export Growth Share", y="Product Group", orientation="h",
                    title="Top 10 Products by Growth Rate", color_discrete_sequence=["#3B82F6"]
                )
                fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                fig = stylize(fig)
                chart = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "100%"})
    
    elif button_id == "nav-predict-exports":
        chart = html.Div(
            className="prediction-container",
            children=[
                html.H2("🔮 Export Prediction Tool", className="prediction-title"),
                html.P("Forecast Rwanda's export potential using economic indicators and machine learning", className="prediction-subtitle"),
                
                html.Div(
                    className="prediction-form",
                    children=[
                        html.Div(
                            className="form-section",
                            children=[
                                html.H4("📅 Time & Product Selection", className="form-section-title"),
                                html.Label("Year for Prediction:", className="form-label"),
                                dcc.Dropdown(
                                    id="year-dropdown",
                                    options=[{"label": str(y), "value": y} for y in range(2022, 2027)],
                                    value=2025,
                                    clearable=False,
                                    className="form-dropdown"
                                ),
                                html.Label("Product Category:", className="form-label"),
                                dcc.Dropdown(
                                    id="product-dropdown",
                                    options=[{"label": x, "value": x} for x in sorted(df["Product Group"].unique())],
                                    value=sorted(df["Product Group"].unique())[0],
                                    className="form-dropdown"
                                ),
                            ],
                        ),
                        html.Div(
                            className="form-section",
                            children=[
                                html.H4("📊 Economic Indicators", className="form-section-title"),
                                html.Label("Global Market Growth (%)", className="form-label"),
                                dcc.Slider(
                                    id="world-growth-input",
                                    min=0, max=6, step=0.1, value=3.5,
                                    marks={i: f"{i}%" for i in range(0, 7)},
                                    className="form-slider"
                                ),
                                html.Small("Expected growth of global market demand", className="form-hint"),
                                
                                html.Label("Rwanda Economy Growth (%)", className="form-label"),
                                dcc.Slider(
                                    id="country-growth-input",
                                    min=5, max=10, step=0.1, value=8.0,
                                    marks={i: f"{i}%" for i in range(5, 11)},
                                    className="form-slider"
                                ),
                                html.Small("Annual GDP growth rate of Rwanda", className="form-hint"),
                            ],
                        ),
                        html.Div(
                            className="form-section",
                            children=[
                                html.H4("💰 GDP Information", className="form-section-title"),
                                html.Label("Nominal GDP (USD)", className="form-label"),
                                dcc.Input(
                                    id="gdp-current-input",
                                    type="number",
                                    value=14_300_000_000,
                                    step=100_000_000,
                                    className="form-input"
                                ),
                                html.Small("Example: 2023 ($14.1B), 2024 ($14.3B est.)", className="form-hint"),
                                
                                html.Label("GDP Growth Rate (%)", className="form-label"),
                                dcc.Slider(
                                    id="gdp-growth-input",
                                    min=5, max=10, step=0.1, value=7.0,
                                    marks={i: f"{i}%" for i in range(5, 11)},
                                    className="form-slider"
                                ),
                            ],
                        ),
                    ],
                ),
                
                html.Button(
                    "🔮 Predict Export Value",
                    id="predict-button",
                    n_clicks=0,
                    className="predict-button"
                ),
                
                html.Div(id="prediction-output", className="prediction-output"),
            ],
        )
    
    elif button_id == "nav-recommendation":
        chart = html.Div(
            className="recommendations-container",
            children=[
                html.Div(
                    className="recommendations-header",
                    children=[
                        html.H2("💡 Strategic Recommendations for SMEs in Rwanda", className="recommendations-title"),
                        html.P(
                            "Export growth is possible when SMEs strategically explore markets with institutions like RDB and NAEB, "
                            "beginning regionally under AfCFTA and EAC. By meeting international quality standards and embedding Rwanda's "
                            "cultural identity in branding, trust and uniqueness are secured. Leveraging export services, clusters, and "
                            "digital platforms opens scalable opportunities. With consistency in quality, branding, and partnerships, "
                            "Rwanda's SMEs can confidently transform into global players.",
                            className="recommendations-intro"
                        ),
                    ],
                ),
                
                html.Div(
                    className="pillars-grid",
                    children=[
                        # Pillar 1: Market Expansion
                        html.Div(
                            className="pillar-card pillar-blue",
                            children=[
                                html.Div(className="pillar-icon", children="🌍"),
                                html.H3("Market Expansion", className="pillar-title"),
                                html.Div(
                                    className="pillar-content",
                                    children=[
                                        html.P("Strategic Regional Focus", className="pillar-subtitle"),
                                        html.Ul(
                                            className="pillar-list",
                                            children=[
                                                html.Li("Begin with AfCFTA and EAC regional markets"),
                                                html.Li("Partner with RDB and NAEB for market intelligence"),
                                                html.Li("Leverage preferential trade agreements"),
                                                html.Li("Explore Sub-Saharan Africa opportunities"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        
                        # Pillar 2: Quality & Identity
                        html.Div(
                            className="pillar-card pillar-green",
                            children=[
                                html.Div(className="pillar-icon", children="⭐"),
                                html.H3("Quality & Identity", className="pillar-title"),
                                html.Div(
                                    className="pillar-content",
                                    children=[
                                        html.P("Standards & Cultural Branding", className="pillar-subtitle"),
                                        html.Ul(
                                            className="pillar-list",
                                            children=[
                                                html.Li("Meet international quality standards"),
                                                html.Li("Embed Rwanda's cultural identity in branding"),
                                                html.Li("Build trust through consistent quality"),
                                                html.Li("Create unique market positioning"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        
                        # Pillar 3: Digital Enablement
                        html.Div(
                            className="pillar-card pillar-purple",
                            children=[
                                html.Div(className="pillar-icon", children="🚀"),
                                html.H3("Digital Enablement", className="pillar-title"),
                                html.Div(
                                    className="pillar-content",
                                    children=[
                                        html.P("Technology & Scale", className="pillar-subtitle"),
                                        html.Ul(
                                            className="pillar-list",
                                            children=[
                                                html.Li("Leverage digital export platforms"),
                                                html.Li("Join industry clusters for collaboration"),
                                                html.Li("Access export services and support"),
                                                html.Li("Scale operations through technology"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        
                        # Pillar 4: Strategic Partnerships
                        html.Div(
                            className="pillar-card pillar-orange",
                            children=[
                                html.Div(className="pillar-icon", children="🤝"),
                                html.H3("Strategic Partnerships", className="pillar-title"),
                                html.Div(
                                    className="pillar-content",
                                    children=[
                                        html.P("Collaboration & Growth", className="pillar-subtitle"),
                                        html.Ul(
                                            className="pillar-list",
                                            children=[
                                                html.Li("Collaborate with RDB and NAEB"),
                                                html.Li("Form strategic alliances with exporters"),
                                                html.Li("Maintain consistency in partnerships"),
                                                html.Li("Build long-term relationship networks"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                
                html.Div(
                    className="action-section",
                    children=[
                        html.H3("🎯 Key Takeaways", className="action-title"),
                        html.Div(
                            className="takeaways-grid",
                            children=[
                                html.Div(
                                    className="takeaway-item",
                                    children=[
                                        html.Div(className="takeaway-number", children="1"),
                                        html.P("Start Regional, Think Global - Begin with AfCFTA/EAC markets"),
                                    ],
                                ),
                                html.Div(
                                    className="takeaway-item",
                                    children=[
                                        html.Div(className="takeaway-number", children="2"),
                                        html.P("Quality + Culture = Competitive Advantage"),
                                    ],
                                ),
                                html.Div(
                                    className="takeaway-item",
                                    children=[
                                        html.Div(className="takeaway-number", children="3"),
                                        html.P("Digital Platforms Enable Scale and Reach"),
                                    ],
                                ),
                                html.Div(
                                    className="takeaway-item",
                                    children=[
                                        html.Div(className="takeaway-number", children="4"),
                                        html.P("Strategic Partnerships Accelerate Growth"),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
    
    else:  # nav-export-trends (default)
        fig = px.line(
            exports_by_year, x="Year", y="Export (US$ Thousand)", markers=True,
            title="Rwanda Export Trend Over Time"
        )
        fig.update_traces(
            line=dict(color="#3B82F6", width=3),
            marker=dict(size=8, color="#60A5FA", line=dict(color="#3B82F6", width=2))
        )
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

    if None in [year, product, gdp_current, gdp_growth, world_growth, country_growth]:
        return html.Div(
            className="prediction-result prediction-error",
            children=[
                html.Div(className="result-icon", children="⚠"),
                html.P("Please fill in all fields before predicting")
            ]
        )
    
    X_new = pd.DataFrame([{
        "Year": year,
        "Product Group": product,
        "World Growth (%)": world_growth,
        "Country Growth (%)": country_growth,
        "GDP_Current_USD": gdp_current,
        "GDP_Growth_Percent": gdp_growth,
    }])
    
    prediction = model.predict(X_new)[0]
    
    return html.Div(
        className="prediction-result prediction-success",
        children=[
            html.Div(className="result-icon", children="✅"),
            html.H3("Predicted Export Value", className="result-label"),
            html.Div(className="result-value", children=f"${prediction:,.0f}K"),
            html.P(f"for {product} in {year}", className="result-detail"),
        ]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  # Use Render's assigned port
    app.run(debug=False, host='0.0.0.0', port=port) # Or simply app.run(debug=False)