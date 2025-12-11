import streamlit as st
import pandas as pd
import plotly.express as px

# === Import Data ===
df = pd.read_csv("OnlineRetail.csv", encoding="latin1", dtype=str, low_memory=False)
df.drop_duplicates(inplace=True)
# Konversi ke numerik
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

# Drop baris rusak
df = df.dropna(subset=['Quantity', 'UnitPrice'])

# Hitung total amount
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceYearMonth'] = df['InvoiceDate'].dt.to_period('M')
df['InvoiceDate_only'] = df['InvoiceDate'].dt.date
df['DayName'] = df['InvoiceDate'].dt.day_name()
df["Hour"] = df["InvoiceDate"].dt.hour
df['InvoiceMonthName'] = df['InvoiceDate'].dt.strftime("%B")

# PAGE CONFIG
st.set_page_config(page_title="A25-CS313", layout="wide")

st.title("Customer Insight Mining: Pendekatan RFM dan Machine Learning untuk Meningkatkan Loyalitas Pelanggan")

# === TAB ===
tab_visualization, tab_rfm, tab_clustering, tab_insight = st.tabs(["VISUALISASI DATA AWAL",
                                                                   "RFM ANALYSIS",
                                                                   "CLUSTERING ANALYSIS",
                                                                   "INTERPRETASI"])

with tab_visualization:
#============ VISUALISASI PEMBELI BERDASARKAN NEGARA (ATLAS WORLD MAP) =============
    st.subheader("ANALISIS PENJUALAN DAN PENDAPATAN BERDASARKAN NEGARA")

    # --- Hitung revenue, transaksi, dll ---
    country_info = (
        df.groupby("Country")
        .agg(
            TotalRevenue=("TotalAmount", "sum"),
            TransactionCount=("InvoiceNo", "nunique")
        )
        .reset_index()
    )

    country_info["Purchased"] = 1  # indikator negara pembeli

    # --- ALL COUNTRY LIST dari Plotly (gapminder) ---
    world = px.data.gapminder()[["country"]].drop_duplicates()
    world.columns = ["Country"]

    # --- Merge: negara pembeli + negara yang tidak beli ---
    world_map = world.merge(country_info, on="Country", how="left")
    world_map["Purchased"] = world_map["Purchased"].fillna(0)

    # --- warna: negara beli = warna atlas, negara lain = abu muda ---
    world_map["ColorValue"] = world_map["Purchased"].astype(int)

    # --- CHOROPLETH ATLAS MAP ---
    fig_atlas = px.choropleth(
        world_map,
        locations="Country",
        locationmode="country names",
        color="ColorValue",
        hover_name="Country",
        hover_data={
            "TotalRevenue": ":,.0f",
            "TransactionCount": ":,",
            "Purchased": False,
            "ColorValue": False
        },
        color_continuous_scale=["#d3d3d3", "#ff9933"],  # abu → oranye atlas
    )

    # --- STYLE SEPERTI ATLAS ---
    fig_atlas.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="white",
        oceancolor="#f8f8f8",
        lakecolor="#f8f8f8",
        projection_type="natural earth"
    )

    fig_atlas.update_layout(
        height=700,
        width=1500,
        coloraxis_showscale=False,   # sembunyikan legend warna
        title="Peta Persebaran Pelanggan Secara Global",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig_atlas, use_container_width=True)



#======== TOTAL PEMASUKAN PER NEGARA ============
    with st.expander("Penjualan Berdasarkan Negara"):
        # Grouping
        country = (
            df.groupby('Country')
            .agg(TotalRevenue=('TotalAmount', 'sum'),
                TransactionCount=('TotalAmount', 'count'),
                TotalQuantity=('Quantity', 'sum'),
                UniqueInvoices=('InvoiceNo', 'nunique'))
            .sort_values('TotalRevenue', ascending=False)
        )

        # Persentase pemasukan
        country['RevenuePercentage'] = (
            country['TotalRevenue'] / country['TotalRevenue'].sum() * 100
        ).round(2)

        # Ambil 5 besar
        top = country.head(5).reset_index()   # <--- PENTING: Country tetap "Country"

        # Palet warna
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D", "#FFBE66",
            "#FFCC80", "#FFD599", "#FFECCC", "#FFF5E6", "#FFE0B2"
        ]

        # Barchart
        fig = px.bar(
            top,
            x="Country",
            y="TotalRevenue",
            text="TotalRevenue",
            color="Country",
            color_discrete_sequence=PALETTE,
            title="5 Negara dengan Penjualan Terbesar"
        )

        # Format Hover + Teks (HASIL PERSENTASE BELUMMM JELAS DAN JELEK MAKANYA DIHAPUS)
        fig.update_traces(
            texttemplate='£%{y:,.0f}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Revenue: £%{y:,.0f}<br>" 
        )

        # Tampilkan chart
        st.plotly_chart(fig, use_container_width=True)

        # Info Negara Terbesar
        top_country = top.iloc[0]["Country"]
        pct = top.iloc[0]["RevenuePercentage"]
        rev = top.iloc[0]["TotalRevenue"]

        st.success(
            f"**Negara dengan Penjualan terbesar: `{top_country}`**\n"
            f"- Total Pemasukan: **£{rev:,.0f}**\n"
            # f"- Share: **{pct:.2f}%**"
        )

    #======== TOTAL PEMASUKAN PER NEGARA (EXCLUDE UK) ============
    with st.expander("Penjualan Berdasarkan Negara (Tanpa UK)"):
        # Filter negara
        df_filtered = df[df['Country'] != 'United Kingdom'].copy()

        # Hitung total amount
        df_filtered['TotalAmount'] = df_filtered['Quantity'] * df_filtered['UnitPrice']

        # Grouping per negara
        country = (
            df_filtered.groupby('Country')
                .agg(
                    TotalRevenue=('TotalAmount', 'sum'),
                    TransactionCount=('TotalAmount', 'count'),
                    TotalQuantity=('Quantity', 'sum'),
                    UniqueInvoices=('InvoiceNo', 'nunique')
                )
                .sort_values('TotalRevenue', ascending=False)
        )

        # Persentase revenue
        country['RevenuePercentage'] = (
            country['TotalRevenue'] / country['TotalRevenue'].sum() * 100
        ).round(2)

        # Ambil 10 teratas
        top = country.head(10).reset_index()

        # Palet warna
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D", "#FFBE66",
            "#FFCC80", "#FFD599", "#FFECCC", "#FFF5E6", "#FFE0B2"
        ]

        # Barchart
        fig = px.bar(
            top,
            x="Country",
            y="TotalRevenue",
            text="TotalRevenue",
            color="Country",
            color_discrete_sequence=PALETTE,
            title="Top 10 Negara (Tanpa UK)"
        )

        fig.update_traces(
            texttemplate='£%{y:,.0f}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Revenue: £%{y:,.0f}<br>"
        )

        fig.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig, use_container_width=True)

        # Insight negara teratas
        top_country = top.iloc[0]["Country"]
        pct = top.iloc[0]["RevenuePercentage"]
        rev = top.iloc[0]["TotalRevenue"]

        st.success(
            f"**Negara dengan penjualan terbesar (tanpa UK): `{top_country}`**\n"
            f"- Total Pemasukan: **£{rev:,.0f}**"
        )

#======== NEGARA DENGAN PENJUALAN PALING SEDIKIT ============
    with st.expander("Negara dengan Penjualan Paling Sedikit"):
        # Ambil 10 negara terbawah berdasarkan TotalRevenue
        bottom = (
            country
            .sort_values('TotalRevenue', ascending=True)
            .head(5)
            .reset_index()
        )

        # Barchart
        fig_bottom = px.bar(
            bottom,
            x="Country",
            y="TotalRevenue",
            text="TotalRevenue",
            color="Country",
            color_discrete_sequence=PALETTE,  # boleh pakai palet warna yang sama
            title="5 Negara dengan Penjualan Paling Sedikit"
        )

        fig_bottom.update_traces(
            texttemplate='£%{y:,.0f}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Revenue: £%{y:,.0f}<br>"
        )

        st.plotly_chart(fig_bottom, use_container_width=True)

        # Insight negara dengan penjualan paling sedikit
        low_country = bottom.iloc[0]["Country"]
        low_rev = bottom.iloc[0]["TotalRevenue"]
        low_pct = bottom.iloc[0]["RevenuePercentage"]

        st.info(
            f"**Negara dengan penjualan paling sedikit: `{low_country}`**\n"
            f"- Total pemasukan: **£{low_rev:,.0f}**\n"
            # f"- Share: **{low_pct:.2f}%**"
        )

# ==================== Tren Pendapatan Bulanan =======================
    with st.expander("Tren Pendapatan Bulanan Tahun 2011-2012"):
        monthly = (
            df.groupby('InvoiceYearMonth')
            .agg(
                TotalAmount=('TotalAmount', 'sum'),
                Orders=('InvoiceNo', 'nunique'),
                Active_Customers=('CustomerID', 'nunique')
            )
            .reset_index()
        )

        # Konversi ke string agar tampil rapi di plot
        monthly['InvoiceYearMonth'] = monthly['InvoiceYearMonth'].astype(str)

        # Hitung Average Order Value (AOV)
        monthly['AOV'] = monthly['TotalAmount'] / monthly['Orders']

        # Warna garis
        LINE_COLOR = "#FF8C00"

        # Plotly line chart
        fig_monthly = px.line(
            monthly,
            x="InvoiceYearMonth",
            y="TotalAmount",
            markers=True,
            title="Tren Pendapatan Bulanan Tahun 2011-2012",
        )

        fig_monthly.update_traces(
            line=dict(width=3, color=LINE_COLOR),
            marker=dict(size=8),
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Total Amount: £%{y:,.0f}<br>" +
                "Orders: %{customdata[0]:,}<br>" +
                "Active Customers: %{customdata[1]:,}<br>" +
                "AOV: £%{customdata[2]:,.2f}<extra></extra>",
            customdata=monthly[['Orders', 'Active_Customers', 'AOV']].values
        )

        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Amount (£)",
            xaxis_tickangle=-45,
            plot_bgcolor="white",
            height=450,
        )

        # Tampilkan chart di Streamlit
        st.plotly_chart(fig_monthly, use_container_width=True)

        # ============ Insight otomatis ============
        best_month = monthly.loc[monthly['TotalAmount'].idxmax()]
        worst_month = monthly.loc[monthly['TotalAmount'].idxmin()]

        insight = f"""
        - **Bulan dengan transaksi (Total Amount) tertinggi:** `{best_month['InvoiceYearMonth']}`  
        Total: **£{best_month['TotalAmount']:,.0f}**

        - **Bulan dengan transaksi terendah:** `{worst_month['InvoiceYearMonth']}`  
        Total: **£{worst_month['TotalAmount']:,.0f}**

        - **Rata-rata AOV keseluruhan:** £{monthly['AOV'].mean():,.2f}
        """

        st.info(insight)

#======== MONTHLY TREND BY COUNTRY ============
    with st.expander("Tren Pendapatan Bulanan Berdasarkan Negara"):
        
        # Dropdown negara
        selected_country = st.selectbox(
            "Pilih Negara:",
            sorted(df['Country'].unique()),
            key="selected_country_monthly"
        )

        # Filter data untuk negara terpilih
        df_country = df[df['Country'] == selected_country].copy()

        # Pastikan TotalAmount sudah ada (jaga-jaga)
        df_country['TotalAmount'] = df_country['Quantity'] * df_country['UnitPrice']

        # Aggregasi bulanan
        monthly_cty = (
            df_country.groupby('InvoiceYearMonth')
            .agg(
                TotalAmount=('TotalAmount', 'sum'),
                Orders=('InvoiceNo', 'nunique'),
                Active_Customers=('CustomerID', 'nunique')
            )
            .reset_index()
            .sort_values('InvoiceYearMonth')
        )

        # Ubah ke string agar tampil rapi
        monthly_cty['InvoiceYearMonth'] = monthly_cty['InvoiceYearMonth'].astype(str)

        # Hitung AOV
        monthly_cty['AOV'] = monthly_cty['TotalAmount'] / monthly_cty['Orders']

        # Plot
        fig_cty = px.line(
            monthly_cty,
            x="InvoiceYearMonth",
            y="TotalAmount",
            markers=True,
            title=f"Tren Pendapatan Bulanan – {selected_country}",
        )

        fig_cty.update_traces(
            line=dict(width=3, color="#FF8C00"),
            marker=dict(size=8),
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Total Amount: £%{y:,.0f}<br>" +
                "Orders: %{customdata[0]:,}<br>" +
                "Active Customers: %{customdata[1]:,}<br>" +
                "AOV: £%{customdata[2]:,.2f}<extra></extra>",
            customdata=monthly_cty[['Orders', 'Active_Customers', 'AOV']].values
        )

        fig_cty.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Amount (£)",
            xaxis_tickangle=-45,
            plot_bgcolor="white",
            height=450,
        )

        st.plotly_chart(fig_cty, use_container_width=True)

        #============= INSIGHT OTOMATIS =============
        if len(monthly_cty) > 0:
            best_m = monthly_cty.loc[monthly_cty['TotalAmount'].idxmax()]
            worst_m = monthly_cty.loc[monthly_cty['TotalAmount'].idxmin()]
            
            insight_cty = f"""
            - **Bulan dengan pendapatan tertinggi:** `{best_m['InvoiceYearMonth']}`  
            Total: **£{best_m['TotalAmount']:,.0f}**

            - **Bulan dengan pendapatan terendah:** `{worst_m['InvoiceYearMonth']}`  
            Total: **£{worst_m['TotalAmount']:,.0f}**

            - **Rata-rata AOV negara `{selected_country}`:** £{monthly_cty['AOV'].mean():,.2f}
            """

            st.info(insight_cty)
        else:
            st.warning("Tidak ada data untuk negara ini.")


#======== PENJUALAN PRODUK BERDASARKAN REVENUE ============
    st.subheader("ANALISIS PENJUALAN DAN PENDAPATAN BERDASARKAN PRODUK")
    with st.expander("Penjualan Produk Berdasarkan Revenue"):
    # --- Agregasi revenue per produk ---
        product = (
            df.groupby('Description')
            .agg(
                TotalRevenue=('TotalAmount', 'sum'),
                UniqueInvoices=('InvoiceNo', 'nunique'),
                AvgPrice=('UnitPrice', 'mean')
            )
            .reset_index()
        )

        # Sort berdasarkan revenue terbesar
        product = product.sort_values('TotalRevenue', ascending=False)

        # Ambil top 10
        top_prod = product.head(10)

        # Warna palet
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D", "#FFBE66",
            "#FFCC80", "#FFD599", "#FFECCC", "#FFF5E6", "#FFE0B2"
        ]

        # --- Barchart Total Revenue ---
        fig_prod = px.bar(
            top_prod,
            x="Description",
            y="TotalRevenue",
            text="TotalRevenue",
            color="Description",
            color_discrete_sequence=PALETTE,
            title="Top 10 Produk dengan Revenue Tertinggi"
        )

        fig_prod.update_traces(
            texttemplate='£%{y:,.0f}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Total Revenue: £%{y:,.0f}<br>" +
                "Avg Price: £%{customdata[0]:.2f}<extra></extra>",
            customdata=top_prod[['AvgPrice']].values
        )

        fig_prod.update_layout(
            xaxis_title="Product",
            yaxis_title="Total Revenue (£)",
            xaxis_tickangle=-45,
            showlegend=False
        )

        st.plotly_chart(fig_prod, use_container_width=True)

        # --- Insight ---
        top_name = top_prod.iloc[0]['Description']
        top_rev = top_prod.iloc[0]['TotalRevenue']

        summary = f"""
        - **Produk dengan Revenue Tertinggi:** `{top_name}`
        - **Total Revenue:** £{top_rev:,.0f}
        """

        st.info(summary)

#======== PENJUALAN PRODUK BERDASARKAN QUANTITY ============
    with st.expander("Penjualan Produk Berdasarkan Jumlah Produk Terjual"):
        # --- Hitung total quantity per produk ---
        product_qty = (
            df.groupby('Description')['Quantity']
            .sum()
            .reset_index()
            .sort_values('Quantity', ascending=False)
        )

        # Ambil top 10
        top_qty = product_qty.head(10)

        # Warna palet
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D", "#FFBE66",
            "#FFCC80", "#FFD599", "#FFECCC", "#FFF5E6", "#FFE0B2"
        ]

        # --- Barchart Quantity Terjual ---
        fig_qty = px.bar(
            top_qty,
            x="Description",
            y="Quantity",
            text="Quantity",
            color="Description",
            color_discrete_sequence=PALETTE,
            title="Top 10 Produk Berdasarkan Jumlah Quantity Terjual"
        )

        fig_qty.update_traces(
            texttemplate='%{y:,}',
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Quantity Terjual: %{y:,}<extra></extra>"
        )

        fig_qty.update_layout(
            xaxis_title="Product",
            yaxis_title="Quantity Sold",
            xaxis_tickangle=-45,
            showlegend=False
        )

        st.plotly_chart(fig_qty, use_container_width=True)

        # --- Insight ---
        top_name = top_qty.iloc[0]['Description']
        top_q = top_qty.iloc[0]['Quantity']

        summary = f"""
        - **Produk dengan Quantity Terjual Terbanyak:** `{top_name}`
        - **Total Quantity Terjual:** {top_q:,}
        """

        st.info(summary)

#======== SCATTER PLOT: REVENUE vs QUANTITY (ALL PRODUCTS) ============
    with st.expander("Persebaran Penjualan Produk Berdasarkan Pendapatan dan Jumlah Produk Terjual"):
        # --- Buat agregasi revenue & quantity per produk ---
        product_scatter = (
            df.groupby('Description')
            .agg(
                TotalRevenue=('TotalAmount', 'sum'),
                TotalQuantity=('Quantity', 'sum'),
                AvgPrice=('UnitPrice', 'mean')
            )
            .reset_index()
        )
        
        product_scatter = product_scatter[product_scatter["TotalRevenue"] > 0]

        # --- Scatter Plot ---
        fig_scatter = px.scatter(
            product_scatter,
            x="TotalQuantity",
            y="TotalRevenue",
            hover_name="Description",
            hover_data={
                "TotalQuantity": True,
                "TotalRevenue": ":,.0f",
                "AvgPrice": ":,.2f"
            },
            title="Scatter Plot: Total Revenue vs Quantity per Product",
        )

        fig_scatter.update_layout(
            xaxis_title="Total Quantity Sold",
            yaxis_title="Total Revenue (£)",
            height=600,
            plot_bgcolor="white"
        )

        fig_scatter.update_traces(
            marker=dict(opacity=0.7, line=dict(width=1, color="black"))
        )

        # --- Tampilkan chart ---
        st.plotly_chart(fig_scatter, use_container_width=True)

#======== ANALISIS AKTIVITAS PELANGGAN ============
    st.subheader("ANALISIS AKTIVITAS PELANGGAN")

#======== ANALISIS AKTIVITAS PELANGGAN PER HARI ============   
    with st.expander("Keaktifan Pelanggan Berdasarkan Hari"):
        # Urutan hari (biar tidak acak)
        order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Grouping count transaksi per hari
        day_sales = (
            df.groupby('DayName')
            .agg(TransactionCount=('InvoiceNo', 'nunique'))
            .reindex(order_days)   # pastikan urut
            .reset_index()
        )
        # Palet warna
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D",
            "#FFBE66", "#FFCC80", "#FFD599"
        ]

        # Barchart
        fig = px.bar(
            day_sales,
            x="DayName",
            y="TransactionCount",
            text="TransactionCount",
            color="DayName",
            color_discrete_sequence=PALETTE,
            title="Jumlah Transaksi Pelanggan Berdasarkan Hari"
        )

        # Hover + Label
        fig.update_traces(
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Transaksi: %{y:,}<extra></extra>"
        )

        # Layout
        fig.update_layout(
            xaxis_title="Hari",
            yaxis_title="Jumlah Transaksi",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Insight
        top_day = day_sales.loc[day_sales['TransactionCount'].idxmax()]
        st.success(
            f"**Hari dengan transaksi terbanyak: `{top_day['DayName']}`**\n"
            f"- Jumlah Transaksi: **{top_day['TransactionCount']:,}**"
        )
#======== ANALISIS AKTIVITAS PER JAM BERDASARKAN HARI ============
    with st.expander("Keaktifan Pelanggan Berdasarkan Jam & Hari"):

        # Dropdown hari
        order_days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

        selected_day = st.selectbox(
            "Pilih Hari:",
            order_days,
            key="selected_day_hour"
        )

        # Filter sesuai hari yang dipilih
        df_day = df[df['DayName'] == selected_day]

        # Grouping per jam
        hourly_sales = (
            df_day.groupby("Hour")
            .agg(TransactionCount=('InvoiceNo', 'nunique'))
            .reset_index()
        )

        # Pastikan jam 0–23 muncul semua
        import pandas as pd
        all_hours = pd.DataFrame({"Hour": range(24)})
        hourly_sales = all_hours.merge(hourly_sales, on="Hour", how="left").fillna(0)

        # Plot line chart
        fig_hour = px.line(
            hourly_sales,
            x="Hour",
            y="TransactionCount",
            markers=True,
            title=f"Trend Jumlah Transaksi per Jam – {selected_day}"
        )

        fig_hour.update_traces(
            line=dict(width=3, color="#FF8C00"),
            marker=dict(size=8),
            hovertemplate=
                "<b>Jam %{x}:00</b><br>" +
                "Transaksi: %{y:,}<extra></extra>"
        )

        # Pastikan X-axis menunjukkan semua jam
        fig_hour.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1  # tampilkan 0–23 tanpa kelipatan
            ),
            xaxis_title="Jam",
            yaxis_title="Jumlah Transaksi",
            plot_bgcolor="white",
            height=450
        )

        st.plotly_chart(fig_hour, use_container_width=True)

        # Insight
        top_hour = hourly_sales.loc[hourly_sales['TransactionCount'].idxmax()]
        st.success(
            f"**Jam paling aktif pada hari {selected_day}: pukul {int(top_hour['Hour'])}:00**\n"
            f"- Total transaksi: **{int(top_hour['TransactionCount']):,}**"
        )
#======== ANALISIS AKTIVITAS PELANGGAN PER BULAN ============   
    with st.expander("Keaktifan Pelanggan Berdasarkan Bulan"):

        # Urutan bulan agar rapi
        order_months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]

        # Grouping count transaksi per bulan
        month_sales = (
            df.groupby('InvoiceMonthName')
            .agg(TransactionCount=('InvoiceNo', 'nunique'))
            .reindex(order_months)     # agar urut
            .reset_index()
        )

        # Warna
        PALETTE = [
            "#FF8C00", "#FFA733", "#FFA726", "#FFB74D",
            "#FFBE66", "#FFCC80", "#FFD599", "#FFE0B2",
            "#FFECCC", "#FFF5E6", "#FFE5CC", "#FFD8B2"
        ]

        # Barchart
        fig_month = px.bar(
            month_sales,
            x="InvoiceMonthName",
            y="TransactionCount",
            text="TransactionCount",
            color="InvoiceMonthName",
            color_discrete_sequence=PALETTE,
            title="Jumlah Transaksi Pelanggan Berdasarkan Bulan"
        )

        # Hover + Label
        fig_month.update_traces(
            textposition='outside',
            hovertemplate=
                "<b>%{x}</b><br>" +
                "Transaksi: %{y:,}<extra></extra>"
        )

        # Layout
        fig_month.update_layout(
            xaxis_title="Bulan",
            yaxis_title="Jumlah Transaksi",
            showlegend=False,
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig_month, use_container_width=True)

        # Insight
        top_month = month_sales.loc[month_sales['TransactionCount'].idxmax()]
        st.success(
            f"**Bulan dengan transaksi terbanyak: `{top_month['InvoiceMonthName']}`**\n"
            f"- Jumlah Transaksi: **{top_month['TransactionCount']:,}**"
        )





