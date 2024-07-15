import pandas as pd
import streamlit as st
from energy_module.energyp import EnergyPrices
from energy_module.energyp import Helper
from energy_module.energyp import PV
from schedule_plot import get_schedule_plot
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
#import locale
#locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

st.set_page_config(layout='wide')

region_col_name = "Germany/Luxembourg [€/MWh] Original resolutions"

def main():
    # Sidebar
    st.sidebar.title("User input data")
    generate = st.sidebar.button(label="Generate random schedule",use_container_width=True)
    pv_installed_capacity = st.sidebar.slider("PV installed capacity in kW", 0, 200, 50)/1000

    current_hour = datetime.now().hour
    if current_hour > 15:
        date_pick = st.sidebar.date_input(label="Select date",value=datetime.now()+timedelta(days=0))
    else:
        date_pick = st.sidebar.date_input(label="Select date",value=datetime.now()+timedelta(days=-1)) # timedelta is kept for debuging purposes

    if generate:
        scheduler = Helper().get_scheduler(load=False)
    else:
        scheduler = Helper().get_scheduler(load=True)

    sch_df = scheduler.get_schedule()

    # Data retrieval
    # power data
    pw_df = scheduler.get_power()
    print(pw_df)

    # Energy prices data
    prices_df = EnergyPrices().get_prices(date_pick)

    # Generation Capacity data
    caps_df = PV().get_caps(date_pick)

    # Metrics
    performance, total_cost, total_energy, total_production, total_number_of_jobs = EnergyPrices().metrics(prices_df,scheduler)
    print("performance {}, total_cost {}, total_energy {}, total_production {}, total_number_of_jobs {}.".format(performance, total_cost, total_energy, total_production, total_number_of_jobs ))

    #Scheduler plot
    dates = pd.date_range(datetime.now().strftime("%Y-%m-%d"), periods=12, freq='2h')
    x_values = [i*60*2 for i in range(12)]
    x_values3 = [prices_df['Start date'][i*2] for i in range(12)]

    prices_profile = EnergyPrices().prices_profile(prices_df)
    fig1, fig11 = get_schedule_plot(prices_profile) # $$

    fig2 = go.Figure()

    for i in range(pw_df.shape[1]):
        if i == pw_df.shape[1]-1:
            name = f"Total power"
            fig2.add_trace(go.Scatter(
                name=name,
                x=pw_df.index.values,
                y=pw_df.iloc[:,i],
                line=dict(color='black', width=4, dash='dot')
            ))
        else:
            name = f"machine {i}"
            fig2.add_trace(go.Scatter(
                name=name,
                x=pw_df.index.values,
                y=pw_df.iloc[:,i],
                fill='tozeroy',
            ))

    fig2.update_xaxes(tickvals=x_values, ticktext=[d.strftime('%H:00') for d in dates])
    fig2.update_layout(
        height=300,
        title="Factory total power",
        yaxis_title="Power [kW]",
        margin=dict(
            l=95,
            r=30,
            b=0,
            t=50,
            pad=4
        ),
        legend=dict(
            orientation='h',
            yanchor="top",
            y=-0.3,
            xanchor="left",
            x=0
        )
    )

    fig3 = px.area(prices_df,x="Start date", y="Germany/Luxembourg [€/MWh] Original resolutions",line_shape='hv',title="Energy prices")
    fig3.update_xaxes(tickvals=x_values3, ticktext=[d.strftime('%H:00') for d in dates])

    # PV data
    pv_power = PV().get_pv_power(pv_installed_capacity, date_pick)
    if pv_power is not None:
        savings_df = PV().get_power_difference(pv_power,scheduler,date_pick)
        fig4 = px.line(savings_df,title="Day-ahead PV generation prediction + energy savings (without energy storage system)")
        #fig4.update_xaxes(tickvals=x_values3, ticktext=[d.strftime('%H:00') for d in dates])
        fig4.update_layout(
            yaxis_title="Energy [kWh]",
            xaxis_title="Time",
            legend=dict(
                orientation='h',
                yanchor="top",
                y=-0.3,
                xanchor="left",
                x=0
            ))

    # View in UI
    with st.container():
        # Your container content here
        #Schedule
        st.subheader("Day-ahead job schedule and energy prices forecast")

        row1 = st.columns(2)
        row2 = st.columns(3)
        # metrics = [
        #     {"label": "Scheduler performance (production/(energy × cost))", "value": locale.format_string("%.4f kg/(kWh × €)",performance)},
        #     {"label": "Total energy cost", "value": locale.format_string("%.2f €",total_cost)},
        #     {"label": "Total energy usage", "value": locale.format_string("%.2f kWh",total_energy)},
        #     {"label": "Total production", "value": locale.format_string("%.0f kg",total_production)},
        #     {"label": "Total jobs completed", "value": f"{total_number_of_jobs} jobs"},
        # ]
        metrics = [
            {"label": "Scheduler performance (production/(energy × cost))", "value": f"{performance:.2} kg/(kWh × €)"},
            {"label": "Total energy cost", "value": f"{total_cost:.5} €"},
            {"label": "Total energy usage", "value": f"{total_energy:.6} kWh"},
            {"label": "Total production", "value": f"{total_production:.7} kg"},
            {"label": "Total jobs completed", "value": f"{total_number_of_jobs} jobs"},
        ]
        for i,col in enumerate((row1 + row2)):
            cont = col.container(height=120)
            cont.metric(metrics[i]["label"],metrics[i]["value"],)

        st.plotly_chart(fig1,use_container_width=True)
        st.plotly_chart(fig11)
        with st.expander("Machine parameters"):
            m_params = pd.DataFrame([[m.id,m.speed,m.energy_usage] for m in scheduler.machines],columns=['Machine','Speed [kg/min]','Energy usage [kWh/min]'])
            m_params.set_index('Machine')
            st.dataframe(m_params,hide_index=True)

        st.plotly_chart(fig2, use_container_width=True)

        # Energy prices Data
        st.plotly_chart(fig3, use_container_width=True)

        # PV Data
        if pv_power is not None:
            st.plotly_chart(fig4, use_container_width=True)

        # PV data
        st.subheader("Dataframes")
        st.write("Energy generation installed capacity in Germany")
        st.dataframe(caps_df)

        # PV data
        #st.dataframe(pv_df)

        data_cols = st.columns([2,5])

        with data_cols[0]:
            st.write("Schedule dataframe")
            st.dataframe(sch_df)
            st.write("Artificially generated")
        with data_cols[1]:
            st.write("Energy prices dataframe.")
            st.dataframe(prices_df)
            st.write("Data source: Bundesnetzagentur | SMARD.de. More info: https://www.smard.de/en/datennutzung")


if __name__ == "__main__":
    main()