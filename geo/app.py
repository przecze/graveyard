import streamlit as st

import death_model
import geo
import view_port

page = st.sidebar.radio("Page", ["Death Rate Model", "Geometry", "Port"])

if page == "Death Rate Model":
    death_model.render()

elif page == "Geometry":
    geo.render()

else:
    view_port.render()
