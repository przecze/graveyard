import contextlib

try:
    import streamlit as st
except ImportError:
    st = None  # only needed when IS_MAIN is False (Streamlit mode)

IS_MAIN = False
# ── UI: standalone (stdout/defaults) vs Streamlit ───────────────────────────────

class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False

def write(x):
    if IS_MAIN: print(x); return
    st.write(x)

def subheader(text):
    if IS_MAIN: print(text); return
    st.subheader(text)

def sidebar_radio(label, options, format_func=None, help=""):
    if IS_MAIN: return options[0]
    return st.sidebar.radio(label, options=options, format_func=format_func, help=help)

def sidebar_slider(label, *, min_value, max_value, value, step=None, help=""):
    if IS_MAIN: return value
    return st.sidebar.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, help=help)

def sidebar_select_slider(label, *, options, value, format_func=None, help=""):
    if IS_MAIN: return value
    return st.sidebar.select_slider(label, options=options, value=value, format_func=format_func, help=help)

def spinner(msg):
    if IS_MAIN: return contextlib.nullcontext()
    return st.spinner(msg)

def warning(msg):
    if IS_MAIN: print(f"[warning] {msg}"); return
    st.warning(msg)

def columns(n):
    if IS_MAIN: return tuple(_Ctx() for _ in range(n))
    return st.columns(n)

def toggle(label, *, value=False):
    if IS_MAIN: return value
    return st.toggle(label, value=value)

def tabs(labels):
    if IS_MAIN: return tuple(_Ctx() for _ in range(len(labels)))
    return st.tabs(labels)

def plotly_chart(fig, width="stretch"):
    if IS_MAIN: return
    st.plotly_chart(fig, width=width)

def expander(title, *, expanded=False):
    if IS_MAIN: return contextlib.nullcontext()
    return st.expander(title, expanded=expanded)

def metric(label, value):
    if IS_MAIN: print(f"{label}: {value}"); return
    st.metric(label, value)

def dataframe(df, width="stretch"):
    if IS_MAIN:
        if df:
            for row in df:
                print(row)
        return
    st.dataframe(df, width=width)
