# app.py - REVISED MAIN BLOCK FOR STABILITY

# ... (Previous code up to line 477 remains unchanged) ...

# --- 4. MAIN ENTRY POINT ---

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="🏀 DK Lineup Optimizer")
    
    # Sidebar
    with st.sidebar:
        st.title("🏀 DK Lineup Optimizer")
        st.caption("Maximize Projection based on Template")
        
        contest_type = st.selectbox("Contest Strategy", ['GPP (Single Entry)', 'GPP (Large Field)', 'CASH'])
        
        c_map = {'GPP (Single Entry)': 'SE', 'GPP (Large Field)': 'LARGE_GPP', 'CASH': 'CASH'}
        contest_code = c_map[contest_type]
        
        st.divider()
        st.subheader("Paste Data (CSV Format)")
        
        pasted_csv_data = st.text_area(
            "Paste your player pool data here (including headers)", 
            height=200,
            key="csv_paste_area",
            help="Copy the entire table, including headers, and paste it here."
        )
        
        load_button = st.button("Load Pasted Data", use_container_width=True)

    # --- DATA LOADING LOGIC (Simplified and made robust) ---
    if 'slate_df' not in st.session_state:
        st.session_state['slate_df'] = pd.DataFrame()
        
    if load_button and pasted_csv_data.strip():
        # Load the data and store the processed DataFrame in session state
        st.session_state['slate_df'] = load_and_preprocess_data(pasted_csv_data)
    
    # Use the DataFrame from session state for the rest of the app
    slate_df = st.session_state['slate_df'] 
    # --- END DATA LOADING LOGIC ---
        
    # Build Template
    template = build_template_from_params(
        contest_type=contest_code, 
        field_size=10000, 
        pct_to_first=30.0,
        roster_size=DEFAULT_ROSTER_SIZE,
        salary_cap=DEFAULT_SALARY_CAP,
        min_games=MIN_GAMES_REQUIRED
    )

    # Tabs
    t1, t2 = st.tabs(["✨ Optimal Lineup Builder", "📝 Contest Analyzer"])
    
    with t1:
        tab_lineup_builder(slate_df, template)
    with t2:
        tab_contest_analyzer(slate_df, template)
