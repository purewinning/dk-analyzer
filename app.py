# app.py (Partial - showing the core changes)

# ... (Imports and other configuration remain the same) ...

from builder import (
    build_template_from_params, 
    # NOTE: optimize_single_lineup is now replaced by generate_top_n_lineups
    generate_top_n_lineups, 
    ownership_bucket,
    PUNT_THR, CHALK_THR, MEGA_CHALK_THR,
    DEFAULT_SALARY_CAP, DEFAULT_ROSTER_SIZE
) 

# ... (load_and_preprocess_data and display_optimized_lineup helpers are modified) ...

# Global session state update for multiple lineups
if 'optimal_lineups_results' not in st.session_state:
    st.session_state['optimal_lineups_results'] = {'lineups': [], 'ran': False}
# ... (edited_df state remains the same) ...

# Function to display a list of lineups
def display_multiple_lineups(slate_df, lineup_list):
    """Function to display the top N optimized lineups."""
    
    if not lineup_list:
        st.error("❌ No valid lineups could be found that meet all constraints.")
        st.warning("Try loosening your constraints or reducing the number of lineups requested.")
        return
        
    st.success(f"Found {len(lineup_list)} optimal lineups.")
    
    all_lineup_dfs = []

    for i, lineup_data in enumerate(lineup_list):
        selected_lineup_ids = lineup_data['player_ids']
        total_points = lineup_data['proj_score']

        lineup_df = slate_df[slate_df['player_id'].isin(selected_lineup_ids)].copy()
        
        # 1. Assign Roster Position (HACK for display)
        ROSTER_ORDER = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        lineup_df = lineup_df.head(8).assign(roster_position=ROSTER_ORDER)

        # 2. Sort the DataFrame by the custom position order
        position_type = pd.CategoricalDtype(ROSTER_ORDER, ordered=True)
        lineup_df['roster_position'] = lineup_df['roster_position'].astype(position_type)
        lineup_df.sort_values(by='roster_position', inplace=True)
        
        # Calculate metrics for summary
        total_salary = lineup_df['salary'].sum()
        games_used = lineup_df['GameID'].nunique()
        
        lineup_df['Lineup'] = i + 1
        lineup_df['Total Proj'] = total_points
        lineup_df['Salary'] = total_salary
        
        all_lineup_dfs.append(lineup_df)

    # Combine all lineups into a single DataFrame for multi-lineup display
    final_df = pd.concat(all_lineup_dfs)
    
    # Define display columns and formatting
    display_cols = ['Lineup', 'Total Proj', 'Salary', 'roster_position', 'Name', 'positions', 'salary', 'proj', 'own_proj']
    final_df = final_df[display_cols].reset_index(drop=True)
    final_df.rename(columns={'roster_position': 'SLOT', 'proj': 'Proj Pts'}, inplace=True)

    # Group the display for better readability
    
    st.subheader("Top Lineups Summary")
    summary_df = final_df[['Lineup', 'Total Proj', 'Salary']].drop_duplicates().set_index('Lineup')
    st.dataframe(
        summary_df.style.format({"Total Proj": "{:.2f}", "Salary": "${:,}"}), 
        use_container_width=True
    )
    
    st.subheader("Lineup Details")
    # Using a list of DataFrames is better for display than one giant merged table
    for i, df in enumerate(all_lineup_dfs):
        st.markdown(f"**Lineup #{i+1}** | Projection: **{df['Total Proj'].iloc[0]:.2f}** | Salary: **${df['Salary'].iloc[0]:,}**")
        st.dataframe(
            df[['roster_position', 'Name', 'positions', 'salary', 'proj', 'own_proj']].style.format({
                "salary": "${:,}", 
                "proj": "{:.1f}", 
                "own_proj": "{:.1f}%"
            }), 
            hide_index=True,
            use_container_width=True
        )
        st.markdown("---")


def tab_lineup_builder(slate_df, template):
    """Render the Interactive Lineup Builder and run the multi-lineup Optimizer."""
    st.header(f"1. Player Pool & Constraints for **{template.contest_label}**")
    
    # ... (Player Pool Editor remains the same) ...

    # --- B. OPTIMIZATION CONTROLS (UPDATED) ---
    st.header("2. Find Optimal Lineups")
    
    col_n, col_slack = st.columns(2)
    
    with col_n:
        n_lineups = st.slider("Number of Lineups to Generate (N)", 
                              min_value=1, max_value=20, value=10, step=1,
                              help="The optimizer will find the N highest projected, unique lineups that meet all constraints.")
    
    with col_slack:
        slack = st.slider("Ownership Target Slack (Flexibility)", 
                          min_value=0, max_value=4, value=1, step=1,
                          help="Higher slack allows the optimizer to deviate more from the template's target player counts for each ownership bucket to find a higher projected score.")
    
    
    run_btn = st.button(f"✨ Generate Top {n_lineups} Lineups", use_container_width=True)
    
    if run_btn:
        final_df = st.session_state['edited_df'].copy()
        
        # Recalculate buckets based on potentially edited ownership
        final_df['bucket'] = final_df['own_proj'].apply(ownership_bucket)
        
        locked_player_ids = final_df[final_df['Lock'] == True]['player_id'].tolist()
        excluded_player_ids = final_df[final_df['Exclude'] == True]['player_id'].tolist()

        # Check for Lock/Exclude conflicts
        conflict = set(locked_player_ids) & set(excluded_player_ids)
        if conflict:
            st.error(f"❌ CONFLICT: Player(s) {', '.join(conflict)} are both locked and excluded.")
            return

        with st.spinner(f'Calculating top {n_lineups} optimal lineups...'):
            top_lineups = generate_top_n_lineups(
                slate_df=final_df,
                template=template,
                n_lineups=n_lineups,
                bucket_slack=slack,
                locked_player_ids=locked_player_ids, 
                excluded_player_ids=excluded_player_ids, 
            )
        
        st.session_state['optimal_lineups_results'] = {
            'lineups': top_lineups, 
            'ran': True
        }
        
        st.success(f"✅ Optimization complete! Found {len(top_lineups)} unique lineups.")

    
    st.markdown("---")
    st.header(f"3. Top {n_lineups} Lineups")
    
    if st.session_state['optimal_lineups_results'].get('ran', False):
        display_multiple_lineups(slate_df, st.session_state['optimal_lineups_results']['lineups'])
        
        # Add download button for the generated list (Optional but good for multi-lineup tools)
        if st.session_state['optimal_lineups_results']['lineups']:
            # Prepare CSV for download
            # For brevity, this CSV preparation logic is omitted but would go here.
            st.download_button(
                label="⬇️ Download All Lineups (CSV)",
                data="Example CSV Data",
                file_name="top_lineups.csv",
                mime="text/csv",
                disabled=True, # Disable download placeholder for now
                help="Functionality to export the generated lineups in a usable format."
            )

    else:
        st.info("Select the number of lineups and click 'Generate Top N Lineups' above to run the multi-lineup builder.")


# ... (tab_contest_analyzer and main entry point remain the same, just changing the function call in tab_lineup_builder) ...
