# ---- USER DROPDOWN ----
username_list = sorted(lineup_summary["username"].unique())
selected_user = st.selectbox("Select User", username_list)

# All lineups for that user (summary-level)
user_group = lineup_summary[lineup_summary["username"] == selected_user].copy()
user_group = user_group.sort_values("rank")

st.subheader("User Lineups (Summary)")
st.dataframe(user_group)

# ---- NEW: MATRIX VIEW OF ALL LINEUPS FOR THIS USER ----
st.subheader("User Lineups (Players in Columns)")

# Filter long_df to this user only
user_long = long_df[long_df["username"] == selected_user].copy()

# Pivot so each lineup is one row, each position (PG/SG/…) is a column
user_matrix = (
    user_long
    .pivot_table(
        index=["entry_id", "rank", "points"],
        columns="pos_slot",
        values="player_name",
        aggfunc="first"
    )
    .reset_index()
    .sort_values("rank")
)

# Streamlit doesn't like the MultiIndex column from pivot, so flatten it
user_matrix.columns.name = None  # remove "pos_slot" header name

st.dataframe(user_matrix)

# ---- EXISTING: SINGLE LINEUP DETAIL VIEW ----
st.subheader("Single Lineup Detail")

selected_entry = st.selectbox("Select Entry ID", user_group["entry_id"])
detail = get_lineup_detail(long_df, selected_entry)

st.table(detail)
