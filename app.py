import streamlit as st
import pandas as pd
import pickle
import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="EduPredict | AI Analytics", page_icon="🎓", layout="wide")

# ==========================================
# ANIMATED BACKGROUND & DARK GLASSMORPHISM CSS
# ==========================================
# Yahan background mein ek cool data/progress animation GIF lagaya hai
page_bg_img = '''
<style>
/* Animated GIF Background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.pinimg.com/originals/66/b5/07/66b507d4b4ddc89e86ba7f7fb0661ff1.gif");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Transparent Header */
[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Dark Glassmorphism Container */
.block-container {
    background-color: rgba(15, 23, 42, 0.85) !important; /* Dark transparent blue-black */
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 3rem 2rem;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Stylish Buttons for Dark Theme */
.stButton>button {
    background: linear-gradient(90deg, #ff6e40 0%, #ff5252 100%);
    color: white !important; 
    border-radius: 8px; 
    padding: 10px 24px; 
    font-weight: bold;
    border: none;
    transition: 0.4s ease;
    box-shadow: 0 4px 15px rgba(255, 110, 64, 0.4);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 110, 64, 0.6);
}

/* Customizing Dataframes and inputs for dark mode */
[data-testid="stDataFrame"] {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ==========================================
# LOAD MODEL & INITIALIZE STATE
# ==========================================
@st.cache_resource
def load_model():
    with open('edupredict_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

if 'batch_data' not in st.session_state:
    st.session_state.batch_data = None

# ==========================================
# HEADER SECTION
# ==========================================
st.title("🎓 EduPredict Dashboard")
st.markdown("**AI-Powered Academic Progress & Performance Tracking**")
st.markdown("---")

# ==========================================
# TABS SETUP
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Predictor", 
    "📁 CSV Upload & Batch Data", 
    "📊 Visual Analytics", 
    "🖨️ Official Report"
])

# ==========================================
# TAB 1: DYNAMIC SINGLE STUDENT PREDICTOR
# ==========================================
with tab1:
    st.markdown("### 👨‍🎓 Predict for an Individual Student")
    
    current_year = st.selectbox("Select Student's Current Year of Study:", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    if current_year == "1st Year":
        label_prev, label_sub1, label_sub2 = "12th Grade Marks (%)", "1st Sem Maths Score", "1st Sem CS Score"
    elif current_year == "2nd Year":
        label_prev, label_sub1, label_sub2 = "1st Year Avg CGPA (Scale to 100)", "Data Structures Score", "Computer Arch. Score"
    elif current_year == "3rd Year":
        label_prev, label_sub1, label_sub2 = "2nd Year Avg CGPA (Scale to 100)", "OS / DBMS Score", "Computer Networks Score"
    else:
        label_prev, label_sub1, label_sub2 = "3rd Year Avg CGPA (Scale to 100)", "AI / ML Score", "Cloud Computing Score"

    with col1:
        prev_score = st.number_input(label_prev, min_value=0, max_value=100, value=75)
    with col2:
        sub1_score = st.number_input(label_sub1, min_value=0, max_value=100, value=70)
    with col3:
        sub2_score = st.number_input(label_sub2, min_value=0, max_value=100, value=80)
        
    if st.button("🔮 Generate AI Prediction", type="primary"):
        input_data = pd.DataFrame({'12th_Marks': [prev_score], 'Maths_Score': [sub1_score], 'CS_Score': [sub2_score]})
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.markdown("### 📈 Prediction Result:")
        if prediction == 0:
            st.error("🚨 **High Risk Alert:** The student is showing signs of **Poor** performance. Immediate counseling is recommended.")
        elif prediction == 1:
            st.warning("⚠️ **Average Performer:** The student is doing okay, but has a risk of falling behind. Continuous monitoring is advised.")
        else:
            st.success("🌟 **Excellent:** The student is on a solid track for **Good** progress! Keep encouraging them.")

# ==========================================
# TAB 2: CSV UPLOAD & BATCH PREDICTION
# ==========================================
with tab2:
    st.markdown("### 📂 Upload CSV or Enter Data Manually")
    
    uploaded_file = st.file_uploader("Upload Student Data (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.success("CSV file loaded successfully! You can edit the data below.")
    else:
        st.info("No CSV uploaded. You can type data directly into the table below.")
        df_input = pd.DataFrame(
            columns=['Student_ID', 'Student_Name', 'Previous_Score', 'Core_Sub_1', 'Core_Sub_2'],
            data=[
                ['S001', 'Amit Sharma', 65, 50, 55],
                ['S002', 'Priya Singh', 85, 90, 88],
                ['S003', 'Rahul Verma', 45, 40, 35],
                ['S004', 'Neha Gupta', 78, 82, 79]
            ]
        )
    
    edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)
    
    if st.button("⚙️ Process Batch Data", type="primary"):
        if not edited_df.empty:
            try:
                X_batch = edited_df[['Previous_Score', 'Core_Sub_1', 'Core_Sub_2']].rename(
                    columns={'Previous_Score': '12th_Marks', 'Core_Sub_1': 'Maths_Score', 'Core_Sub_2': 'CS_Score'}
                )
                predictions = model.predict(X_batch)
                
                status_map = {0: "Poor (At Risk) 🔴", 1: "Average 🟡", 2: "Good 🟢"}
                edited_df['Predicted_Status'] = [status_map[p] for p in predictions]
                
                st.session_state.batch_data = edited_df
                st.success("✅ Batch predictions generated! Head over to the Analytics or Report tab.")
            except KeyError:
                st.error("❌ Error: Make sure columns match: 'Previous_Score', 'Core_Sub_1', 'Core_Sub_2'.")
        else:
            st.warning("Please enter some data first.")

# ==========================================
# TAB 3: VISUAL ANALYTICS
# ==========================================
with tab3:
    st.markdown("### 📊 Interactive Analytics Dashboard")
    
    if st.session_state.batch_data is not None:
        df_results = st.session_state.batch_data
        
        total_students = len(df_results)
        at_risk = len(df_results[df_results['Predicted_Status'] == 'Poor (At Risk) 🔴'])
        good_students = len(df_results[df_results['Predicted_Status'] == 'Good 🟢'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Total Students", total_students)
        col2.metric("🚨 Students At Risk", at_risk)
        col3.metric("⭐ Top Performers", good_students)
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Performance Distribution")
            status_counts = df_results['Predicted_Status'].value_counts()
            st.bar_chart(status_counts, color="#ff6e40")
            
        with c2:
            st.markdown("#### Search Individual Profile")
            student_list = df_results['Student_Name'].tolist()
            selected_student = st.selectbox("Select Student:", student_list)
            
            if selected_student:
                student_data = df_results[df_results['Student_Name'] == selected_student].iloc[0]
                st.write(f"**Current Prediction Status:** {student_data['Predicted_Status']}")
                st.line_chart(student_data[['Previous_Score', 'Core_Sub_1', 'Core_Sub_2']], color="#00e676")
    else:
        st.info("ℹ️ No data available. Please generate predictions in Tab 2 first.")

# ==========================================
# TAB 4: OFFICIAL HOD REPORT
# ==========================================
with tab4:
    st.markdown("### 📑 Official Academic Prediction Report")
    
    if st.session_state.batch_data is not None:
        df_results = st.session_state.batch_data
        
        st.markdown(f"**Date:** {datetime.datetime.now().strftime('%d %B %Y')}")
        st.markdown("**Department:** Computer Science & Engineering")
        st.markdown("> *Confidential: Early warning system report generated by EduPredict AI.*")
        
        st.dataframe(df_results, use_container_width=True)
        
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download HOD Report (CSV)",
            data=csv,
            file_name=f"EduPredict_Report_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("ℹ️ Please process batch data in Tab 2 to generate the report.")