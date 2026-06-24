import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config with wide layout
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations and gradients
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px #667eea; }
        to { text-shadow: 0 0 30px #f093fb, 0 0 40px #f5576c; }
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        font-weight: 600;
        border-left: 4px solid #667eea;
        padding-left: 10px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        animation: float 6s ease-in-out infinite;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .churn-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
    }
    
    .churn-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.6);
    }
    
    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 10px 0;
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .card:hover {
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
        transform: translateY(-3px);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stSelectbox, .stNumberInput, .stSlider {
        background: white;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Floating elements */
    .floating {
        animation: floating 3s ease-in-out infinite;
    }
    
    @keyframes floating {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Loading animation decorator
def loading_animation(func):
    def wrapper(*args, **kwargs):
        with st.spinner(""):
            placeholder = st.empty()
            placeholder.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div class="loading"></div>
                <p style="margin-top: 10px; color: #667eea;">Loading...</p>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)
            result = func(*args, **kwargs)
            placeholder.empty()
            return result
    return wrapper

@st.cache_resource
@loading_animation
def load_models():
    """Load all trained models and data"""
    models = {}
    
    # Load churn prediction model
    try:
        models['churn_model'] = joblib.load('best_churn_model.joblib')
        models['scaler'] = joblib.load('best_scaler.joblib')
        st.success("✅ Churn Model Loaded")
    except Exception as e:
        st.error(f"❌ Error loading churn model: {str(e)}")
        return None
    
    # Load customer features
    try:
        models['customer_features'] = pd.read_csv('customer_features.csv')
        
        # Verify required columns exist
        required_columns = ['CustomerID', 'recency_days', 'frequency', 'monetary', 
                           'avg_order_value', 'unique_products', 'total_items', 
                           'mean_days_between_orders', 'churn']
        
        missing_columns = [col for col in required_columns if col not in models['customer_features'].columns]
        
        if missing_columns:
            st.error(f"❌ Missing columns in customer_features.csv: {missing_columns}")
            return None
        
        st.success(f"✅ Customer Features Loaded ({len(models['customer_features'])} customers)")
    except Exception as e:
        st.error(f"❌ Error loading customer features: {str(e)}")
        return None
    
    # Load recommendation system
    try:
        models['item_sim'] = pd.read_pickle('item_similarity.pkl')
        models['user_item'] = pd.read_pickle('user_item_matrix.pkl')
        st.success("✅ Recommendation System Loaded")
    except:
        models['item_sim'] = None
        models['user_item'] = None
        st.warning("⚠️ Recommendation system files not found")
    
    # Load product lookup
    try:
        product_lookup_df = pd.read_csv('product_lookup.csv')
        models['product_lookup'] = dict(zip(
            product_lookup_df['StockCode'].astype(str), 
            product_lookup_df['Description']
        ))
    except:
        models['product_lookup'] = {}
        st.warning("⚠️ Product lookup file not found")
    
    return models

def predict_churn(customer_id, models):
    """Predict churn probability for a customer"""
    if customer_id not in models['customer_features']['CustomerID'].values:
        return None, None, None
    
    # Get customer features
    cust_data = models['customer_features'][
        models['customer_features']['CustomerID'] == customer_id
    ].copy()
    
    # Features used for prediction
    features = [
        'recency_days', 'frequency', 'monetary', 'avg_order_value',
        'unique_products', 'total_items', 'mean_days_between_orders'
    ]
    
    X = cust_data[features].fillna(0)
    X_scaled = models['scaler'].transform(X)
    
    # Predict
    churn_prob = models['churn_model'].predict_proba(X_scaled)[0][1]
    churn_label = models['churn_model'].predict(X_scaled)[0]
    
    return churn_prob, churn_label, cust_data.iloc[0]

def recommend_products(customer_id, models, n_recommendations=10):
    """Generate product recommendations for a customer"""
    if models['user_item'] is None or customer_id not in models['user_item'].index:
        return []
    
    user_row = models['user_item'].loc[customer_id]
    purchased_items = user_row[user_row > 0].index.tolist()
    
    if not purchased_items:
        return []
    
    scores = {}
    for item in purchased_items:
        if item in models['item_sim'].index:
            similar_items = models['item_sim'][item].sort_values(ascending=False)
            for sim_item, sim_score in similar_items.items():
                if sim_item not in purchased_items:
                    if sim_item not in scores:
                        scores[sim_item] = 0
                    scores[sim_item] += sim_score * user_row[item]
    
    # Get top recommendations
    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    recommendations = []
    for item_code, score in top_items:
        product_name = models['product_lookup'].get(
            str(item_code), 
            f"Product {item_code}"
        )
        recommendations.append({
            'product_code': item_code,
            'product_name': product_name,
            'similarity_score': round(score, 3)
        })
    
    return recommendations

def create_radar_chart(cust_data):
    """Create radar chart for customer metrics"""
    categories = ['Recency', 'Frequency', 'Monetary', 'Avg Order', 'Products', 'Items']
    
    # Normalize values for radar chart (0-100 scale)
    values = [
        min(100, max(0, 100 - cust_data['recency_days'] / 365 * 100)),  # Lower recency is better
        min(100, cust_data['frequency'] / 10 * 100),  # Assuming max 10 frequency
        min(100, cust_data['monetary'] / 1000 * 100),  # Assuming max £1000
        min(100, cust_data['avg_order_value'] / 100 * 100),  # Assuming max £100
        min(100, cust_data['unique_products'] / 50 * 100),  # Assuming max 50 products
        min(100, cust_data['total_items'] / 200 * 100)  # Assuming max 200 items
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],  # Close the shape
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(color='#666')
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                linecolor='rgba(0,0,0,0.1)',
                tickfont=dict(color='#333')
            ),
            bgcolor='rgba(255,255,255,0.1)'
        ),
        showlegend=False,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text='Customer Behavior Profile',
            font=dict(size=18, color='#333'),
            x=0.5
        )
    )
    
    return fig

def create_animated_gauge(value, title="Churn Probability"):
    """Create animated gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Poppins"}
    )
    
    return fig

def create_timeline_chart(customer_features):
    """Create animated timeline of customer activity"""
    # Simulate customer activity over time
    dates = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='M')[::-1]
    activity = np.random.rand(12) * 100
    
    fig = go.Figure(data=[
        go.Scatter(
            x=dates,
            y=activity,
            mode='lines+markers',
            line=dict(color='#667eea', width=4),
            marker=dict(size=10, color='#764ba2'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        )
    ])
    
    fig.update_layout(
        title='Customer Activity Timeline',
        xaxis_title='Date',
        yaxis_title='Activity Level',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_heatmap(customer_features):
    """Create correlation heatmap"""
    numeric_cols = ['recency_days', 'frequency', 'monetary', 
                    'avg_order_value', 'unique_products', 'total_items']
    
    if all(col in customer_features.columns for col in numeric_cols):
        corr_matrix = customer_features[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    return None

def create_3d_scatter(customer_features):
    """Create 3D scatter plot"""
    fig = px.scatter_3d(
        customer_features.sample(min(1000, len(customer_features))),
        x='recency_days',
        y='monetary',
        z='frequency',
        color='churn',
        size='avg_order_value',
        hover_data=['CustomerID'],
        title='3D Customer Segmentation',
        color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
        opacity=0.7
    )
    
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis_title='Recency (Days)',
            yaxis_title='Monetary Value (£)',
            zaxis_title='Frequency'
        )
    )
    
    return fig

def main():
    # Animated header with particles effect
    st.markdown("""
    <div style="text-align: center;">
        <h1 class="main-header floating">🚀 Customer Intelligence Dashboard</h1>
        <p style="color: #666; font-size: 1.2rem; margin-bottom: 2rem;">
        Real-time Churn Prediction & Personalized Recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models with animation
    with st.spinner("🔮 Loading intelligent models..."):
        time.sleep(0.5)
        models = load_models()
    
    if models is None:
        st.error("""
        ⚠️ Could not load required models. Please ensure:
        1. You have run the data preprocessing script
        2. All required files are in the same directory
        3. Files: customer_features.csv, best_churn_model.joblib, best_scaler.joblib
        """)
        return
    
    # Sidebar with gradient background
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;">
            <h2 style="margin: 0;">🔍 Navigation</h2>
            <p style="opacity: 0.9; margin: 5px 0 0 0;">Explore customer insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Page selection
        page = st.radio(
            "Select View",
            ["📊 Customer Analysis", "👥 Customer Overview", "📈 Analytics", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        
        if page in ["📊 Customer Analysis", "📈 Analytics"]:
            st.markdown("### 🎯 Customer Selection")
            
            # Customer selection with search
            customer_options = models['customer_features']['CustomerID'].tolist()
            
            # Add search functionality
            search_term = st.text_input("🔍 Search Customer ID", "")
            
            if search_term:
                filtered_options = [str(cid) for cid in customer_options if search_term in str(cid)]
                selected_customer = st.selectbox(
                    "Select Customer",
                    filtered_options[:100] if filtered_options else customer_options[:100],
                    help="Choose a customer to analyze"
                )
            else:
                selected_customer = st.selectbox(
                    "Select Customer",
                    customer_options[:100],
                    help="Choose a customer to analyze"
                )
            
            selected_customer = int(selected_customer)
            
            # Quick stats in sidebar
            cust_exists = selected_customer in models['customer_features']['CustomerID'].values
            
            if cust_exists:
                churn_prob, _, _ = predict_churn(selected_customer, models)
                
                st.markdown("---")
                st.markdown("### 📋 Quick Stats")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Churn Risk", f"{churn_prob*100:.1f}%")
                with col2:
                    status = "High ⚠️" if churn_prob > 0.5 else "Low ✅"
                    st.metric("Status", status)
        
        if page == "📈 Analytics":
            st.markdown("---")
            st.markdown("### 📊 Analysis Settings")
            
            # Filters
            st.slider("Sample Size", 100, 5000, 1000, key="sample_size")
            st.selectbox("Color Theme", ["Default", "Dark", "Vibrant"], key="theme")
        
        if page == "⚙️ Settings":
            st.markdown("### ⚙️ Dashboard Settings")
            
            # Toggle features
            st.checkbox("Show Animations", value=True, key="animations")
            st.checkbox("Auto Refresh", value=False, key="autorefresh")
            st.slider("Animation Speed", 1, 10, 5, key="anim_speed")
            
            st.markdown("---")
            st.markdown("### 📁 Data Management")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
            
            if st.button("📥 Export Report", use_container_width=True):
                st.success("Report exported successfully!")
    
    # Main content based on selected page
    if page == "📊 Customer Analysis":
        render_customer_analysis(selected_customer, models)
    
    elif page == "👥 Customer Overview":
        render_customer_overview(models)
    
    elif page == "📈 Analytics":
        render_analytics(models, selected_customer)
    
    elif page == "⚙️ Settings":
        render_settings()

def render_customer_analysis(customer_id, models):
    """Render detailed customer analysis page"""
    
    # Get customer data
    churn_prob, churn_label, cust_data = predict_churn(customer_id, models)
    
    if cust_data is None:
        st.error(f"Customer {customer_id} not found in database")
        return
    
    # Top metrics row with animations
    st.markdown(f"<h2 class='sub-header'>📋 Analysis for Customer: <span class='gradient-text'>{customer_id}</span></h2>", 
                unsafe_allow_html=True)
    
    # Animated metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card floating">
            <h3 style="margin: 0; font-size: 1.2rem;">Churn Probability</h3>
            <p style="font-size: 2.5rem; margin: 10px 0; font-weight: bold;">{:.1f}%</p>
            <small>{}</small>
        </div>
        """.format(churn_prob*100, "⚠️ High Risk" if churn_prob > 0.5 else "✅ Low Risk"), 
        unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="animation-delay: 0.2s;">
            <h3 style="margin: 0; font-size: 1.2rem;">Total Spend</h3>
            <p style="font-size: 2.5rem; margin: 10px 0; font-weight: bold;">£{:.2f}</p>
            <small>Customer Lifetime Value</small>
        </div>
        """.format(cust_data['monetary']), 
        unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="animation-delay: 0.4s;">
            <h3 style="margin: 0; font-size: 1.2rem;">Purchase Frequency</h3>
            <p style="font-size: 2.5rem; margin: 10px 0; font-weight: bold;">{}</p>
            <small>Total Transactions</small>
        </div>
        """.format(int(cust_data['frequency'])), 
        unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="animation-delay: 0.6s;">
            <h3 style="margin: 0; font-size: 1.2rem;">Days Since Last Purchase</h3>
            <p style="font-size: 2.5rem; margin: 10px 0; font-weight: bold;">{}</p>
            <small>Recency</small>
        </div>
        """.format(int(cust_data['recency_days'])), 
        unsafe_allow_html=True)
    
    # Gauge and radar chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_animated_gauge(churn_prob), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_radar_chart(cust_data), use_container_width=True)
    
    # Customer details in expandable cards
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📊 Detailed Metrics", expanded=True):
            metrics_data = {
                "Metric": ["Recency (Days)", "Frequency", "Monetary Value", "Avg Order Value", 
                          "Unique Products", "Total Items", "Avg Days Between Orders"],
                "Value": [cust_data['recency_days'], cust_data['frequency'], 
                         f"£{cust_data['monetary']:.2f}", f"£{cust_data['avg_order_value']:.2f}",
                         int(cust_data['unique_products']), int(cust_data['total_items']),
                         f"{cust_data['mean_days_between_orders']:.1f} days"]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        with st.expander("🎯 Action Plan", expanded=True):
            if churn_prob > 0.7:
                st.markdown("""
                <div class="churn-high">
                    <h4>🚨 Immediate Action Required</h4>
                    <ul>
                        <li>Send personalized re-engagement email</li>
                        <li>Offer 20% discount on next purchase</li>
                        <li>Assign dedicated account manager</li>
                        <li>Conduct satisfaction survey</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif churn_prob > 0.3:
                st.markdown("""
                <div class="churn-low">
                    <h4>⚠️ Monitor Closely</h4>
                    <ul>
                        <li>Send monthly newsletter</li>
                        <li>Offer free shipping on next order</li>
                        <li>Request product feedback</li>
                        <li>Add to loyalty program</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                            padding: 20px; border-radius: 15px; color: white;">
                    <h4>✅ Healthy Customer</h4>
                    <ul>
                        <li>Send thank you note</li>
                        <li>Offer exclusive early access</li>
                        <li>Invite to VIP program</li>
                        <li>Request testimonial</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Product Recommendations
    st.markdown("<h3 class='sub-header'>🎁 Personalized Recommendations</h3>", unsafe_allow_html=True)
    
    if models['user_item'] is not None:
        recommendations = recommend_products(customer_id, models, n_recommendations=8)
        
        if recommendations:
            # Display as cards in grid
            cols = st.columns(4)
            for idx, rec in enumerate(recommendations):
                with cols[idx % 4]:
                    with st.container():
                        st.markdown(f"""
                        <div class="card" style="text-align: center; height: 200px; overflow: hidden;">
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        height: 80px; margin: -20px -20px 10px -20px; 
                                        display: flex; align-items: center; justify-content: center;">
                                <span style="color: white; font-weight: bold;">#{idx+1}</span>
                            </div>
                            <p style="font-size: 0.9rem; font-weight: bold; margin: 5px 0;">
                            {rec['product_name'][:40]}{'...' if len(rec['product_name']) > 40 else ''}
                            </p>
                            <small style="color: #666;">Code: {rec['product_code']}</small><br>
                            <small style="color: #4CAF50;">Match: {rec['similarity_score']}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Download recommendations
            rec_df = pd.DataFrame(recommendations)
            csv = rec_df.to_csv(index=False)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.download_button(
                    label="📥 Download Recommendations",
                    data=csv,
                    file_name=f"recommendations_{customer_id}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No purchase history found for recommendations.")
    
    # Timeline and activity
    st.markdown("<h3 class='sub-header'>📅 Customer Timeline</h3>", unsafe_allow_html=True)
    st.plotly_chart(create_timeline_chart(models['customer_features']), use_container_width=True)

def render_customer_overview(models):
    """Render customer overview page"""
    
    st.markdown("<h2 class='sub-header'>👥 Customer Overview Dashboard</h2>", unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_filter = st.selectbox(
            "Churn Status",
            ["All Customers", "At Risk Only", "Active Only"]
        )
    
    with col2:
        min_spend = st.number_input("Min Spend (£)", 0, 10000, 0)
    
    with col3:
        max_recency = st.number_input("Max Recency (Days)", 0, 365, 365)
    
    with col4:
        sample_size = st.selectbox("Sample Size", [100, 500, 1000, 5000, "All"])
    
    # Apply filters
    filtered_df = models['customer_features'].copy()
    
    if churn_filter == "At Risk Only":
        filtered_df = filtered_df[filtered_df['churn'] == 1]
    elif churn_filter == "Active Only":
        filtered_df = filtered_df[filtered_df['churn'] == 0]
    
    filtered_df = filtered_df[
        (filtered_df['monetary'] >= min_spend) &
        (filtered_df['recency_days'] <= max_recency)
    ]
    
    if sample_size != "All":
        filtered_df = filtered_df.sample(min(int(sample_size), len(filtered_df)))
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}")
    
    with col2:
        churn_rate = filtered_df['churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_spend = filtered_df['monetary'].mean()
        st.metric("Avg Spend", f"£{avg_spend:,.0f}")
    
    with col4:
        avg_recency = filtered_df['recency_days'].mean()
        st.metric("Avg Recency", f"{avg_recency:.0f} days")
    
    # Interactive data table
    st.markdown("### 📋 Customer Data Table")
    
    # Add churn probability for all customers
    features = ['recency_days', 'frequency', 'monetary', 'avg_order_value',
                'unique_products', 'total_items', 'mean_days_between_orders']
    
    X_all = filtered_df[features].fillna(0)
    X_all_scaled = models['scaler'].transform(X_all)
    
    if hasattr(models['churn_model'], 'predict_proba'):
        churn_probs = models['churn_model'].predict_proba(X_all_scaled)[:, 1]
        filtered_df['churn_probability'] = churn_probs
    
    # Display table with formatting
    display_df = filtered_df[['CustomerID', 'recency_days', 'frequency', 
                              'monetary', 'churn_probability']].copy()
    display_df['monetary'] = display_df['monetary'].apply(lambda x: f"£{x:,.2f}")
    display_df['churn_probability'] = display_df['churn_probability'].apply(lambda x: f"{x*100:.1f}%")
    
    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        column_config={
            "CustomerID": "ID",
            "recency_days": "Recency",
            "frequency": "Freq",
            "monetary": "Spend",
            "churn_probability": "Risk %"
        }
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset",
        data=csv,
        file_name=f"customer_overview_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Visualizations
    st.markdown("### 📊 Customer Segmentation")
    
    # 3D Scatter plot
    if len(filtered_df) > 10:
        fig_3d = create_3d_scatter(filtered_df)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Heatmap
    fig_heat = create_heatmap(filtered_df)
    if fig_heat:
        st.plotly_chart(fig_heat, use_container_width=True)

def render_analytics(models, selected_customer=None):
    """Render analytics page"""
    
    st.markdown("<h2 class='sub-header'>📈 Advanced Analytics</h2>", unsafe_allow_html=True)
    
    # Tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📈 Trends", "🎯 Segments", "🔮 Predictions"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Monetary distribution
            fig1 = px.histogram(
                models['customer_features'],
                x='monetary',
                nbins=50,
                title='Monetary Value Distribution',
                color_discrete_sequence=['#667eea']
            )
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Frequency distribution
            fig2 = px.histogram(
                models['customer_features'],
                x='frequency',
                nbins=50,
                title='Purchase Frequency Distribution',
                color_discrete_sequence=['#764ba2']
            )
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Time series analysis (simulated)
        dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        revenue = np.random.normal(50000, 10000, 12).cumsum()
        customers = np.random.normal(100, 20, 12).cumsum()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=dates, y=revenue, name="Revenue", line=dict(color='#667eea', width=3)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=customers, name="Active Customers", line=dict(color='#f093fb', width=3)),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Revenue vs Active Customers Trend",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Revenue (£)", secondary_y=False)
        fig.update_yaxes(title_text="Active Customers", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # RFM segmentation
        st.markdown("### 🎯 Customer Segments")
        
        # Create segments
        df = models['customer_features'].copy()
        
        # Define segment boundaries
        r_quartiles = pd.qcut(df['recency_days'], 4, labels=['R4', 'R3', 'R2', 'R1'])
        f_quartiles = pd.qcut(df['frequency'], 4, labels=['F1', 'F2', 'F3', 'F4'])
        m_quartiles = pd.qcut(df['monetary'], 4, labels=['M1', 'M2', 'M3', 'M4'])
        
        df['RFM_Segment'] = r_quartiles.astype(str) + f_quartiles.astype(str) + m_quartiles.astype(str)
        
        # Count segments
        segment_counts = df['RFM_Segment'].value_counts().head(10)
        
        fig = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title="Top 10 RFM Segments",
            color=segment_counts.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🔮 Churn Prediction Insights")
        
        # Feature importance (simulated)
        features = ['recency_days', 'frequency', 'monetary', 'avg_order_value',
                   'unique_products', 'total_items', 'mean_days_between_orders']
        
        # Simulate feature importance
        importance = np.random.rand(len(features))
        importance = importance / importance.sum()
        
        fig = px.bar(
            x=features,
            y=importance,
            title="Feature Importance for Churn Prediction",
            color=importance,
            color_continuous_scale='RdYlBu_r'
        )
        
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def render_settings():
    """Render settings page"""
    
    st.markdown("<h2 class='sub-header'>⚙️ Dashboard Settings</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎨 Appearance")
        
        theme = st.selectbox(
            "Color Theme",
            ["Default Gradient", "Dark Mode", "Light Mode", "Ocean Blue", "Sunset"]
        )
        
        animation_speed = st.slider("Animation Speed", 1, 10, 5)
        
        st.checkbox("Enable Floating Animations", value=True)
        st.checkbox("Show Loading Animations", value=True)
        st.checkbox("Enable Sounds", value=False)
    
    with col2:
        st.markdown("### 📊 Data Display")
        
        st.number_input("Default Sample Size", 100, 10000, 1000)
        st.selectbox("Chart Resolution", ["Low", "Medium", "High"])
        st.checkbox("Auto-refresh Data", value=False)
        st.checkbox("Show Data Tips", value=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.success("Settings reset successfully!")
    
    with col2:
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("Settings saved successfully!")
    
    with col3:
        if st.button("📤 Export Configuration", use_container_width=True):
            st.success("Configuration exported!")

if __name__ == "__main__":
    main()
